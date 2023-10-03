import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torchvision.transforms as transforms

import numpy as np
import random
import copy
import math

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset,\
     AvalancheConcatDataset
from avalanche.training.storage_policy import BalancedExemplarsBuffer, ExemplarsBuffer
from avalanche.training.templates.supervised import SupervisedTemplate

from datasets.load_c_score import ImagenetCScore, CIFARIdx

_default_imgenet_val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

class ReservoirSamplingBuffer(ExemplarsBuffer):
    """ Buffer updated with reservoir sampling. """

    def __init__(self, max_size: int, mode: str = 'random'):
        """
        :param max_size:
        """
        super().__init__(max_size)
        self._buffer_weights = torch.zeros(0)
        self.mode = mode

    def update(self, strategy: 'SupervisedTemplate', **kwargs):
        """ Update buffer. """
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset, new_weights = None):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        if new_weights is None:
            new_weights = torch.rand(len(new_data))
        elif type(new_weights) == list:
            new_weights = torch.tensor(new_weights)

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        self.cat_data = AvalancheConcatDataset([new_data, self.buffer])

        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        if self.mode == 'top':
            self.buffer_idxs = sorted_idxs[:self.max_size]
        else:
            self.buffer_idxs = random.sample( sorted_idxs.tolist(), self.max_size)
        
        self.buffer = AvalancheSubset(self.cat_data, self.buffer_idxs)
        self._buffer_weights = sorted_weights[:self.max_size]

    def resize(self, strategy, new_size):
        """ Update the maximum size of the buffer. """
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer_idxs = self.buffer_idxs[:self.max_size]
        self.buffer = AvalancheSubset(self.cat_data, self.buffer_idxs)
        self._buffer_weights = self._buffer_weights[:self.max_size]


class MOEBuffer(BalancedExemplarsBuffer):
    def __init__(self, max_size: int, adaptive_size: bool = True,
                 total_num_classes: int = None, num_proxy: int = 1,
                 num_neighbours: int = 5, rem_neighbours: int = 20,
                 mode: str = 'random', filter_moe: bool = True):

        if not adaptive_size:
            assert total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        self.num_neighbours = num_neighbours
        self.rem_neighbours = rem_neighbours
        self.num_proxy = num_proxy
        self.filter_mode = filter_moe
        self.mode = mode

        if self.num_proxy == 2:
            self.pre_model = resnet50(pretrained=True)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Update seen classes
        self.seen_classes.update(cl_idxs.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll


        if self.num_proxy == 2:
            new_subset = new_data.add_transforms_group('feats', _default_imgenet_val_transform, None) 
            new_subset = new_subset.with_transforms('feats')
            dataloader = DataLoader(new_subset, batch_size=512,
                                        shuffle=False, num_workers=2)
            feats, labels = self.get_features(self.pre_model, dataloader, strategy.device)
            new_data = new_subset.with_transforms('train')
        elif self.num_proxy == 3:
            dataloader = DataLoader(new_data, batch_size=512,
                            shuffle=False, num_workers=2)
            feats, labels = self.get_features(copy.deepcopy(strategy.model), dataloader, strategy.device)
        else:
            assert False, "It must be num_proxy 2 or 3"




        scores, dist_mm = self.get_represent_index(feats, labels)
        if self.filter_mode:
            index_dist = dist_mm.sort(descending=True, dim=1)[1]
            buffer, _ = self.get_buffer_list(scores, dist_mm, index_dist, labels, class_to_len)
        else:
            buffer = {}
            index_score = scores.sort(descending=True)[1]
            for k in cl_idxs.keys():
                buffer[k] = index_score[ labels == k ]


    
        for class_id, indx_clss in buffer.items():
            ll = class_to_len[class_id]
            new_data_c = AvalancheSubset(new_data, indices=indx_clss)
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c, scores[indx_clss])
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll, self.mode)
                new_buffer.update_from_dataset(new_data_c, scores[indx_clss])
                self.buffer_groups[class_id] = new_buffer
        
        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy,
                                                class_to_len[class_id])

    def get_features(self, model, dataloader, device):
        model = model.to(device)
        features = []
        labels = []
        model.fc = nn.Identity()
        with torch.no_grad():
            for batch in dataloader:
                imgs = batch[0].to(device)
                feats = model(imgs)
                features.append(feats.detach().cpu())
                labels.append(batch[1])

        feats_list = torch.cat(features)
        target_list = torch.cat(labels)
        return feats_list, target_list
    
    def get_represent_index(self, feats, labels, feats_t = None, labels_t = None):
        if feats_t is None:
            feats_t = copy.deepcopy(feats)
            labels_t = copy.deepcopy(labels)

        a_norm = feats / feats_t.norm(dim=1)[:, None]
        b_norm = feats_t / feats_t.norm(dim=1)[:, None]
        
        dist_mm = torch.zeros((feats.size(0),feats_t.size(0)))
        scores = []
        
        bs = 256
        buffer_size = feats.size(0)
        for i in range(math.ceil(buffer_size / bs)):
            buffer = a_norm[i*bs : i*bs+bs]
            buffer_labels = labels[i*bs : i*bs+bs]
            
            res = torch.mm(buffer, feats_t.transpose(0,1))
            dist_mm[i*bs : i*bs+bs, :] = res[:,:]
            
            idxs_top = res.sort(descending=True, dim=1)[1][:, 1: self.num_neighbours + 1]
            scores.extend([ (labels_t[idxs_top[j]] == l).sum() / self.num_neighbours for j, l in enumerate(buffer_labels) ])

        return torch.Tensor(scores), dist_mm

    def get_buffer_list(self, scores, dist_mm, index_dist, labels, class_to_len, buffer = {}, indet=0):
        reject = {}
        for idx in scores.sort(descending=True)[1]:
            idx = idx.item() + indet
            clss_idx = labels[idx].item()

            if clss_idx not in buffer:
                buffer[clss_idx] = []

            if idx >= len(scores): # and len(buffer[clss_idx]) > class_to_len[clss_idx]*3:
                continue
    
            try:
                reject[idx]
            except:
                buffer[clss_idx].append(idx)

                for can_reject in index_dist[idx][1:self.rem_neighbours]:
                    p = random.random()
                    if labels[can_reject].item() == clss_idx and p < 0.6:
                        reject[can_reject.item()] = 0
        
        for k,v in buffer.items():
            if len(v) < class_to_len[k]:
                buffer, reject = self.get_buffer_list(scores, dist_mm, index_dist, labels, class_to_len, buffer, indet=1)

        return buffer, list(reject.keys())
