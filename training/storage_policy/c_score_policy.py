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

    def __init__(self, max_size: int, mode: str = 'random', 
                    min_bucket: float = 0.9, perct_caws: str = 'median'):
        """
        :param max_size:
        """
        super().__init__(max_size)
        self._buffer_weights = torch.zeros(0)
        self.mode = mode
        self.min_bucket = min_bucket
        self.perct_caws = perct_caws

    def update(self, strategy: 'SupervisedTemplate', **kwargs):
        """ Update buffer. """
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset, new_weights = None):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        if new_weights is None or self.mode == 'random':
            new_weights = torch.rand(len(new_data))
        elif type(new_weights) == list:
            new_weights = torch.tensor(new_weights)

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        self.cat_data = AvalancheConcatDataset([new_data, self.buffer])

        if self.mode == 'lower':
            sorted_weights, sorted_idxs = cat_weights.sort(descending=False)
        elif self.mode == 'upper' or self.mode == 'caws':
            sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        else: # random
            sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        
        if self.mode == 'caws-perc':
            if self.perct_caws == 'median':
                threshold = torch.median(sorted_weights)
            elif self.perct_caws == 'mean':
                threshold = torch.mean(sorted_weights)
            else:
                threshold = -1
                mem_size = int(min(len(sorted_idxs),  self.max_size*2))
            
            if threshold > -1:
                if self.max_size > ( sorted_weights > threshold ).sum():
                    self.buffer_idxs = sorted_idxs[:self.max_size]
                else:
                    self.buffer_idxs = random.sample( sorted_idxs[ sorted_weights > threshold ].tolist(), self.max_size)
            else:
                self.buffer_idxs = random.sample( sorted_idxs[:mem_size].tolist(), self.max_size)

        elif self.mode == 'caws':
            if self.max_size > ( sorted_weights > self.min_bucket ).sum():
                self.buffer_idxs = sorted_idxs[:self.max_size]
            else:
                self.buffer_idxs = random.sample( sorted_idxs[ sorted_weights > self.min_bucket ].tolist(), self.max_size)
        elif self.mode == 'cobs':
            self.buffer_idxs = []
            b_past = 0
            size_bucket = self.max_size // 10
            residual = 0
            for b in np.linspace(0.1, 1 , 10):
                index_bucket = ( sorted_weights >= b_past ) * ( sorted_weights <= b)
                if index_bucket.sum() < size_bucket + residual:
                    self.buffer_idxs += random.sample( sorted_idxs[ index_bucket ].tolist(), index_bucket.sum())
                    residual += size_bucket - index_bucket.sum()
                else:
                    self.buffer_idxs += random.sample( sorted_idxs[ index_bucket ].tolist(), size_bucket + residual)
                    residual = 0
                b_past = b
            
            if len(self.buffer_idxs) < self.max_size:
                self.buffer_idxs += random.sample( sorted_idxs.tolist(), self.max_size - len(self.buffer_idxs))

            random.shuffle(self.buffer_idxs)
        else:
            self.buffer_idxs = sorted_idxs[:self.max_size]

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


class CScoreBuffer(BalancedExemplarsBuffer):
    def __init__(self, max_size: int, adaptive_size: bool = True,
                 total_num_classes: int = None, name_dataset: str = 'cifar10',
                 mode: str = 'random', min_bucket: float = 0.9,
                 use_proxy: bool = False, num_proxy: int = 1,
                 num_neighbours: int = 5, perct_caws: str = 'median',
                 true_labels: bool = True, save_score: bool = False):

        if not adaptive_size:
            assert total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()
        self.mode = mode
        self.min_bucket = min_bucket
        self.perct_caws = perct_caws
        self.true_labels = true_labels
        self.save_score = save_score

        self.num_neighbours = num_neighbours
        self.num_proxy = num_proxy
        self.use_proxy = use_proxy
        self.name_dataset = name_dataset

        if name_dataset == 'cifar10' or name_dataset == 'cifar100':
            if use_proxy and num_proxy == 1:
                self.scores = np.load(f"c_score/{name_dataset}/score_{num_neighbours}_proxy.npy")
            else:
                self.scores = np.load(f"c_score/{name_dataset}/scores.npy")
        elif name_dataset == 'mnist':
            data = torch.load('c_score/mnist_with_c_score.pth')
            self.scores = data['train_scores']
        elif name_dataset == 'imagenet':
            self.scores = np.load(f"c_score/{name_dataset}/scores_train.npy")
        elif name_dataset == 'tiny_imagenet':
            if use_proxy and num_proxy == 1:
                scores = np.load(f"c_score/{name_dataset}/score_{num_neighbours}_proxy.npy")
                self.scores = {}
                for elem in scores:
                    p = '/'.join(elem[1].split('/')[-5:]).strip()
                    self.scores[p] = float(elem[0].split('(')[1][:-1])
            else:
                scores = np.load(f"c_score/{name_dataset}/tinyimg_train.npy")
                self.scores = { elem[1]: float(elem[0]) for elem in scores }                
        elif name_dataset == 'core50':
            self.scores = {}
        else:
            assert False, "Dataset {} not found".format(name_dataset)

        if self.num_proxy == 2:
            self.pre_model = resnet50(pretrained=True)

        if self.save_score:
            self.scores_saved = {}

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset

        if self.use_proxy:
            if self.num_proxy == 1:
                if self.name_dataset =='tiny_imagenet':
                    scores = self.scores
                else:
                    scores = np.array([ 1-s for s in self.scores ])
            elif self.num_proxy == 2:
                new_subset = new_data.add_transforms_group('feats', _default_imgenet_val_transform, None) 
                new_subset = new_subset.with_transforms('feats')
                dataloader = DataLoader(new_subset, batch_size=128,
                            shuffle=False, num_workers=2)
                feats, labels = self.get_features(self.pre_model, dataloader, strategy.device)
                scores = self.get_label_uniformity(feats, labels)
                new_data = new_subset.with_transforms('train')
            else:
                dataloader = DataLoader(new_data, batch_size=128,
                            shuffle=False, num_workers=2)
                feats, labels = self.get_features(copy.deepcopy(strategy.model), dataloader, strategy.device)
                scores = self.get_label_uniformity(feats, labels)
            
            if self.save_score:
                self.scores_saved[len(self.scores_saved.keys())+1] = (scores,labels)
        else:
            scores = self.scores

        # Get sample idxs per class
        cl_idxs = {}
        cl_score = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
                cl_score[target] = []
            cl_idxs[target].append(idx)
            if self.use_proxy:
                if self.num_proxy == 1:
                    if self.name_dataset == 'tiny_imagenet':
                        p = '/'.join(new_data._original_dataset._dataset.samples[idx][0].split('/')[-5:]).strip()
                        cl_score[target].append(self.scores[p])
                    else:
                        cl_score[target].append(scores[new_data._indices[idx]])
                else:
                    cl_score[target].append(scores[idx])
            else:
                if self.name_dataset == 'tiny_imagenet':
                    p = '/'.join(new_data._original_dataset._dataset.samples[idx][0].split('/')[-5:])
                    cl_score[target].append(scores[p])
                else:
                    cl_score[target].append(scores[new_data._indices[idx]])
        
        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = AvalancheSubset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c, cl_score[class_id])
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll, self.mode, self.min_bucket, self.perct_caws)
                new_buffer.update_from_dataset(new_data_c, cl_score[class_id])
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy,
                                                class_to_len[class_id])

    def get_features(self, model, dataloader, device):
        model = model.to(device)
        features = []
        labels = []
        if self.true_labels:
            classifier = None
        else:
            classifier = copy.deepcopy(model.classifier)
        model.classifier = nn.Identity()
        with torch.no_grad():
            for batch in dataloader:
                imgs = batch[0].to(device)
                feats = model(imgs)
                features.append(feats.detach().cpu())
                if self.true_labels:
                    labels.append(batch[1])
                else:
                    out = classifier(feats)
                    _, predicted = torch.max(out.data, 1)
                    labels.append(predicted.detach().cpu())
                #labels.append(batch[1])

        feats_list = torch.cat(features)
        target_list = torch.cat(labels)
        return feats_list, target_list
    
    def get_label_uniformity(self, feats, labels):        
        # return np.array(l_uniformity)
        a_norm = feats / feats.norm(dim=1)[:, None]
        # b_norm = feats[0:5] / feats[0:5].norm(dim=1)[:, None]
        l_uniformity = []
        bs = 256
        buffer_size = feats.size(0)
        for i in range(math.ceil(buffer_size / bs)):
            buffer = a_norm[i*bs : i*bs+bs]
            buffer_labels = labels[i*bs : i*bs+bs]
            res = torch.mm(buffer, a_norm.transpose(0,1))
            idxs_top = res.sort(descending=True, dim=1)[1][:, 1: self.num_neighbours + 1]
            l_uniformity.extend([ (labels[idxs_top[j]] == l).sum() / self.num_neighbours for j, l in enumerate(buffer_labels) ])
            
        return np.array(l_uniformity)
