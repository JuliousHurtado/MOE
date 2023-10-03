import torch
from torch.utils.data import DataLoader
import random

import itertools
import torch.nn as nn

from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.benchmarks.utils import (
    AvalancheConcatDataset,
    AvalancheDataset,
    AvalancheSubset,
)
from avalanche.training.storage_policy import BalancedExemplarsBuffer, ExemplarsBuffer



class ReservoirSamplingBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""
    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)


    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
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
        cat_data = AvalancheConcatDataset([new_data, self.buffer])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = AvalancheSubset(cat_data, buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = AvalancheSubset(self.buffer, torch.arange(self.max_size))
        self._buffer_weights = self._buffer_weights[: self.max_size]



class AllRightBuffer(BalancedExemplarsBuffer):
    """ Buffer updated with reservoir sampling. """

    def __init__(self, max_size: int, adaptive_size: bool = True,):
        super().__init__(max_size, adaptive_size)

        self.x_memory = []
        self.y_memory = []
        self.order = []

        self.seen_classes = set()
        self.loss = nn.CrossEntropyLoss(reduction='none')

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

        for class_id, c_idxs in cl_idxs.items():
            ll = class_to_len[class_id]
            cd = AvalancheSubset(new_data, indices=c_idxs)
            score, new_index = self.get_score(strategy, cd, class_id, ll)
            cd = AvalancheSubset(cd, indices=new_index)

            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(cd, score)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(cd, score)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, _ in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy,
                                                class_to_len[class_id])


    def get_score(self, strategy: "SupervisedTemplate", dataset: AvalancheDataset, class_id, ll):
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        scores = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(strategy.device)
                y = batch[1].to(strategy.device)

                outs = strategy.model(x)

                _, predicted = torch.max(outs.data, 1)

                loss = self.loss(outs, y)

                scores.extend( [ l.item() for l in loss ] )
                labels.extend(predicted)
        
        labels = torch.Tensor(labels)
        scores = torch.Tensor(scores)
        scores = (1 - scores/scores.max())

        b_mask = ( labels == class_id )
        while b_mask.sum() <= ll:
            p = random.randint(0, b_mask.size(0))
            b_mask[p] = True

            if ll > b_mask.size(0) or b_mask.size(0) == b_mask.sum():
                break 

        print(b_mask.sum())

        return scores[ b_mask ], b_mask
