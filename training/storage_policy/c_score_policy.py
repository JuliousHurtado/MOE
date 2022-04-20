import torch
import numpy as np
import random

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset,\
     AvalancheConcatDataset
from avalanche.training.storage_policy import BalancedExemplarsBuffer, ExemplarsBuffer
from avalanche.training.strategies import BaseStrategy

from datasets.load_c_score import ImagenetCScore, CIFARIdx

class ReservoirSamplingBuffer(ExemplarsBuffer):
    """ Buffer updated with reservoir sampling. """

    def __init__(self, max_size: int, mode: str = 'random', 
                       mix_upper: float = 0.5):
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
        self.mode = mode
        self.mix_upper = mix_upper

    def update(self, strategy: 'BaseStrategy', **kwargs):
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
        elif self.mode == 'upper':
            sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        else: # random
            sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        
        if self.mode == 'mix':
            num_upper = int(self.max_size*self.mix_upper)
            upper_list = random.sample(sorted_idxs[:self.max_size].tolist(), num_upper)
            lower_list = random.sample(sorted_idxs[self.max_size:].tolist(), self.max_size - num_upper)
            self.buffer_idxs = upper_list + lower_list
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
                 mode: str = 'random', mix_upper: float = 0.5):

        if not adaptive_size:
            assert total_num_classes > 0, \
                """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()
        self.mode = mode
        self.mix_upper = mix_upper

        if name_dataset == 'cifar10' or name_dataset == 'cifar100':
            self.scores = np.load(f"c_score/{name_dataset}/scores.npy")
        else:
            assert False, "Dataset {} not found".format(name_dataset)

    def update(self, strategy: "BaseStrategy", **kwargs):
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        cl_score = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
                cl_score[target] = []
            cl_idxs[target].append(idx)
            cl_score[target].append(self.scores[new_data._indices[idx]])
        
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
                new_buffer = ReservoirSamplingBuffer(ll, self.mode, self.mix_upper)
                new_buffer.update_from_dataset(new_data_c, cl_score[class_id])
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy,
                                                class_to_len[class_id])