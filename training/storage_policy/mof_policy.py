import torch
from torch.utils.data import DataLoader

import itertools
from math import ceil

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

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = AvalancheConcatDataset([new_data, self.buffer])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=False)

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



class MeanOfFeaturesBuffer(BalancedExemplarsBuffer):
    """ Buffer updated with reservoir sampling. """

    def __init__(self, max_size: int, adaptive_size: bool = True,):
        super().__init__(max_size, adaptive_size)

        self.x_memory = []
        self.y_memory = []
        self.order = []
        self.adaptive_size = True

        self.seen_classes = set()

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
            score = self.get_score(strategy, cd, ll)

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


    def get_score(self, strategy: "SupervisedTemplate", dataset: AvalancheDataset, ll: int):
        class_patterns, _, _ = next(
                iter(DataLoader(dataset.eval(), batch_size=len(dataset)))
            )
        class_patterns = class_patterns.to(strategy.device)

        with torch.no_grad():
            mapped_prototypes = strategy.model.feature_extractor(
                    class_patterns
                ).detach()
        D = mapped_prototypes.T
        D = D / torch.norm(D, dim=0)

        mu = torch.mean(D, dim=1)
        order = torch.zeros(class_patterns.shape[0])
        w_t = mu

        i, added, selected = 0, 0, []
        while not added == ll and i < 1000:
            tmp_t = torch.mm(w_t.unsqueeze(0), D)
            ind_max = torch.argmax(tmp_t)

            if ind_max not in selected:
                order[ind_max] = 1 + added
                added += 1
                selected.append(ind_max.item())

            w_t = w_t + mu - D[:, ind_max]
            i += 1

        order[ order == 0 ] = order.max() + 1

        return order
