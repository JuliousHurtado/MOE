from typing import Dict, Sequence
from itertools import chain

import torch
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.utils import AvalancheDataset


def _default_collate_mbatches_fn(mbatches):
    """Combines multiple mini-batches together.
    Concatenates each tensor in the mini-batches along dimension 0 (usually this
    is the batch size).
    :param mbatches: sequence of mini-batches.
    :return: a single mini-batch
    """
    batch = []
    for i in range(len(mbatches[0])):
        t = torch.cat([el[i] for el in mbatches], dim=0)
        batch.append(t)
    return batch


class ReplayDataLoader:
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(
        self,
        data: AvalancheDataset,
        memory: AvalancheDataset = None,
        oversample_small_tasks: bool = False,
        collate_mbatches=_default_collate_mbatches_fn,
        batch_size: int = 32,
        batch_size_mem: int = 32,
        task_balanced_dataloader: bool = False,
        **kwargs
    ):
        """Custom data loader for rehearsal strategies.
        The iterates in parallel two datasets, the current `data` and the
        rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).
        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.
        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        self.data = data
        self.memory = memory
        self.loader_data: Sequence[DataLoader] = {}
        self.loader_memory: Sequence[DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches

        num_keys = len(self.memory.task_set)
        if task_balanced_dataloader:
            assert batch_size_mem >= num_keys, (
                "Batch size must be greator or equal "
                "to the number of tasks in the memory "
                "and current data."
            )

        # Create dataloader for data items
        self.loader_data, _ = self._create_dataloaders(
            data, batch_size, 0, False, **kwargs
        )

        # Create dataloader for memory items
        if task_balanced_dataloader:
            single_group_batch_size = batch_size_mem // num_keys
            remaining_example = batch_size_mem % num_keys
        else:
            single_group_batch_size = batch_size_mem
            remaining_example = 0

        self.loader_memory, remaining_example = self._create_dataloaders(
            memory,
            single_group_batch_size,
            remaining_example,
            task_balanced_dataloader,
            **kwargs
        )

        self.max_len = max(
            [
                len(d)
                for d in chain(
                    self.loader_data.values(), self.loader_memory.values()
                )
            ]
        )

    def __iter__(self):
        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in self.loader_data.keys():
            iter_data_dataloaders[t] = iter(self.loader_data[t])
        for t in self.loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(self.loader_memory[t])

        max_len = max([len(d) for d in iter_data_dataloaders.values()])

        try:
            for it in range(max_len):
                mb_curr = []
                self._get_mini_batch_from_data_dict(
                    self.data,
                    iter_data_dataloaders,
                    self.loader_data,
                    False,
                    mb_curr,
                )

                self._get_mini_batch_from_data_dict(
                    self.memory,
                    iter_buffer_dataloaders,
                    self.loader_memory,
                    self.oversample_small_tasks,
                    mb_curr,
                )

                yield self.collate_mbatches(mb_curr)
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    def _get_mini_batch_from_data_dict(
        self,
        data,
        iter_dataloaders,
        loaders_dict,
        oversample_small_tasks,
        mb_curr,
    ):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    continue
            mb_curr.append(tbatch)

    def _create_dataloaders(
        self,
        data_dict,
        single_exp_batch_size,
        remaining_example,
        task_balanced_dataloader,
        **kwargs
    ):
        loaders_dict: Dict[int, DataLoader] = {}
        if task_balanced_dataloader:
            for task_id in data_dict.task_set:
                data = data_dict.task_set[task_id]
                current_batch_size = single_exp_batch_size
                if remaining_example > 0:
                    current_batch_size += 1
                    remaining_example -= 1
                loaders_dict[task_id] = DataLoader(
                    data, batch_size=current_batch_size, **kwargs
                )
        else:
            loaders_dict[0] = DataLoader(
                data_dict, batch_size=single_exp_batch_size, **kwargs
            )

        return loaders_dict, remaining_example
