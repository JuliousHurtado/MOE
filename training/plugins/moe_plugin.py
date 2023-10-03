from typing import Optional, TYPE_CHECKING
import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from training.data_loader.dataloader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer, \
    ExperienceBalancedBuffer


if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


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

class GetScores(object):
    def __init__(self, dataset, scores):
        self.dataset = dataset
        self.scores = scores

    def __getitem__(self, index):
        # print(index)
        # print(self.dataset[index])
        return (*self.dataset[index], self.scores[index])
    
    def __len__(self):
         return len(self.dataset)



class MOEPlugin(SupervisedPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks. 
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced 
    such that there are the same number of examples for each experience.    
    
    The `after_training_exp` callback is implemented in order to add new 
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored 
    in the external memory.
    """

    def __init__(self, mem_size: int = 200,
                        batch_size: int = None,
                        batch_size_mem: int = None,
                        task_balanced_dataloader: bool = False,
                        storage_policy: Optional["ExemplarsBuffer"] = None,
                        weigth_loss: bool = False):
        """
        :param storage_policy: The policy that controls how to add new exemplars
                        in memory
        """
        super().__init__()
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.collate_mbatches = _default_collate_mbatches_fn
        self.weigth_loss = weigth_loss

        self.loss = nn.CrossEntropyLoss(reduction='none')

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size,
                adaptive_size=True)

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        datasets = []
        scores = []
        for class_id in self.storage_policy.buffer_groups.keys():
            data = self.storage_policy.buffer_groups[class_id].buffer
            datasets.append(data)
            scores.extend(self.storage_policy.buffer_groups[class_id]._buffer_weights)

        data = GetScores(ConcatDataset(datasets), scores)
        self.loaders_dict = DataLoader(
                    data, batch_size=batch_size_mem, shuffle=True, **kwargs
                )

        self.iter_buffer_dataloaders = iter(self.loaders_dict)

    def iter_buffer(self):
        try:
            tbatch = next(self.iter_buffer_dataloaders)
        except StopIteration:
            self.iter_buffer_dataloaders = iter(self.loaders_dict)
            tbatch = next(self.iter_buffer_dataloaders)

        return tbatch

    def before_backward(self, strategy: "BaseStrategy", **kwargs):
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        if self.weigth_loss:
            strategy.loss = (self.loss(strategy.mb_output, strategy.mb_y)*(1 + self.weight)).mean()
        else:
            strategy.loss = (self.loss(strategy.mb_output, strategy.mb_y)).mean()

    def before_forward(self, strategy: "BaseStrategy", **kwargs):
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch = self.iter_buffer()
            
        # Aqui calculamos loss con cross entropy por sample
        x = batch[0].to(strategy.device)
        y_real = batch[1].to(strategy.device)
        task_id = batch[2].to(strategy.device)
        weight = batch[3].to(strategy.device)

        mb_x = strategy.mb_x.to(strategy.device)
        mb_y = strategy.mb_y.to(strategy.device)
        mb_task_id = strategy.mb_task_id.to(strategy.device)
        mb_weight = torch.ones(strategy.mb_x.size(0)).to(strategy.device)

        mb_x = torch.cat([mb_x, x], dim=0)
        mb_y = torch.cat([mb_y, y_real], dim=0)
        mb_task_id = torch.cat([strategy.mb_task_id, task_id], dim=0)
        self.weight = torch.cat([ mb_weight, weight ], dim=0)

        strategy.mbatch[0] = mb_x
        strategy.mbatch[1] = mb_y
        strategy.mbatch[2] = mb_task_id

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        strategy.train_mb_size = self.batch_size_mem
        self.storage_policy.update(strategy, **kwargs)