import torch
import torch.nn.functional as F
import copy
from typing import Optional, TYPE_CHECKING
from torch.utils.data import DataLoader

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.storage_policy import ExemplarsBuffer, \
    ExperienceBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy



class MIRPlugin(StrategyPlugin):
    """ Maximally Interfered Retrieval Plugin.
    """

    def __init__(self, mem_size: int = 200,
                       innit_lr: float = 0.1,
                       mir_replay: bool = True,
                       storage_policy: Optional["ExemplarsBuffer"] = None):
        

        self.mem_size = mem_size
        self.mir_replay = mir_replay
        self.batch_size = 100 #batch_size
        self.base_lr = innit_lr
        self.buffer_batch_size = 10 # 50 # batch_size // 2

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size,
                adaptive_size=True)
    
    def before_training_exp(self, strategy: "BaseStrategy", **kwargs):
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return  

        self.dataloader = DataLoader(self.storage_policy.buffer, 
                                batch_size=self.batch_size, 
                                shuffle=True, num_workers=2)
        self.iterator = iter(self.dataloader)

    def before_update(self, strategy: "BaseStrategy", **kwargs):
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        try:
            batch_buffer = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch_buffer = next(self.iterator)

        if self.mir_replay:
            grads = self.get_grad_vector(strategy.model)
            new_model = self.get_future_step_parameters(strategy.model, grads).to(strategy.device)

            b_x = batch_buffer[0].to(strategy.device)
            b_y = batch_buffer[1].to(strategy.device)
            with torch.no_grad():
                logits_track_pre = strategy.model(b_x)
                logits_track_post = new_model(b_x)

                pre_loss = F.cross_entropy(logits_track_pre, b_y , reduction="none")
                post_loss = F.cross_entropy(logits_track_post, b_y , reduction="none")
                scores = post_loss - pre_loss

                idxs = scores.sort(descending=True)[1][:self.buffer_batch_size]
            
            mem_x = batch_buffer[0][idxs].to(strategy.device)
            mem_y = batch_buffer[1][idxs].to(strategy.device)
        else:
            mem_x = batch_buffer[0][:self.buffer_batch_size].to(strategy.device)
            mem_y = batch_buffer[1][:self.buffer_batch_size].to(strategy.device)
        logits_buffer = strategy.model(mem_x)
        F.cross_entropy(logits_buffer, mem_y).backward()

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        
    def get_grad_vector(self, model):
        grads = {}
        for n,p in model.named_parameters():
            if p.grad is not None:
                grads[n] = p.grad
        
        return grads
    
    def get_future_step_parameters(self, model, grads):
        new_net = copy.deepcopy(model)

        with torch.no_grad():
            for n, param in new_net.named_parameters():
                if n in grads:
                    param.data=param.data - self.base_lr*grads[n]
        return new_net