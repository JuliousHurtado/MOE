import torch
import numpy as np
import random

from avalanche.benchmarks.utils.data_loader import \
    GroupBalancedInfiniteDataLoader
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.models import avalanche_forward



class AGEMPluginMod(StrategyPlugin):
    """ Average Gradient Episodic Memory Plugin.
    
    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, sample_size: int,
                 mode: str = 'random', mix_upper: float = 0.5,
                 name_dataset: str = 'cifar10'):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.sample_size = int(sample_size)
        self.mode = mode
        self.mix_upper = mix_upper

        self.buffers = []  # one AvalancheDataset for each experience.
        self.buffer_dataloader = None
        self.buffer_dliter = None

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None

        if name_dataset == 'cifar10' or name_dataset == 'cifar100':
            self.scores = np.load(f"c_score/{name_dataset}/scores.npy")
        elif name_dataset == 'mnist':
            data = torch.load('c_score/mnist_with_c_score.pth')
            self.scores = data['train_scores']
        elif name_dataset == 'imagenet':
            self.scores = np.load(f"c_score/{name_dataset}/scores_train.npy")
        else:
            assert False, "Dataset {} not found".format(name_dataset)

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            xref, yref, tid = mb[0], mb[1], mb[-1]
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            loss = strategy._criterion(out, yref)
            loss.backward()
            # gradient can be None for some head on multi-headed models
            self.reference_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()]
            self.reference_gradients = torch.cat(self.reference_gradients)
            strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffers) > 0:
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()]
            current_gradients = torch.cat(current_gradients)

            assert current_gradients.shape == self.reference_gradients.shape, \
                "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(self.reference_gradients,
                                          self.reference_gradients)
                grad_proj = current_gradients - \
                    self.reference_gradients * alpha2
                
                count = 0 
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count:count+n_param].view_as(p))
                    count += n_param

    def after_training_exp(self, strategy, **kwargs):
        """ Update replay memory with patterns from current experience. """
        self.update_memory(strategy.experience.dataset)

    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(self.buffer_dliter)

    @torch.no_grad()
    def update_memory(self, dataset):
        """
        Update replay memory with patterns from current experience.
        """
        # Get sample idxs per class
        cl_idxs = {}
        cl_score = {}
        for idx, target in enumerate(dataset.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
                cl_score[target] = []
            cl_idxs[target].append(idx)
            cl_score[target].append(self.scores[dataset._indices[idx]])

        num_clss = len(cl_idxs.keys())
        ll = int(self.patterns_per_experience/num_clss)

        # Make AvalancheSubset per class
        # cl_datasets = {}
        select_idxs = []
        for c, c_idxs in cl_idxs.items():
            # cl_datasets[c] = AvalancheSubset(dataset, indices=c_idxs)
            c_idxs = torch.tensor(c_idxs)

            if self.mode == 'lower':
                _, sorted_idxs = torch.tensor(cl_score[c]).sort(descending=False)
            elif self.mode == 'upper' or self.mode == 'mix':
                _, sorted_idxs = torch.tensor(cl_score[c]).sort(descending=True)
            else: # random
                new_weights = torch.rand(len(cl_score[c]))
                _, sorted_idxs = new_weights.sort(descending=True)

            if self.mode == 'mix':
                num_upper = int(ll*self.mix_upper)
                upper_list = random.sample(sorted_idxs[:ll].tolist(), num_upper)
                lower_list = random.sample(sorted_idxs[ll:].tolist(), ll - num_upper)
                select_idxs.extend(c_idxs[ sorted_idxs[upper_list + lower_list] ])
            else:
                select_idxs.extend(c_idxs[ sorted_idxs[:ll] ])

        class_dataset = AvalancheSubset(dataset, indices=select_idxs)
        self.buffers.append(class_dataset)
        
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=self.sample_size // len(self.buffers),
            num_workers=2,
            #pin_memory=True,
            #persistent_workers=True)
            )
        self.buffer_dliter = iter(self.buffer_dataloader)