import copy
from typing import TYPE_CHECKING

from avalanche.training.plugins.strategy_plugin import StrategyPlugin

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy

from training.storage_policy.c_score_policy import CScoreBuffer

class GDumbPluginMod(StrategyPlugin):
    """ GDumb plugin.

    At each experience the model is trained  from scratch using a buffer of
    samples collected from all the previous learning experiences.
    The buffer is updated at the start of each experience to add new classes or
    new examples of already encountered classes.
    In multitask scenarios, mem_size is the memory size for each task.
    This plugin can be combined with a Naive strategy to obtain the
    standard GDumb strategy.
    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """

    def __init__(self, mem_size: int = 200, name_dataset: str = 'cifar10',
                 mode: str = 'random', mix_upper: float = 0.5, mix_lower: float = 0.5):
        super().__init__()
        self.mem_size = mem_size

        # model initialization
        self.buffer = {}
        self.storage_policy = CScoreBuffer(
            max_size=self.mem_size,
            adaptive_size=True,
            name_dataset=name_dataset,
            mode=mode,
            mix_upper=mix_upper,
            mix_lower=mix_lower
        )
        self.init_model = None


    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        """ Reset model. """
        if self.init_model is None:
            self.init_model = copy.deepcopy(strategy.model)
        else:
            strategy.model = copy.deepcopy(self.init_model)
        strategy.model_adaptation(self.init_model)

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        strategy.model_adaptation(self.init_model)

    def after_train_dataset_adaptation(self, strategy: "BaseStrategy",
                                       **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        strategy.adapted_dataset = self.storage_policy.buffer