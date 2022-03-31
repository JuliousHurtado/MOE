from collections import defaultdict
import os
from avalanche.evaluation import PluginMetric

from datasets.load_c_score import ImagenetCScore, CIFARIdx

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset, DataLoader
import torch

class CScoreMetric(PluginMetric[float]):
    """
    This metric will return a `float` value after
    each training epoch
    """

    def __init__(self, name_dataset, train_transform=None, val_transform=None,
                     root='../data', top_percentaje: float=0.2,
                     save_in_file=True, path_save_file='./results'):
        """
        Initialize the metric
        """
        super().__init__()

        self.task_to_classes = defaultdict(dict)
        self.class_to_dataloader = defaultdict(dict)
        self.acc_result_classes = defaultdict(dict)

        self.acc_tasks = []

        self.save_in_file = save_in_file
        self.path_save_file = path_save_file

        self.top_percentaje = top_percentaje

        if name_dataset == 'cifar10':
            train_dataset = CIFARIdx(CIFAR10)(transform=train_transform, root=root, train=True, download=True)
            val_dataset = CIFARIdx(CIFAR10)(transform=val_transform, root=root, train=False, download=True)
        elif name_dataset == 'cifar100':
            train_dataset = CIFARIdx(CIFAR100)(transform=train_transform, root=root, train=True, download=True)
            val_dataset = CIFARIdx(CIFAR100)(transform=val_transform, root=root, train=False, download=True)
        elif name_dataset == 'imagenet':
            train_dataset = ImagenetCScore(transform=train_transform, root=root, train=True)
            val_dataset = ImagenetCScore(transform=val_transform, root=root, train=False)
        else:
            assert False, "Dataset {} not found".format(name_dataset)

        if self.top_percentaje > 0:
            self.get_dataloader(train_dataset, val_dataset)

    def get_dataloader(self, train_dataset, val_dataset) -> None:
        total_labels = set(train_dataset.targets)

        total_top_train = int((len(train_dataset)*self.top_percentaje)/len(total_labels))
        total_top_val = int((len(val_dataset)*self.top_percentaje)/len(total_labels))

        if total_top_train < 5:
            total_top_train = 5
            print("The amount for the analysis in Train must be >= 5, setting number to 5")

        if total_top_val < 5:
            total_top_val = 5
            print("The amount for the analysis in Val must be >= 5, setting number to 5")

        for i in total_labels:
            train_index = torch.Tensor(range(len(train_dataset)))
            train_sub_index = train_index[ torch.Tensor(train_dataset.targets) == i ].int()
            
            train_score_index = torch.argsort(torch.from_numpy(train_dataset.scores[train_sub_index]))
            sort_train_score_index = train_sub_index[ train_score_index ]

            val_index = torch.Tensor(range(len(val_dataset)))
            val_sub_index = val_index[ torch.Tensor(val_dataset.targets) == i ].int()

            val_score_index = torch.argsort(torch.from_numpy(val_dataset.scores[val_sub_index]))
            sort_val_score_index = val_sub_index[ val_score_index ]

            self.class_to_dataloader[i] = {
                'train_lower' : DataLoader(Subset(train_dataset, sort_train_score_index[:total_top_train]), batch_size=128),
                'train_upper' : DataLoader(Subset(train_dataset, sort_train_score_index[total_top_train:]), batch_size=128),
                'val_lower' : DataLoader(Subset(val_dataset, sort_val_score_index[:total_top_val]), batch_size=128),
                'val_upper' : DataLoader(Subset(val_dataset, sort_val_score_index[total_top_val:]), batch_size=128),
            }

    def reset(self) -> None:
        """
        Reset the metric
        """
        pass

    def result(self) -> float:
        """
        Emit the result
        """
        pass

    def after_training_exp(self, strategy: 'PluggableStrategy') -> None:
        # print(self.acc_result_classes)
        self.acc_tasks.append(self.acc_epochs)

    def before_training_exp(self, strategy: 'PluggableStrategy') -> None:
        self.task_to_classes[strategy.experience.current_experience] = \
                    strategy.experience.classes_in_this_experience
        
        for c in strategy.experience.classes_in_this_experience:
            self.acc_result_classes[c]['train_lower'] = []
            self.acc_result_classes[c]['train_upper'] = []
            self.acc_result_classes[c]['val_lower'] = []
            self.acc_result_classes[c]['val_upper'] = []
        
        self.acc_epochs = []

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        if self.top_percentaje > 0:
            for t in self.task_to_classes.keys():
                for c in self.task_to_classes[t]:
                    self.update_accuracy_class(strategy, c, 'train_lower')
                    self.update_accuracy_class(strategy, c, 'train_upper')
                    self.update_accuracy_class(strategy, c, 'val_lower')
                    self.update_accuracy_class(strategy, c, 'val_upper')
        
        self.acc_epochs.append(strategy.evaluator.metrics[0]._metric.result()[0])

    def after_training(self, strategy: 'PluggableStrategy'):
        if self.save_in_file:
            torch.save({
                'acc_per_class': self.acc_result_classes,
                'acc_task': self.acc_tasks,
                }, os.path.join(self.path_save_file, strategy.save_file_name))
        
    def update_accuracy_class(self, strategy, c, group):
        dataloder = self.class_to_dataloader[c][group]

        running_vacc = 0
        total_items = 0
        with torch.no_grad():
            for _, x, target, _ in dataloder:
                x = x.to(strategy.device)
                target = target.to(strategy.device)

                outputs = strategy.model(x)

                _, preds = torch.max(outputs, 1)
                running_vacc += (preds == target).sum().item()
                total_items += len(target)
        
        self.acc_result_classes[c][group].append(running_vacc/total_items)

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Acc_top_classes"