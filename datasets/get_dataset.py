from torch.utils.data import Dataset
import torch
import numpy as np
import os
from os.path import join

from torchvision.transforms import ToTensor, Normalize, Compose, \
    RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop

from typing import Any, Tuple, Optional, Sequence
from PIL import Image

from avalanche.benchmarks import nc_benchmark

from datasets.mnist import MNIST, _default_mnist_train_transform, _default_mnist_eval_transform
from datasets.tiny_imagenet import TinyImageNetDataset


def get_dataset_class(dataset_class, root, mode, transform=None):
    class IndexedDataset(dataset_class):
        def __getitem__(self, index):
            return (*super().__getitem__(index), index)
    return IndexedDataset(root, mode, transform=transform)

def make_path(path):
    folder = path.split("_")[0]
    return join(folder, path)

_default_imgenet_train_transform = Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

_default_imgenet_val_transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

def get_mnist(n_experiences: int, return_task_id=False, 
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform):

    mnist_train = MNIST(train = True)
    mnist_test = MNIST(train = False)

    if return_task_id:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)

def get_tiny_imagenet(root: str, n_experiences: int, return_task_id=False, 
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_imgenet_train_transform,
        eval_transform: Optional[Any] = _default_imgenet_val_transform):
    
    tiny_train = get_dataset_class(TinyImageNetDataset, root, 'train')#, train_transform)
    tiny_val = get_dataset_class(TinyImageNetDataset, root, 'val')#, eval_transform)
    
    if return_task_id:
        return nc_benchmark(
            train_dataset=tiny_train,
            test_dataset=tiny_val,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=tiny_train,
            test_dataset=tiny_val,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)

if __name__ == '__main__':
    dataset = MNIST()

    print(dataset[0][0].shape)