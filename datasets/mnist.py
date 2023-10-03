from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import torch

from typing import Any, Tuple, Optional, Sequence

import numpy as np

_default_mnist_train_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

_default_mnist_eval_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

class MNIST(Dataset):
    def __init__(self, root: str=".", 
                       train = True,
                       img_root="c_score/mnist_with_c_score.pth",
                       transform=None,
                        **kwargs: Any):

        data, self.targets = self.load_data(img_root, train)
        self.data = data.astype(np.float32) / 255

        self.transform = transform

    def load_data(self, name_file, train):
        data = torch.load(name_file)

        if train:
            return data['train_image'], data['train_labels']
        else:
            return data['test_image'], data['test_labels']

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode="L")
        # img = img.astype(np.float32) / 255

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target
