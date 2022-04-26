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

_default_mnist_train_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

_default_mnist_eval_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

def get_imagenet(root: str, n_experiences: int, return_task_id=False, 
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_imgenet_train_transform,
        eval_transform: Optional[Any] = _default_imgenet_val_transform):
    
    img_train = ImageNet(os.path.join(root, 'train'), True, transform = train_transform)
    img_train = ImageNet(os.path.join(root, 'val'), False, transform = eval_transform)

    if return_task_id:
        return nc_benchmark(
            train_dataset=img_train,
            test_dataset=img_train,
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
            train_dataset=img_train,
            test_dataset=img_train,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)

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

class ImageNet(Dataset):
    def __init__(self, root: str=".", 
                       train = True,
                       img_root="c_score/imagenet",
                       transform=None,
                        **kwargs: Any):
        super(ImageNet, self).__init__()
        # Load file list
        # Load scores list
        split = "train" if train else "test"
        self.transform = transform
        self.files = np.load(join(img_root,f"filenames_{split}.npy"), allow_pickle=True)
        self.targets = np.load(join(img_root,f"labels_{split}.npy"), allow_pickle=True)
        self.root = root
        
        for i in range(len(self.files)):
            self.files[i] = make_path(str(self.files[i]).replace("b'","")[:-1])
        # self.scores = np.load(join(img_root,f"scores_{split}.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.files[index]
        target = self.targets[index]
        img = Image.open(join(self.root,img)) # open img in img folder
        if img.mode != "RGB":
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg
        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == '__main__':
    dataset = MNIST()

    print(dataset[0][0].shape)