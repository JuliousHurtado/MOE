from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torch

from typing import Any, Tuple
from os.path import join
from PIL import Image
import numpy as np

def make_path(path):
    folder = path.split("_")[0]
    return join(folder, path)

class ImagenetCScore(Dataset):
    def __init__(self, root: str=".", 
                       train = True,
                       img_root="c_score/imagenet",
                       transform=None,
                        **kwargs: Any):
        super(ImagenetCScore, self).__init__()
        # Load file list
        # Load scores list
        split = "train" if train else "test"
        self.transform = transform
        self.files = np.load(join(img_root,f"filenames_{split}.npy"), allow_pickle=True)
        self.targets = np.load(join(img_root,f"labels_{split}.npy"), allow_pickle=True)
        self.root = root
        
        for i in range(len(self.files)):
            self.files[i] = make_path(str(self.files[i]).replace("b'","")[:-1])
        self.scores = np.load(join(img_root,f"scores_{split}.npy"))

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

        score = self.scores[index]
        return index, img, target, score

# CIFAR10

def CIFARIdx(cl):
    dataset = "cifar10" if cl == CIFAR10 else "cifar100"
    
    class DatasetCIFARIdx(cl):
        scores = np.load(f"c_score/{dataset}/scores.npy")
        def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return index, img, target, self.scores[index]

    return DatasetCIFARIdx


# MNIST

class MNISTIdx(Dataset):
    def __init__(self, root: str=".", 
                       train = True,
                       img_root="c_score/mnist_with_c_score.pth",
                       transform=None,
                        **kwargs: Any):

        data, self.targets, self.scores = self.load_data(img_root, train)
        self.data = data.astype(np.float32) / 255

        self.transform = transform

    def load_data(self, name_file, train):
        data = torch.load(name_file)

        if train:
            return data['train_image'], data['train_labels'], data['train_scores']
        else:
            assert False, "We don't have score for val or test"

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

        return index, img, target, self.scores[index]