import torchvision
from PIL import Image
import numpy as np
import torch
import ssl
import math

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, test_transform=None, download=True):
        ssl._create_default_https_context = ssl._create_unverified_context
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.test_transform = test_transform
        self.val_mode = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None and not self.val_mode:
            img = self.transform(img)
        elif self.test_transform is not None and self.val_mode:
            img = self.test_transform(img)

        return img, target

class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, transform, test_transform=None, download=True, coarse=False):
        ssl._create_default_https_context = ssl._create_unverified_context
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.test_transform = test_transform
        self.val_mode = False
        self.coarse = coarse

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None and not self.val_mode:
            img = self.transform(img)
        elif self.test_transform is not None and self.val_mode:
            img = self.test_transform(img)

        if self.coarse:
            target = math.floor(target / 5)  # Convert to coarse label (20 classes)

        return img, target