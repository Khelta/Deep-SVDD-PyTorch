from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import SVHN
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
import numpy as np


class SVHN_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=5, p=0.25):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min = -34.924463588638204
        max = 15.2464923804279

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min] * 3,
                                                             [max - min] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set_all = MySVHN(root=self.root, download=True,
                               transform=transform, target_transform=target_transform)
        self.train_set_all = train_set_all
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set_all.labels, self.normal_classes)
        train_idx_anormal = get_target_label_idx(train_set_all.labels, self.normal_classes, invert=True)
        i = int(len(train_idx_normal) / (1 - p) * p)
        train_idx_anormal = list(np.random.choice(train_idx_anormal, i, replace=False))
        self.train_set = Subset(train_set_all, np.concatenate((train_idx_normal, train_idx_anormal)))
        self.test_set = MySVHN(root=self.root, download=True,
                               transform=transform, target_transform=target_transform)


class MySVHN(SVHN):
    """Torchvision SVHN class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MySVHN, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the SVHN class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.reshape(32, 32, 3)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index  # only line changed
