from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx
import torch

import numpy as np
import torchvision.transforms as transforms

min = 1
max = 1


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    global max
    global min
    max = torch.max(x) if torch.max(x) > max else max
    min = torch.min(x) if torch.min(x) < min else min
    return x


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0, p=0.25):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        min = -0.8827
        max = 20.1081

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min], [max - min])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set_all = MyMNIST(root=self.root, train=True, download=True,
                                transform=transform, target_transform=target_transform)
        self.train_set_all = train_set_all
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set_all.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        train_idx_anormal = get_target_label_idx(train_set_all.train_labels.clone().data.cpu().numpy(), self.normal_classes, invert=True)
        i = int(len(train_idx_normal) / (1 - p) * p)
        train_idx_anormal = list(np.random.choice(train_idx_anormal, i, replace=False))
        self.train_set = Subset(train_set_all, np.concatenate((train_idx_normal, train_idx_anormal)))
        self.test_set = MyMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index  # only line changed
