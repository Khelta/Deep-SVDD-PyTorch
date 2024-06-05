from .mnist import MNIST_Dataset
from .fashion import FashionMNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .cifar100 import CIFAR100_Dataset
from .svhn import SVHN_Dataset


def load_dataset(dataset_name, data_path, normal_class, p=0.25):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fashion', 'cifar10', 'cifar100', 'svhn')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class, p=p)

    if dataset_name == 'fashion':
        dataset = FashionMNIST_Dataset(root=data_path, normal_class=normal_class, p=p)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class, p=p)

    if dataset_name == 'cifar100':
        dataset = CIFAR100_Dataset(root=data_path, normal_class=normal_class, p=p)

    if dataset_name == 'svhn':
        dataset = SVHN_Dataset(root=data_path, normal_class=normal_class, p=p)

    return dataset
