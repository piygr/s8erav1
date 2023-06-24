import torch
from torchvision import datasets, transforms
from transforms import get_test_transforms, get_train_transforms

dataset_mean, dataset_std = (0.4914, 0.4822, 0.4465), \
            (0.2470, 0.2435, 0.2616)

def get_dataset_mean_variance(dataset):

    if dataset_mean and dataset_std:
        return dataset_std, dataset_std

    imgs = [item[0] for item in dataset]
    imgs = torch.stack(imgs, dim=0)

    mean = []
    std = []
    for i in range(imgs.shape[1]):
        mean.append(imgs[:, i, :, :].mean().item())
        std.append(imgs[:, i, :, :].std().item())

    return tuple(mean), tuple(std)


def get_loader(**kwargs):

    dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.ToTensor())
    mean, std = get_dataset_mean_variance(dataset)
    train_data = datasets.CIFAR10('../data', train=True, download=True, transform=get_train_transforms(mean=mean, std=std))
    test_data = datasets.CIFAR10('../data', train=False, download=True, transform=get_test_transforms(mean=mean, std=std))

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)


def get_dataset_labels():
    return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def get_data_label_name(idx):
    if idx < 0:
        return ''

    return get_dataset_labels()[idx]


def get_data_idx_from_name(name):
    if not name:
        return -1

    return get_dataset_labels.index(name.lower()) if name.lower() in get_dataset_labels() else -1

