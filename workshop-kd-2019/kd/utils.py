import torch
import torchvision as tv
from torch.utils.data import DataLoader, Dataset


def get_mnist() -> (Dataset, Dataset):
    train_ds = tv.datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=tv.transforms.Compose([tv.transforms.ToTensor()]),
    )
    test_ds = tv.datasets.MNIST(
        "../data",
        train=False,
        transform=tv.transforms.Compose([tv.transforms.ToTensor()]),
    )
    return train_ds, test_ds


def get_mnist_loaders(
    batch_size: int = 64, num_workers: int = 4
) -> (DataLoader, DataLoader):
    train_ds, test_ds = get_mnist()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
