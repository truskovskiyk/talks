import torch
import torchvision as tv
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size: int = 64, num_workers: int = 4) -> (DataLoader, DataLoader):
    train_loader = torch.utils.data.DataLoader(
        tv.datasets.MNIST('../data', train=True, download=True,
                          transform=tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize((0.5,), (0.5,))
                          ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        tv.datasets.MNIST('../data', train=False, transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,))
        ])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader
