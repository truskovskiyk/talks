import argparse
import json
from pathlib import Path
import torch
import torchvision as tv

from models.gan_trainer import Generator, MNISTTrainer


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch MNIST GAN Example')
    parser.add_argument('--config-path', type=Path, required=True, metavar='C')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    return config


def get_dataset():
    batch_size = 64
    num_workers = 4
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


def main():
    config = get_config()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataset()

    model = Net().to(device)
    mnist_trainer = MNISTTrainer(model, train_loader, test_loader, lr=0.001, device=device, log_interval=50)
    mnist_trainer.train_model()


if __name__ == '__main__':
    main()
