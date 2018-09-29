import argparse
import json
from pathlib import Path
import torch
import torchvision as tv

from models import NetConv, NetFC, MNISTTrainer


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--config-path', type=Path, required=True, metavar='C')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    return config


def get_dataset(batch_size: int = 64, num_workers: int = 4):
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


def get_model(model_type="fc"):
    if model_type == "conv":
        return NetConv()
    else:
        return NetFC()


def main():
    config = get_config()
    torch.manual_seed(config['seed'])
    log_interval = config['log_interval']
    lr = config['lr']
    model_type = config['model_type']
    batch_size = config['batch_size']
    num_workers = config['num_workers']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataset(num_workers=num_workers, batch_size=batch_size)
    model = get_model(model_type=model_type).to(device)
    mnist_trainer = MNISTTrainer(model, train_loader, test_loader, lr=lr, device=device, log_interval=log_interval)
    mnist_trainer.train_model()


if __name__ == '__main__':
    main()
