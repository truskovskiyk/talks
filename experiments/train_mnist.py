import argparse
import json
from pathlib import Path
import torch

from models import NetConv, NetFC, MNISTTrainer
from common import get_mnist_loaders


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--config-path', type=Path, required=True, metavar='C')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    return config


def get_model(model_type="fc"):
    if model_type == "conv":
        return NetConv()
    elif model_type == "fc":
        return NetFC()
    else:
        raise ValueError(f"wrong type of {model_type}")


def main():
    config = get_config()
    torch.manual_seed(config['seed'])
    log_interval = config['log_interval']
    lr = config['lr']
    model_type = config['model_type']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(num_workers=num_workers, batch_size=batch_size)
    model = get_model(model_type=model_type).to(device)
    mnist_trainer = MNISTTrainer(model, train_loader, test_loader, lr=lr, device=device, log_interval=log_interval)
    mnist_trainer.train_model()


if __name__ == '__main__':
    main()
