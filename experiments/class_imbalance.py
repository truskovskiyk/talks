import argparse
import json
import logging
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from imblearn.datasets import make_imbalance

from models import NetConv, NetFC, MNISTTrainer
from common import get_mnist

logger = logging.getLogger(__name__)


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch MNIST class imbalance example')
    parser.add_argument('--config-path', type=Path, required=True, metavar='C')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    return config


def create_imbalance_ration(labels: np.array, num_minor_classes: int, imbalance_ratio: float, classes: tuple):
    assert len(classes) > num_minor_classes

    class_counter = Counter(labels)
    minor_classes = np.random.choice(classes, size=num_minor_classes, replace=False)
    logger.info(f"have chosen {minor_classes} as minor_classes")
    for minor_class in minor_classes:
        class_index = classes.index(minor_class)
        # update old one class_ratio
        class_ratio = int(imbalance_ratio * class_counter[class_index])
        class_counter[class_index] = class_ratio
    return dict(class_counter)


def make_imbalance_dataset(dataset: Dataset,
                           indexes: np.array,
                           labels: np.array,
                           num_minor_classes: int,
                           imbalance_ratio: float,
                           classes: np.array):
    ratio = create_imbalance_ration(labels=labels, num_minor_classes=num_minor_classes,
                                    imbalance_ratio=imbalance_ratio, classes=classes)
    x, y = make_imbalance(indexes.reshape(-1, 1), labels.reshape(-1, 1), ratio=ratio)
    ds = Subset(dataset=dataset, indices=x.flatten())
    return ds


def get_unbalanced_dataset(imbalance_ratio: float, num_minor_classes: int, batch_size: int, num_workers: int):
    train_ds, test_ds = get_mnist()

    labels = train_ds.train_labels.numpy()
    indexes = np.arange(len(train_ds))
    classes = train_ds.train_labels.unique().numpy().tolist()

    new_train_ds = make_imbalance_dataset(dataset=train_ds, indexes=indexes, labels=labels,
                                          num_minor_classes=num_minor_classes, imbalance_ratio=imbalance_ratio,
                                          classes=classes)
    train_loader = torch.utils.data.DataLoader(new_train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
    return train_loader, test_loader


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
    imbalance_ratio = config['imbalance_ratio']
    num_minor_classes = config['num_minor_classes']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_unbalanced_dataset(imbalance_ratio=imbalance_ratio,
                                                       num_minor_classes=num_minor_classes,
                                                       num_workers=num_workers,
                                                       batch_size=batch_size)
    model = get_model(model_type=model_type).to(device)
    mnist_trainer = MNISTTrainer(model=model, train_loader=train_loader, test_loader=test_loader,
                                 lr=lr, device=device, log_interval=log_interval)
    mnist_trainer.train_model()


if __name__ == '__main__':
    main()
