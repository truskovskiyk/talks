import argparse
import json
import pickle
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from imblearn.datasets import make_imbalance

from models import NetFC, MNISTTrainer
from common import get_mnist
from common.sampler import fix_dataset


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
                                               num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_model(model_type="fc"):
    if model_type == "fc":
        return NetFC()
    else:
        raise ValueError(f"wrong type of {model_type}")


def dataset_as_numpy(ds: Dataset, num_workers: int = 4, shuffle: bool = True):
    loader = DataLoader(ds, batch_size=len(ds), shuffle=shuffle,
                        num_workers=num_workers)
    x, y = next(iter(loader))
    x, y = x.squeeze().numpy(), y.numpy()
    return x, y


def main():
    config = get_config()
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_interval = config['log_interval']
    lr = config['lr']
    model_type = config['model_type']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    imbalance_ratio = config['imbalance_ratio']
    num_minor_classes = config['num_minor_classes']
    use_gan = config['use_gan']
    n_epoch = config['n_epoch']
    name = config['name']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_unbalanced_dataset(imbalance_ratio=imbalance_ratio,
                                                       num_minor_classes=num_minor_classes,
                                                       num_workers=num_workers,
                                                       batch_size=batch_size)

    print(len(train_loader.dataset))
    print(train_loader.dataset)

    if use_gan:
        # Step 1 train gan
        # t = CGANTrainer()
        # t.train_model(train_loader)

        # # Step 2 sample from the gan
        train_loader = fix_dataset(train_loader)
        with open("train_loader.pth", "wb") as f:
            pickle.dump(train_loader, f)

        # with open("data/train_loader.pth", "rb") as f:
        #     train_loader = pickle.load(f)
        print(len(train_loader.dataset))
        print(train_loader.dataset)

    model = get_model(model_type=model_type).to(device)
    mnist_trainer = MNISTTrainer(model=model,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 lr=lr,
                                 device=device,
                                 log_interval=log_interval,
                                 n_epoch=n_epoch,
                                 name=name)
    mnist_trainer.train_model()


if __name__ == '__main__':
    main()
