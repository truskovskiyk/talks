from collections import Counter

import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import torch.nn as nn
import torch.nn.functional as F
import torch


class ModelG(nn.Module):
    def __init__(self, z_dim=100):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * 28 * 28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x


def dataset_as_numpy(ds, num_workers: int = 4, shuffle: bool = True):
    loader = DataLoader(ds, batch_size=len(ds), shuffle=shuffle, num_workers=num_workers)
    x, y = next(iter(loader))
    x, y = x.squeeze().numpy(), y.numpy()
    return x, y


class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def _sample(labels, generator):
    images = []
    batch_size = 32
    latent_dim = 100
    for i in range(0, len(labels), batch_size):
        labels_batch = labels[i:i + batch_size]

        bs = len(labels_batch)

        z = np.random.normal(0, 1, (bs, latent_dim))

        labels_batch_one_hot = np.zeros((bs, 10))
        labels_batch_one_hot[np.arange(bs), labels_batch] = 1

        labels_batch_one_hot = torch.FloatTensor(labels_batch_one_hot).cuda()
        z = torch.FloatTensor(z).cuda()

        _images = generator(z, labels_batch_one_hot)
        images.append(_images.detach().cpu())
    return images


def fix_dataset(train_loader: DataLoader):
    generator = ModelG()
    state_dict = torch.load("modelsCGAN_DCGAN/model_g_epoch_15.pth")['state_dict']
    generator.load_state_dict(state_dict)
    generator = generator.cuda()

    _, y = dataset_as_numpy(train_loader.dataset)
    c = Counter(y)
    max_total = c.most_common()[0][1]
    labels = []
    for l in c:
        labels.extend([l] * (max_total - c[l]))
    labels = np.array(labels)
    np.random.shuffle(labels)

    images = _sample(labels=labels, generator=generator)
    images = torch.cat(images)
    labels = torch.LongTensor(labels)
    d = TensorDataset(x=images.cpu().detach(), y=labels.cpu().detach())

    new_d = ConcatDataset([d, train_loader.dataset])
    new_train_loader = DataLoader(new_d, num_workers=train_loader.num_workers,
                                  batch_size=train_loader.batch_size,
                                  shuffle=True)
    return new_train_loader
