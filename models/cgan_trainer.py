import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter

from models.utils import weights_init_normal
from models.mnist_cgan_models import Generator, Discriminator


class ConditionalGANTrainer:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, device, train_loader: DataLoader,
                 lr: float, log_interval: int, latent_dim: int, epochs: int):
        self.adversarial_loss = torch.nn.BCELoss().to(device)

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        b1 = 0.5
        b2 = 0.999

        self.optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

        self.train_loader = train_loader
        self.n_epochs = epochs
        self.device = device
        self.latent_size = latent_dim
        self.summary_writer = SummaryWriter()
        self.log_interval = log_interval
        self.n_show_samples = 10
        self.global_step = 0

    def train_model(self):
        for epoch in range(self.n_epochs):
            for i, (imgs, labels) in enumerate(self.train_loader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = discriminator(gen_imgs, gen_labels)
                g_loss = adversarial_loss(validity, valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = discriminator(real_imgs, labels)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                               d_loss.item(), g_loss.item()))

                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    sample_image(n_row=10, batches_done=batches_done)


# # Loss functions
# adversarial_loss = torch.nn.MSELoss()
#
# # Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
#
# if cuda:
#     generator.cuda()
#     discriminator.cuda()
#     adversarial_loss.cuda()
#
# # Configure data loader
# os.makedirs('../../data/mnist', exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data/mnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize(opt.img_size),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=opt.batch_size, shuffle=True)
#
# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#
# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
#
#
# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     # Sample noise
#     z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
#     # Get labels ranging from 0 to n_classes for n rows
#     labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#     labels = Variable(LongTensor(labels))
#     gen_imgs = generator(z, labels)
#     save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)
