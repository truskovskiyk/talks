import argparse
import os
import numpy as np
import math

import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter

cuda = True if torch.cuda.is_available() else False

from models.cgan_models import Generator, Discriminator

# Configure data loader
# os.makedirs('../../data/mnist', exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data/mnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize(opt.img_size),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=opt.batch_size, shuffle=True)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     # Sample noise
#     z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
#     # Get labels ranging from 0 to n_classes for n rows
#     labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#     labels = Variable(LongTensor(labels))
#     gen_imgs = generator(z, labels)
#     save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)



from torch.utils.data import Dataset



class CGANTrainer:

    def sample_image(self, n_row):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = self.generator(z, labels)

        # save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)

        y = vutils.make_grid(gen_imgs, normalize=True, scale_each=True)
        self.summary_writer.add_image('gan_mnist/conditional', y, self.global_step)

    # ----------
    #  Training
    # ----------
    def train_model(self, dataloader):
        img_shape = (1, 28, 28)
        n_classes = 10
        latent_dim = 100
        n_epochs = 200
        save_each = 5000
        # Loss functions
        adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        generator = Generator(n_classes=n_classes, latent_dim=latent_dim, img_shape=img_shape)
        discriminator = Discriminator(n_classes=n_classes, img_shape=img_shape)

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        lr = 0.0002
        b1 = 0.5
        b2 = 0.999

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

        summary_writer = SummaryWriter()
        global_step = 0
        n_show_samples = 10
        log_interval = 10

        self.latent_dim = latent_dim
        self.generator = generator
        self.summary_writer = summary_writer
        self.global_step = global_step
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        for epoch in range(n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):

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
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, self.n_classes, batch_size)))

                print(z.shape, gen_labels.shape)
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

                self.global_step += 1
                if self.global_step % log_interval == 0:
                    # sample_image(n_row=10, batches_done=batches_done)

                    summary_writer.add_scalar('gan_mnist/loss_discriminator', d_loss.item(), self.global_step)
                    summary_writer.add_scalar('gan_mnist/loss_generator', g_loss.item(), self.global_step)

                    x = vutils.make_grid(gen_imgs[:n_show_samples, :, :, :], normalize=True, scale_each=True)
                    summary_writer.add_image('gan_mnist/fake', x, self.global_step)

                    y = vutils.make_grid(real_imgs[:n_show_samples, :, :, :], normalize=True, scale_each=True)
                    summary_writer.add_image('gan_mnist/real', y, self.global_step)

                    self.sample_image(n_row=10)

                if self.global_step % save_each == 0:
                    state_dict = self.generator.state_dict()
                    torch.save(state_dict, f"generator_{self.global_step}.pth")


if __name__ == '__main__':
    img_size = (1, 28, 28)
    batch_size = 64

    # Configure data loader
    os.makedirs('../../data/mnist', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((28, 28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=batch_size, shuffle=True)

    t = CGANTrainer()
    t.train_model(dataloader)
