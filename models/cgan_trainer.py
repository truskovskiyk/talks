import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from common import weights_init_normal


class CGANTrainer:

    def __init__(self, *, generator: nn.Module, discriminator: nn.Module, device, train_loader: DataLoader,
                 lr: float, log_interval: int, latent_dim: int, n_epoch: int, name: str, n_classes: int,
                 b1: float = 0.5, b2: float = 0.999):
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss().to(device)
        # Initialize generator and discriminator
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

        self.summary_writer = SummaryWriter(name)
        self.log_interval = log_interval
        self.latent_dim = latent_dim
        self.n_epoch = n_epoch
        self.train_loader = train_loader
        self.device = device
        self.n_classes = n_classes

        self.global_step = 0
        self.n_show_samples = 10
        self.save_each = 5000
        self.n_row = 10

    def sample_image(self, n_row):
        # Sample noise
        z = torch.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))).to(self.device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = torch.LongTensor(labels).to(self.device)
        gen_imgs = self.generator(z, labels)
        y = vutils.make_grid(gen_imgs, normalize=True, scale_each=True)
        self.summary_writer.add_image('gan_mnist/conditional', y, self.global_step)

    def train_model(self):
        for epoch in range(self.n_epoch):
            for i, (real_imgs, labels) in enumerate(self.train_loader):

                batch_size = real_imgs.shape[0]

                # Adversarial ground truths
                valid = torch.ones(batch_size, device=self.device)
                fake = torch.zeros(batch_size, device=self.device)
                # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))).to(self.device)
                gen_labels = torch.LongTensor(np.random.randint(0, self.n_classes, batch_size)).to(self.device)

                # Generate a batch of images
                gen_imgs = self.generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = self.discriminator(gen_imgs, gen_labels)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                validity_real = self.discriminator(real_imgs, labels)
                d_real_loss = self.adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = self.discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = self.adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                self.global_step += 1
                if self.global_step % self.log_interval == 0:
                    self.summary_writer.add_scalar('cgan_mnist/loss_discriminator', d_loss.item(), self.global_step)
                    self.summary_writer.add_scalar('cgan_mnist/loss_generator', g_loss.item(), self.global_step)

                    x = vutils.make_grid(gen_imgs[:self.n_show_samples, :, :, :], normalize=True, scale_each=True)
                    self.summary_writer.add_image('cgan_mnist/fake', x, self.global_step)

                    y = vutils.make_grid(real_imgs[:self.n_show_samples, :, :, :], normalize=True, scale_each=True)
                    self.summary_writer.add_image('cgan_mnist/real', y, self.global_step)

                    self.sample_image(n_row=self.n_row)

                if self.global_step % self.save_each == 0:
                    state_dict = self.generator.state_dict()
                    torch.save(state_dict, f"generator_{self.global_step}.pth")
