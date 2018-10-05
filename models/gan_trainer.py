import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


from models.utils import weights_init_normal


class MNISTGANTrainer:
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
            for i, (real_imgs, _) in enumerate(self.train_loader):
                batch_size = real_imgs.shape[0]
                real_imgs = real_imgs.to(self.device)
                # Adversarial ground truths
                valid = torch.ones(batch_size, device=self.device)
                fake = torch.zeros(batch_size, device=self.device)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_generator.zero_grad()

                # Sample noise as generator input
                z = torch.randn(batch_size, self.latent_size, device=self.device)
                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_generator.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_discriminator.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_discriminator.step()

                self.global_step += 1

                self.summary_writer.add_scalar('gan_mnist/loss_discriminator', d_loss.item(), self.global_step)
                self.summary_writer.add_scalar('gan_mnist/loss_generator', g_loss.item(), self.global_step)

                if self.global_step % self.log_interval == 0:
                    x = vutils.make_grid(gen_imgs[:self.n_show_samples, :, :, :], normalize=True, scale_each=True)
                    self.summary_writer.add_image('gan_mnist/fake', x, self.global_step)

                    y = vutils.make_grid(real_imgs[:self.n_show_samples, :, :, :], normalize=True, scale_each=True)
                    self.summary_writer.add_image('gan_mnist/real', y, self.global_step)
