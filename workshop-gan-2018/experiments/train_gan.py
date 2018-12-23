import argparse
import json
from pathlib import Path
import torch

from models import Generator, Discriminator, MNISTGANTrainer
from common import get_mnist_loaders


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch MNIST GAN Example')
    parser.add_argument('--config-path', type=Path, required=True, metavar='C')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    return config


def get_model(latent_dim, img_shape, model_type="fc"):
    if model_type == "fc":
        generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        discriminator = Discriminator(img_shape=img_shape)
        return generator, discriminator
    else:
        raise ValueError(f"wrong type of {model_type}")


def main():
    config = get_config()

    seed = config['seed']
    lr = config['lr']
    log_interval = config['log_interval']
    latent_dim = config['latent_dim']
    img_shape = tuple(config["image_shape"])
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    n_epoch = config['n_epoch']
    model_type = config['model_type']
    name = config['name']

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator, discriminator = get_model(latent_dim=latent_dim, img_shape=img_shape, model_type=model_type)
    train_loader, _ = get_mnist_loaders(num_workers=num_workers, batch_size=batch_size)

    mnist_gan_trainer = MNISTGANTrainer(generator=generator,
                                        discriminator=discriminator,
                                        device=device,
                                        train_loader=train_loader,
                                        lr=lr,
                                        log_interval=log_interval,
                                        latent_dim=latent_dim,
                                        n_epoch=n_epoch,
                                        name=name)
    mnist_gan_trainer.train_model()


if __name__ == '__main__':
    main()
