import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn

from kd.models import NetStudent, NetTeacher
from kd.trainer import MNISTTrainer
from kd.utils import get_mnist_loaders

logger = logging.getLogger(__name__)


def init_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_config() -> Dict[str, Union[int, float, str, bool]]:
    parser = argparse.ArgumentParser(description="PyTorch MNIST KD Example")
    parser.add_argument("--config-path", type=Path, required=True, metavar="C")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)

    return config


def get_model(model_type: str) -> nn.Module:
    if model_type == "student":
        return NetStudent()
    elif model_type == "teacher":
        return NetTeacher()
    elif model_type == "distill":
        return NetStudent()
    else:
        raise ValueError(f"wrong type of model_type - {model_type}")


def set_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    config = get_config()
    print(config)
    set_seed(config["seed"])
    log_interval = config["log_interval"]
    lr = config["lr"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    n_epoch = config["n_epoch"]
    name = config["name"]
    distill = config["distill"]
    temperature = config["temperature"]
    alpha = config["alpha"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(
        num_workers=num_workers, batch_size=batch_size
    )
    model = get_model(model_type=model_type).to(device)
    if distill:

        teacher = get_model(model_type="teacher").to(device)
        teacher.load_state_dict(torch.load("teacher.pth"))
        teacher.eval()
    else:
        teacher = None

    print(f"teacher model = {teacher}")
    print(model)
    mnist_trainer = MNISTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=lr,
        device=device,
        log_interval=log_interval,
        n_epoch=n_epoch,
        name=name,
        distill=distill,
        teacher=teacher,
        temperature=temperature,
        alpha=alpha,
    )
    mnist_trainer.train_model()

    torch.save(model.state_dict(), f"{model_type}.pth")


if __name__ == "__main__":
    main()
