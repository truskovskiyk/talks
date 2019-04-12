from torchvision import models
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Tuple
import torch.nn as nn
from thop import profile
from tqdm import tqdm


def plot(result: pd.DataFrame):
    sns.set(rc={'figure.figsize': (16, 16)})
    p1 = sns.regplot(data=result, x="gflops", y="params", fit_reg=False,
                     marker="o",
                     color="skyblue", scatter_kws={'s': 400})

    # add annotations one by one with a loop
    for line in range(0, result.shape[0]):
        print(result.model_name[line])
        p1.text(result.gflops[line] + 0.2, result.params[line],
                result.model_name[line],
                horizontalalignment='left', size='medium', color='black',
                weight='semibold')
    plt.show()


def get_image_net_models() -> List[Tuple[str, nn.Module]]:
    return [
        ("vgg11", models.vgg11(pretrained=False)),
        ("vgg13", models.vgg13(pretrained=False)),
        ("vgg16", models.vgg16(pretrained=False)),
        ("vgg19", models.vgg19(pretrained=False)),
        ("resnet18", models.resnet18(pretrained=False)),
        ("resnet34", models.resnet34(pretrained=False)),
        ("resnet50", models.resnet50(pretrained=False)),
        ("resnet101", models.resnet101(pretrained=False)),
        ("resnet152", models.resnet152(pretrained=False)),
        ("densenet121", models.densenet121(pretrained=False)),
        ("densenet161", models.densenet161(pretrained=False)),
        ("densenet169", models.densenet169(pretrained=False)),
        ("densenet201", models.densenet201(pretrained=False)),
    ]


def build_image_net_plot():
    image_net_models = get_image_net_models()
    input_size = (1, 3, 224, 224)
    result = []
    for model_name, model in tqdm(image_net_models):
        model.eval()
        flops, params = profile(model, input_size=input_size)
        gflops = flops / 1_000_000_000
        params = params / 1_000_000
        result.append(
            {"model_name": model_name, "gflops": gflops, "params": params})
    result = pd.DataFrame(result)
    print(result)
    plot(result)


def build_mnist_plot():
    from kd.models import NetStudent, NetTeacher
    image_net_models = [
        ('NetStudent', NetStudent()),
        ('NetTeacher', NetTeacher())
    ]
    input_size = (1, 1, 28, 28)
    result = []
    for model_name, model in tqdm(image_net_models):
        model.eval()
        flops, params = profile(model, input_size=input_size)
        gflops = flops / 1_000_000_000
        params = params / 1_000_000
        result.append(
            {"model_name": model_name, "gflops": gflops, "params": params})
    result = pd.DataFrame(result)
    print(result)
    plot(result)


if __name__ == "__main__":
    build_image_net_plot()
