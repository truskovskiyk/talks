from torchvision import models
from thop import profile
import matplotlib.pyplot as plt
from pprint import pprint
from ptflops import get_model_complexity_info
from pthflops import count_ops
import torch

def plot(result):
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(
        [x['gflops'] for x in result],
        [x['params'] for x in result],
        marker='o',
        # c=[x for x in result],
        # s=[x['params'] for x in result],
        cmap=plt.get_cmap('Spectral'))

    for rec in result:
        label, x, y = rec['model_name'], rec['gflops'], rec['params']

        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.show()


models = [
    # ('vgg11', models.vgg11(pretrained=False)),
    # ('vgg13', models.vgg13(pretrained=False)),
    # ('vgg16', models.vgg16(pretrained=False)),
    # ('vgg19', models.vgg19(pretrained=False)),

    # ('resnet18', models.resnet18(pretrained=False)),
    # ('resnet34', models.resnet34(pretrained=False)),
    ('resnet50', models.resnet50(pretrained=False)),
    # ('resnet101', models.resnet101(pretrained=False)),
    # ('resnet152', models.resnet152(pretrained=False)),
    # ('densenet121', models.densenet121(pretrained=False)),
    # ('densenet161', models.densenet161(pretrained=False)),
    # ('densenet169', models.densenet169(pretrained=False)),
    # ('densenet201', models.densenet201(pretrained=False))
]

if __name__ == '__main__':
    result = []
    for model_name, model in models:
        model.eval()
        flops, params = profile(model, input_size=(1, 3, 224, 224))
        gflops = flops / 1_000_000_000
        params = params / 1_000_000
        result.append(
            {'model_name': model_name, 'gflops': gflops, 'params': params})

        flops, params = get_model_complexity_info(model, (224, 224), as_strings=True, print_per_layer_stat=True)
        print(flops, params)

        ops = count_ops(model, torch.rand(1,3,224,224))
        print(ops)
    pprint(result)
    # plot(result)

# https://github.com/pytorch/pytorch/issues/5013
# https://github.com/albanie/convnet-burden
# https://github.com/sovrasov/flops-counter.pytorch
# https://github.com/Lyken17/pytorch-OpCounter
# https://github.com/1adrianb/pytorch-estimate-flops
