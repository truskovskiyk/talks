## Plan

1. Intro
2. Setup
3. Profile network
4. Train classifier
5. Distill knowledge
6. Explore distiller

## Presentation

[Slides](https://docs.google.com/presentation/d/10TYXI0ySctLna9lWDvTMr9HbCf4aqXfagLc0a_OV5mk/edit#slide=id.p5)


# Setup
```bash
virtualenv -p python3.6 venv
source ./venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```


# Profile network
```bash
python kd/profile_nets.py
```

# Run tensorboard
```bash
tensorboard --logdir .
```


# Train classifier
```bash
python kd/train_mnist.py --config-path ./configs/train_mnist_techer.json
python kd/train_mnist.py --config-path ./configs/train_mnist_student.json
```

# Distill knowledge
```bash
python kd/train_mnist.py --config-path ./configs/train_mnist_distill.json
python kd/train_mnist.py --config-path ./configs/train_mnist_distill_unlabeled.json
```


# References
* [Model Compression](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
* [Model compression via distillation and quantization](https://arxiv.org/abs/1802.05668)
* [Hardware-Software Codesign of Accurate, Multiplier-free Deep Neural Networks](https://arxiv.org/abs/1705.04288)
* [Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](https://arxiv.org/abs/1711.05852)
* [N2N learning: Network to Network Compression via Policy Gradient Reinforcement Learning](https://arxiv.org/abs/1709.06030)
* [Faster gaze prediction with dense networks and Fisher pruning](https://arxiv.org/abs/1801.05787)
* [Data-Free Knowledge Distillation for Deep Neural Networks](https://arxiv.org/abs/1710.07535)

