## Plan

1. Intro
2. Setup
3. MNIST trainer
4. GAN trainer
5. Class imbalance
6. Class imbalance

# Setup
```bash
virtualenv -p python3.6 venv
source ./venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

# Run tensorboard
```bash
tensorboard --logdir . --port 8888 --host 0.0.0.0
```

# MNIST trainer
```bash
python experiments/train_mnist.py --config-path ./configs/train_mnist.json
```

# MNIST GAN trainer
```bash
python experiments/train_gan.py  --config-path ./configs/train_gan.json
```

# Class Imbalance
```bash
python experiments/class_imbalance.py  --config-path ./configs/class_imbalance.json
```


# References
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* [A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/abs/1710.05381)
* [BAGAN: Data Augmentation with Balancing GAN](https://arxiv.org/abs/1803.09655)
* [Deep Imbalanced Learning for Face Recognition and Attribute Prediction](https://arxiv.org/abs/1806.00194)
* [A Classification–Based Study of Covariate Shift in GAN Distributions](http://proceedings.mlr.press/v80/santurkar18a/santurkar18a.pdf)
* [Which Training Methods for GANs do actually Converge?](http://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf)
* [An empirical study on evaluation metrics of generative adversarial networks](https://arxiv.org/abs/1806.07755)
* [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
