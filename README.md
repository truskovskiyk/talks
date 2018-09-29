## Plan

1. Setup
2. MNIST trainer
3. GAN trainer
4. Class imbalance

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