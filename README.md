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
```


# MNIST trainer
```bash
python experiments/train_mnist.py --config-path ./configs/train_mnist.json
tensorboard --logdir . --port 8888 --host 0.0.0.0
```