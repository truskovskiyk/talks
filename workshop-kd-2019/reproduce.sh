echo "clean"
rm -fr mnist_train_*
rm -r *.pth
echo "run experiments"
python kd/train_mnist.py --config-path ./configs/train_mnist_techer.json
python kd/train_mnist.py --config-path ./configs/train_mnist_student.json
python kd/train_mnist.py --config-path ./configs/train_mnist_distill.json
python kd/train_mnist.py --config-path ./configs/train_mnist_distill_unlabeled.json
