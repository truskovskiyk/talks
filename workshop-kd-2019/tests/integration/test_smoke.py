def test_smoke():
    from kd.utils import get_mnist, get_mnist_loaders
    from kd.models import NetTeacher, NetStudent
    from kd.train_mnist import get_config, get_model
    from kd.trainer import MNISTTrainer

    assert 1 == 1
