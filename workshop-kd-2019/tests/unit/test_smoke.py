def test_smoke():
    from kd.utils import get_mnist, get_mnist_loaders  # noqa: F401
    from kd.models import NetTeacher, NetStudent  # noqa: F401
    from kd.train_mnist import get_config, get_model  # noqa: F401
    from kd.trainer import MNISTTrainer  # noqa: F401

    assert 1 == 1
