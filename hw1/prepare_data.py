from torchvision.datasets import CIFAR10


def prepare_data():
    train_dataset = CIFAR10("CIFAR10/train", download=True)
    test_dataset = CIFAR10("CIFAR10/test", download=True)
    