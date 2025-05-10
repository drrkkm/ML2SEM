import pytest
import torch
import shutil
import os
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from prepare_data import prepare_data
from train import main, compute_accuracy
from compute_metrics import main as metrics_main
from hparams import config


@pytest.fixture
def train_dataset():
    shutil.rmtree('CIFAR10')
    prepare_data()
    assert CIFAR10(root='CIFAR10/train',
                            train=True,
                            transform=transform,
                            download=False,
                            )

    assert CIFAR10(root='CIFAR10/test',
                           train=False,
                           transform=transform,
                           download=False,
                           )


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    original_debug = config.get("debug_one_batch", False)
    original_epochs = config["epochs"]

    config["debug_one_batch"] = True
    config["epochs"] = 1

    try:
        main()
    finally:
        config["debug_one_batch"] = original_debug
        config["epochs"] = original_epochs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model_path = "model.pt"
    run_id_path = "run_id.txt"
    metrics_path = "final_metrics.json"

    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(run_id_path):
        os.remove(run_id_path)

    main() 

    assert os.path.exists(model_path), "Модель не была сохранена в model.pt"
    assert os.path.exists(run_id_path), "wandb run_id не был сохранён"

    state_dict = torch.load(model_path, map_location=device)
    assert isinstance(state_dict, dict), "Содержимое model.pt не является state_dict"

    with open(run_id_path) as f:
        run_id = f.read().strip()
        assert len(run_id) > 0, "Файл run_id.txt пустой"

    metrics_main()

    assert os.path.exists(metrics_path), "final_metrics.json не создан"

    with open(metrics_path) as f:
        metrics = json.load(f)
        assert "accuracy" in metrics, "В metrics нет ключа 'accuracy'"
        assert 0 <= metrics["accuracy"] <= 1, "accuracy вне диапазона [0, 1]"