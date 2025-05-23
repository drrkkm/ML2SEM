import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange

from hw3.hparams import config


wandb.init(config=config, project="effdl_example", name="baseline")


def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        # transforms.Resize((224, 224)),
    ])

    train_dataset = CIFAR10(root='CIFAR10/train',
                            train=True,
                            transform=transform,
                            download=False,
                            )

    test_dataset = CIFAR10(root='CIFAR10/test',
                           train=False,
                           transform=transform,
                           download=False,
                           )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config["batch_size"])


    model = resnet18(pretrained=False, num_classes=10, zero_init_residual=config["zero_init_residual"])
    model.to(device)

    if not config.get("debug_one_batch", False):
        wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    max_epochs = 1 if config.get("debug_one_batch", False) else config["epochs"]

    for epoch in trange(max_epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                all_preds = []
                all_labels = []

                for test_images, test_labels in test_loader:
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    with torch.inference_mode():
                        outputs = model(test_images)
                        preds = torch.argmax(outputs, 1)

                        all_preds.append(preds)
                        all_labels.append(test_labels)

                accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))

                metrics = {'test_acc': accuracy, 'train_loss': loss}
                if not config.get("debug_one_batch", False):
                    wandb.log(metrics, step=epoch * len(train_dataset) + (i + 1) * config["batch_size"])
                else:
                    print(f"[Debug] Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
            if config.get("debug_one_batch", False):
                break
    if not config.get("debug_one_batch", False):
        torch.save(model.state_dict(), "model.pt")

        with open("run_id.txt", "w+") as f:
            print(wandb.run.id, file=f)

