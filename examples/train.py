"""Train the LeNet-5 classifier on MNIST or FashionMNIST.

Trains separately per dataset with Adam (lr 0.001, weight decay 0.0001),
batch size 32, up to 50 epochs, with early stopping on validation accuracy —
the settings used in the paper. No weights ship with the repo, so run this
before the example scripts (or to reproduce / replace a checkpoint).

    python examples/train.py --dataset mnist

Saves ``<DATASET>_batchsize32_lr0.001_weight_decay0.0001_model_best.pth.tar``
into ``examples/ckpt/`` (the name the example scripts expect).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LeNet  

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_HERE, "datasets")
_CKPT_DIR = os.path.join(_HERE, "ckpt")
_DATASETS = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST}


def loaders(name, batch_size):
    tfm = transforms.Compose([transforms.ToTensor()])
    full = _DATASETS[name](root=_DATA_ROOT, train=True, download=True, transform=tfm)
    n_train = int(0.8 * len(full))
    train, val = torch.utils.data.random_split(full, [n_train, len(full) - n_train])
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(_DATASETS), default="mnist")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10,
                   help="early stopping: stop after this many epochs without val-acc improvement")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training {args.dataset} on {device}")

    train_loader, val_loader = loaders(args.dataset, args.batch_size)
    model = LeNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = ReduceLROnPlateau(opt, "min", patience=10)
    loss_fn = nn.CrossEntropyLoss()
    acc = Accuracy(task="multiclass", num_classes=10).to(device)

    os.makedirs(_CKPT_DIR, exist_ok=True)
    name = f"{args.dataset.upper()}_batchsize{args.batch_size}_lr{args.lr}_weight_decay{args.weight_decay}"
    best_path = os.path.join(_CKPT_DIR, name + "_model_best.pth.tar")
    best_acc = 0.0
    counter = 0

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += loss_fn(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
        val_acc = correct / total
        sched.step(val_loss / total)

        is_best = val_acc > best_acc
        print(f"epoch {epoch + 1}/{args.epochs} - val_acc {val_acc:.4f}" + (" [*]" if is_best else ""))

        if is_best:
            best_acc = val_acc
            counter = 0
            torch.save(
                {"epoch": epoch + 1, "model_state": model.state_dict(),
                 "optim_state": opt.state_dict(), "best_valid_acc": best_acc},
                best_path,
            )
        else:
            counter += 1
            if counter > args.patience:
                print(f"[!] No val-acc improvement for {args.patience} epochs; early stopping.")
                break

    print(f"Best val acc {best_acc:.4f}; saved to {best_path}")


if __name__ == "__main__":
    main()