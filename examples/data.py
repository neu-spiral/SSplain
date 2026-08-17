"""Minimal MNIST / FashionMNIST loaders for the examples.

Datasets are downloaded on first use to ``examples/datasets/``.
"""

import os

import torch
from torchvision import datasets, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "datasets")

_DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
}


def get_test_loader(name="mnist", batch_size=500, shuffle=False):
    """Return a DataLoader over the test split of ``name`` in {"mnist","fmnist"}."""
    name = name.lower()
    if name not in _DATASETS:
        raise ValueError(f"name must be one of {list(_DATASETS)}; got {name!r}")
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = _DATASETS[name](root=_ROOT, train=False, download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_one_image(name="mnist", index=0):
    """Return a single ``(image, label)`` pair from the test set.

    image has shape ``(1, 28, 28)``; label is a Python int.
    """
    loader = get_test_loader(name, batch_size=index + 1, shuffle=False)
    x, y = next(iter(loader))
    return x[index], int(y[index])
