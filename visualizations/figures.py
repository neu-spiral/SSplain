"""Post-hoc SSplain figures (SSplain only — no baselines).

Loads a trained LeNet, computes SSplain attributions for a batch of test
images, and writes two figures to ``results/plots/``:

  <dataset>_ssplain_maps.pdf                 qualitative: inputs + SSplain maps
  <dataset>_ssplain_insertion_deletion.pdf   post-hoc accuracy (insertion / deletion)

Run from the repository root:

    python visualizations/figures.py --dataset mnist --n 8

Requires a checkpoint from examples/train.py.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# make the sibling packages importable when run from the repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                                   # ssplain/, examples/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # metrics.py (same folder)

from ssplain import ssplain                       
from examples.models import LeNet                 
from examples.data import get_test_loader         
from metrics import Insertion, Deletion           

_CKPT = {
    "mnist": "MNIST_batchsize32_lr0.001_weight_decay0.0001_model_best.pth.tar",
    "fmnist": "FMNIST_batchsize32_lr0.001_weight_decay0.0001_model_best.pth.tar",
}


def load_model(dataset):
    path = os.path.join(_ROOT, "examples", "ckpt", _CKPT[dataset])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint at {path}.\n"
            f"Train one first:  python examples/train.py --dataset {dataset}"
        )
    model = LeNet()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def compute_ssplain(model, images, labels, target="truth", **kw):
    """SSplain maps (N,1,H,W) plus the class explained for each image.

    target="truth" explains the ground-truth label; target="pred" explains the
    model's predicted class (ssplain picks it when label=None).
    """
    maps, used = [], []
    for i in range(len(labels)):
        lbl = int(labels[i]) if target == "truth" else None
        d = ssplain(images[i], model, label=lbl, return_full=True, **kw)
        maps.append(d["attribution_full"])
        used.append(d["label"])
    return torch.cat(maps, dim=0), torch.tensor(used)


def figure_maps(images, maps, labels, out_path, cmap="viridis"):
    """Qualitative panel: top row = inputs, bottom row = SSplain maps."""
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(2, 1)
    for j in range(n):
        axes[0, j].imshow(images[j][0].cpu(), cmap="gray")
        axes[0, j].set_title(f"{int(labels[j])}", fontsize=9)
        axes[0, j].axis("off")
        axes[1, j].imshow(maps[j, 0].cpu(), cmap=cmap)
        axes[1, j].axis("off")
    fig.text(0.015, 0.72, "input", rotation=90, va="center", fontsize=10)
    fig.text(0.015, 0.28, "SSplain", rotation=90, va="center", fontsize=10)
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def figure_curves(model, images, labels, maps, out_path, step=25):
    """Insertion / deletion post-hoc accuracy curves for SSplain."""
    ins, _ = Insertion(model, images, labels, maps, "SSplain", step=step)
    dele, _ = Deletion(model, images, labels, maps, "SSplain", step=step)
    ins = np.asarray(ins).ravel()
    dele = np.asarray(dele).ravel()
    xs = np.linspace(0, 100, len(ins))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(xs, dele[::-1], "o-", color="#1f77b4")
    a1.set_xlabel("% features deleted")
    a1.set_ylabel("accuracy (%)")
    a1.set_title(r"Deletion ($\downarrow$ better)")
    a1.set_ylim(0, 100)
    a1.invert_xaxis()

    a2.plot(xs, ins, "o-", color="#1f77b4")
    a2.set_xlabel("% features inserted")
    a2.set_ylabel("accuracy (%)")
    a2.set_title(r"Insertion ($\uparrow$ better)")
    a2.set_ylim(0, 100)

    fig.suptitle("SSplain post-hoc post-hoc accuracy")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"SSplain  insertion mean acc = {ins.mean():.2f}   deletion mean acc = {dele.mean():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist")
    p.add_argument("--n", type=int, default=8, help="number of images shown in the maps panel")
    p.add_argument("--n_metrics", type=int, default=500,
                   help="number of images used for the insertion/deletion curves "
                        "(small values make the endpoints look like 0%%/100%%; use a few hundred)")
    p.add_argument("--target", choices=["truth", "pred"], default="truth",
                   help="explain the ground-truth label ('truth') or the model's prediction ('pred')")
    p.add_argument("--sparsity", type=float, default=4.0, help="alpha = 25% of ||X||_0")
    p.add_argument("--max_iteration", type=int, default=20)
    p.add_argument("--admm_lr", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--lambda_tv", type=float, default=None,
                   help="TV weight; default is the paper value per dataset (mnist 1e-3, fmnist 1e-4)")
    p.add_argument("--step", type=int, default=25, help="insertion/deletion steps")
    args = p.parse_args()

    if args.lambda_tv is None:
        args.lambda_tv = 1e-3 if args.dataset == "mnist" else 1e-4

    model = load_model(args.dataset)
    n_total = max(args.n, args.n_metrics)
    x, y = next(iter(get_test_loader(args.dataset, batch_size=n_total, shuffle=False)))
    kw = dict(sparsity=args.sparsity, max_iteration=args.max_iteration,
              admm_lr=args.admm_lr, rho=args.rho, lambda_tv=args.lambda_tv)

    print(f"Computing SSplain for {n_total} {args.dataset} test images (target={args.target}) ...")
    maps, used = compute_ssplain(model, x, y, target=args.target, **kw)

    out_dir = "results/plots"
    tag = f"{args.dataset}_{args.target}_target"   # e.g. mnist_truth_target / mnist_pred_target
    # maps: a small qualitative panel (first --n images); titled by the explained class
    figure_maps(x[:args.n], maps[:args.n], used[:args.n],
                os.path.join(out_dir, f"{tag}_ssplain_maps.png"))
    # curves: measured against the GROUND-TRUTH labels y (accuracy = did the model
    # get the true class). --target only chooses which class SSplain explains; it
    # does not change what the metric scores against.
    figure_curves(model, x[:args.n_metrics], y[:args.n_metrics], maps[:args.n_metrics],
                  os.path.join(out_dir, f"{tag}_ssplain_insertion_deletion.png"),
                  step=args.step)
    print(f"Saved figures to {out_dir}/")


if __name__ == "__main__":
    main()