"""Minimal end-to-end SSplain example on MNIST.

Run from the repository root:

    python examples/example_mnist.py --index 0 --sparsity 2.0 --max_iteration 20

It loads the trained LeNet (see examples/train.py), picks one test image, generates a SSplain
attribution with the standalone ``ssplain`` API, and saves a visualization to
``results/mnist_example.png``.
"""

import argparse
import os
import sys

import torch

# make the top-level ssplain.py importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssplain import ssplain, visualize  
from models import LeNet  
from data import get_one_image  

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(
    _HERE, "ckpt", "MNIST_batchsize32_lr0.001_weight_decay0.0001_model_best.pth.tar"
)


def load_model(ckpt_path=CKPT):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}.\n"
            "Train one first:  python examples/train.py --dataset mnist"
        )
    model = LeNet()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=0, help="which test image to explain")
    p.add_argument("--target", choices=["truth", "pred"], default="truth",
                   help="explain the ground-truth label ('truth') or the model's prediction ('pred')")
    p.add_argument("--sparsity", type=float, default=4.0)      # alpha = 25% of ||X||_0
    p.add_argument("--max_iteration", type=int, default=20)
    p.add_argument("--admm_lr", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--lambda_tv", type=float, default=1e-3)     # MNIST
    p.add_argument("--no_show", action="store_true", help="don't open a window")
    args = p.parse_args()

    model = load_model()
    image, label = get_one_image("mnist", index=args.index)
    print(f"Explaining MNIST test image #{args.index} (true label {label})")

    # target class: ground-truth label, or None -> ssplain uses the model's prediction
    ssplain_label = None if args.target == "pred" else label

    result = ssplain(
        image, model,
        label=ssplain_label,
        sparsity=args.sparsity,
        max_iteration=args.max_iteration,
        admm_lr=args.admm_lr,
        rho=args.rho,
        lambda_tv=args.lambda_tv,
        progress=True,
        return_full=True,
    )
    attribution = result["attribution"]
    used_label = result["label"]
    print(f"Explained target = {args.target} (class {used_label})")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "mnist_example.png")
    visualize(image, attribution, label=used_label, save_path=out, show=not args.no_show)
    print(f"Saved visualization to {out}")


if __name__ == "__main__":
    main()