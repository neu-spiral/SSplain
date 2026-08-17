# SSplain: Sparse and Smooth Explainer

This repository includes the code used in our work:

> Sunger, E., Imbiriba, T., Campbell, P., Erdogmus, D., Ioannidis, S., Dy, J.
> (2026). *SSplain: Sparse and Smooth Explainer for Retinopathy of Prematurity
> Classification.* In 2026 IEEE/CVF Winter Conference on Applications of Computer
> Vision (WACV), pp. 1705–1715. IEEE.

SSplain produces attribution maps that are **sparse** (few active pixels) *and*
**smooth** (spatially coherent) for an image classifier, by solving a
constrained optimization problem with **ADMM** (Algorithm 1 in the paper).

The core method lives in the self-contained **`ssplain/`** package (`ssplain/core.py`). Given
an image, a model, and hyperparameters, it returns an attribution map and can
visualize it. Everything dataset-specific (loading data, training the models) is
under **`examples/`**, and all the analysis/plotting code is under
**`visualizations/`**.

> **Datasets.** The paper is applied to a private Retinopathy of Prematurity
> (ROP) dataset whose data and weights cannot be shared, so ROP is not included
> here. **MNIST** and **FashionMNIST** are provided as complete, runnable
> examples of the same method.

---

## Repository layout

```
ssplain/
    __init__.py          exposes the public API:  from ssplain import ssplain, visualize
    core.py              the standalone explainer:  ssplain(image, model, ...) + visualize(...)
requirements.txt
examples/
    models.py            LeNet classifier (shared)
    data.py              MNIST / FashionMNIST loaders + get_one_image()
    example_mnist.py     end-to-end: load model → one image → SSplain → save figure
    example_fmnist.py    same, on FashionMNIST
    train.py             train LeNet and save a checkpoint (run this first)
    ckpt/                where trained checkpoints are written (empty until you train)
visualizations/
    metrics.py           insertion / deletion post-hoc accuracy metrics
    figures.py           SSplain-only post-hoc figures: maps + insertion/deletion curves
```

Anything generated (attribution maps, figures) is written to a `results/`
folder created at runtime.

---

## `ssplain` API

```python
from ssplain import ssplain, visualize

attribution = ssplain(
    image, model,          # image: (H,W)/(C,H,W)/(1,C,H,W), any #channels; model: eval() classifier
    label=None,            # target class; int = that class, None = model's prediction
    sparsity=4.0,          # alpha; keeps n_active / sparsity pixels (4.0 = 25% of ||X||_0)
    max_iteration=20,      # ADMM iterations
    grad_steps=1,          # gradient steps on the mask M per ADMM iteration
    admm_lr=0.1,           # mask learning rate
    rho=0.01,              # ADMM penalty / pruning rate
    lambda_tv=1e-3,        # total-variation (smoothness) weight (paper: 1e-3 MNIST, 1e-4 FMNIST)
    mask_init="normalized",# "normalized" | "ones" | "random"
    s1="l0",               # sparsity constraint: "l0" | "l1"
    use_s0=True,           # restrict support to non-background pixels
)                          # -> (H, W) attribution tensor

visualize(image, attribution, label=label, save_path="results/out.png")
```

Two things worth knowing:

* **SSplain runs on the CPU.** The ADMM inner loop uses NumPy projections, so
  `ssplain` moves your model to CPU internally. This is independent of how the
  rest of your code uses a GPU. On the small LeNet models here, a single image
  takes well under a second.
* **The explanation is always a single-channel `(H, W)` map.** Multi-channel
  inputs are supported: for an RGB image the one spatial mask is shared across
  all channels (it multiplies every channel identically), and support selection
  and initialization use the per-pixel mean over channels. Grayscale inputs
  behave exactly as a 1-channel case.

---

## Setup

Tested on **Python 3.10** (3.9–3.11 fine); runs on **macOS, Linux, CPU, and
NVIDIA GPU** with identical commands. Install with pip into a fresh
environment:

```bash
python -m venv .venv
source .venv/bin/activate           # macOS / Linux  (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

If you prefer conda, just use it for the interpreter and still install the
packages with pip:

```bash
conda create -n ssplain python=3.10 -y
conda activate ssplain
pip install -r requirements.txt
```

---

## Try it step by step


### Step 1 - Train a classifier

No weights ship with the repo, so train the small LeNet-5 first (datasets
download automatically to `examples/datasets/` on first run):

```bash
python examples/train.py --dataset mnist
python examples/train.py --dataset fmnist      # when you want FashionMNIST too
```

This matches the paper: Adam (lr 0.001, weight decay 0.0001), batch size 32,
up to 50 epochs, with early stopping (stops after 10 epochs with no validation
improvement; override with `--patience`). Each run writes
`examples/ckpt/<DATASET>_batchsize32_lr0.001_weight_decay0.0001_model_best.pth.tar`
- exactly the filename the example scripts look for. Training uses a GPU
(CUDA/MPS) automatically if one is available, otherwise CPU.

### Step 2 - Run SSplain on one MNIST image

```bash
python examples/example_mnist.py --index 0
```

This loads the LeNet you trained in Step 2, explains test image `#0`, and saves
`results/mnist_example.png` (input on the left, SSplain map on the right).
A window also opens unless you pass `--no_show`. (If you skipped Step 2 you'll
get a clear "no checkpoint - train one first" message.)

Try different images and hyperparameters:

```bash
python examples/example_mnist.py --index 7 --sparsity 3.0 --max_iteration 30 --lambda_tv 1e-4
```

- Increase `--sparsity` → fewer active pixels.
- Increase `--lambda_tv` → smoother, more connected map.
- Increase `--max_iteration` → more ADMM refinement (slower).

By default SSplain explains the **ground-truth** label. Use `--target pred` to
explain the model's **predicted** class instead:

```bash
python examples/example_mnist.py --index 0 --target pred
```

### Step 3 - Run SSplain on FashionMNIST

```bash
python examples/example_fmnist.py --index 0
```

Saves `results/fmnist_example.png`. Same knobs as Step 3.

### Step 4 - Use the API directly (your own loop / notebook)

```python
import torch
from ssplain import ssplain, visualize
from examples.models import LeNet
from examples.data import get_one_image

model = LeNet()
ckpt = torch.load("examples/ckpt/MNIST_batchsize32_lr0.001_weight_decay0.0001_model_best.pth.tar",
                  map_location="cpu")
model.load_state_dict(ckpt["model_state"]); model.eval()

image, label = get_one_image("mnist", index=3)
attr = ssplain(image, model, label=label, sparsity=4.0, max_iteration=20)
visualize(image, attr, label=label)
```

To plug in **your own model**, pass any `torch.nn.Module` in `eval()` mode as
`model`. The input may be grayscale or multi-channel (e.g. RGB); the returned
attribution is a single-channel map either way. Nothing else changes.

### Step 5 (optional) - Post-hoc accuracy metrics

```python
from visualizations.metrics import Insertion, Deletion
```

`Insertion` / `Deletion` progressively add / remove the most important pixels
(as ranked by an attribution) and report balanced accuracy at each step - the
insertion/deletion curves from the paper.

### Step 6 (optional) - SSplain post-hoc figures

`visualizations/figures.py` computes SSplain attributions for a batch of test
images and writes two paper-style figures (SSplain only, no baselines):

```bash
python visualizations/figures.py --dataset mnist --n 8
```

It produces, in `results/plots/` (the target is baked into the filename):

- `mnist_truth_target_ssplain_maps.png` - a grid of inputs (top) and their SSplain maps (bottom)
- `mnist_truth_target_ssplain_insertion_deletion.png` - the insertion / deletion accuracy curves

(with `--target pred` the names become `mnist_pred_target_...`)

It reuses the same SSplain knobs (`--sparsity`, `--max_iteration`, `--admm_lr`,
`--rho`, `--lambda_tv`) plus `--step` for the insertion/deletion resolution, and
`--target {truth,pred}` to explain the ground-truth label (default) or the
model's prediction. It needs a checkpoint from Step 2.

`--n` controls how many maps are shown; `--n_metrics` (default 500) controls how
many images the insertion/deletion curves are averaged over. Keep `--n_metrics`
in the hundreds: with only a handful of images the curve endpoints collapse to a
misleading 0% / 100% (an all-zero image maps to whatever single class the model
predicts, and a tiny batch is either all-right or all-wrong), whereas over many
images the endpoints settle to the true baseline accuracy and the model's test
accuracy.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{sunger2026ssplain,
  title={SSplain: Sparse and Smooth Explainer for Retinopathy of Prematurity Classification},
  author={Sunger, Elifnur and Imbiriba, Tales and Campbell, Peter and Erdogmus, Deniz and Ioannidis, Stratis and Dy, Jennifer},
  booktitle={2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1705--1715},
  year={2026},
  organization={IEEE}
}
```

---
