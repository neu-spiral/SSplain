"""
SSplain: Sparse and Smooth Explainer  (standalone reference implementation)
===========================================================================

Given a single image and a trained PyTorch classifier, ``ssplain`` produces a
sparse *and* smooth attribution map by solving a constrained optimization
problem with ADMM (Algorithm 1 in the paper). ``visualize`` renders the result.

This file is self-contained: it depends only on ``torch``, ``numpy`` and
``torchmetrics`` (and ``matplotlib`` for ``visualize``). It does not import any
other file in this repository.

Quick start
-----------
    from ssplain import ssplain, visualize

    model.eval()
    attribution = ssplain(image, model, label=target_class)   # (H, W) tensor
    visualize(image, attribution, label=target_class)

Notes
-----
* The ADMM inner loop uses NumPy projections, so SSplain runs on the **CPU**.
  ``ssplain`` moves ``model`` to CPU for you. This is unrelated to how the rest
  of your pipeline uses a GPU; it only affects this explainer.
* The explanation is always a **single-channel** spatial map (H, W). For a
  multi-channel input (e.g. an RGB image), that one mask is shared across all
  channels: it multiplies every channel identically, and support selection /
  initialization use the per-pixel mean over channels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.image import TotalVariation


__all__ = ["ssplain", "visualize"]


# ---------------------------------------------------------------------------
# L1-ball / positive-simplex Euclidean projection.
# Adapted from Duchi, Shalev-Shwartz, Singer & Chandra (ICML 2008),
# via A. Gaidon's public implementation. Used only when s1="l1".
# ---------------------------------------------------------------------------
def _euclidean_proj_simplex(v, u, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    if v.sum() == s and np.all(v >= 0):
        return v
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = float(cssv[rho] - s) / rho
    return (v - theta).clip(min=0)


def _euclidean_proj_l1ball(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    u = np.abs(v)
    u_sort = np.sort(u)[::-1]
    s_new = int(sum(u_sort[: s - 1]))
    if u.sum() <= s_new:
        return v
    w = _euclidean_proj_simplex(u, u_sort, s=s_new)
    w *= np.sign(v)
    return w


# ---------------------------------------------------------------------------
# ADMM state and updates (Eqs. 5, 8, 9 in the paper).
# ---------------------------------------------------------------------------
class _ADMM:
    """Holds the ADMM variables and their optimizer for one image."""

    def __init__(self, M_initial, indices, x, y, device, active_pixels,
                 admm_lr, rho, lambda_tv, initialization, s1):
        self.indices = indices
        self.x = x
        self.y = y
        self.device = device
        self.active_pixels = active_pixels
        self.rho = rho
        self.lambda_tv = lambda_tv
        self.s1 = s1

        M = M_initial.detach().clone().cpu().numpy()

        if initialization == "normalized":
            M = (M - M.min()) / (M.max() - M.min())
            self.M = torch.tensor(M, requires_grad=True, dtype=torch.float32).to(device)
        elif initialization == "random":
            self.M = torch.rand(M.shape, requires_grad=True).to(device)
        elif initialization == "ones":
            self.M = torch.ones(M.shape, requires_grad=True, dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Unknown mask_init: {initialization!r}")

        self.M1 = torch.Tensor(M).to(device)
        self.M2 = torch.Tensor(M).to(device)
        self.U1 = torch.zeros(M.shape).to(device)
        self.U2 = torch.zeros(M.shape).to(device)

        self.optimizer = torch.optim.Adam([self.M], admm_lr)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.tv = TotalVariation(reduction="none")


def _M1_update(M, U, active_pixels, device, s1):
    """Project (M + U) onto the sparsity set S1 (Eq. 9)."""
    Z = (M + U).detach().cpu().numpy()
    if s1 == "l0":
        idx = np.argsort(Z.ravel())[: -active_pixels - 1 : -1]
        idx = np.column_stack(np.unravel_index(idx, Z.shape))
        z = np.zeros(Z.shape)
        z[idx[:, 0], idx[:, 1]] = Z[idx[:, 0], idx[:, 1]]
    elif s1 == "l1":
        z = _euclidean_proj_l1ball(Z.flatten(), active_pixels).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown s1: {s1!r}")
    return torch.from_numpy(z).to(M.dtype).to(device)


def _M2_update(M, U2, device):
    """Project (M + U2) onto the box set S2 = [0, 1] (Eq. 9)."""
    return torch.clamp((M + U2).detach(), min=0.0, max=1.0)


def _admm_step(A, model, grad_steps=1):
    _, _, H, W = A.x.shape
    # --- primal M-update: one or more gradient steps on the mask (Eq. 5a) ---
    for _ in range(grad_steps):
        A.optimizer.zero_grad()

        # single-channel (H, W) mask; broadcasts across all input channels,
        # so a 3-channel (RGB) image is masked by the same spatial map.
        mask_full = torch.zeros([1, 1, H, W], device=A.device)
        mask_full[0, 0, A.indices[:, 0], A.indices[:, 1]] = A.M[:, 0]
        masked_input = mask_full * A.x                      # (1, C, H, W)

        output = model(masked_input)
        ce = A.criterion(output, A.y)

        admm_loss = 0.5 * A.rho * (
            (torch.norm(A.M - A.M1 + A.U1, p=2, dim=[0, 1]) ** 2)
            + (torch.norm(A.M - A.M2 + A.U2, p=2, dim=[0, 1]) ** 2)
        )

        tv_loss = A.lambda_tv * A.tv(mask_full)

        loss = admm_loss + ce + tv_loss
        loss.backward(retain_graph=True)
        A.optimizer.step()

    # --- auxiliary/dual updates: once per ADMM iteration (Eqs. 8 and 5c) ---
    A.M1 = _M1_update(A.M, A.U1, A.active_pixels, A.device, A.s1)
    A.U1 = A.rho * (A.M - A.M1) + A.U1
    A.M2 = _M2_update(A.M, A.U2, A.device)
    A.U2 = A.rho * (A.M - A.M2) + A.U2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ssplain(
    image,
    model,
    *,
    label=None,
    sparsity=4.0,
    max_iteration=20,
    grad_steps=1,
    admm_lr=0.1,
    rho=0.01,
    lambda_tv=1e-3,
    mask_init="normalized",
    s1="l0",
    use_s0=True,
    return_full=False,
    progress=False,
):
    """Generate a sparse-and-smooth SSplain attribution for one image.

    Args:
        image (torch.Tensor): image, shape ``(H, W)``, ``(C, H, W)`` or
            ``(1, C, H, W)``. Any number of channels is accepted (e.g. 1 for
            grayscale, 3 for RGB); the returned attribution is single-channel.
            Pixel values as fed to the model.
        model (torch.nn.Module): trained classifier. Moved to CPU and set to
            eval() internally.
        label (int, optional): target class to explain. Defaults to the model's
            predicted class for ``image``.
        sparsity (float): sparsity level alpha. The number of retained active
            pixels is ``n_active / sparsity`` (higher -> sparser). Default 4.0
            (keeps 25% of the nonzero pixels, as in the paper).
        max_iteration (int): number of ADMM iterations. Default 20.
        grad_steps (int): number of gradient (Adam) steps taken on the mask M
            within each ADMM iteration, before the projection/dual updates.
            Default 1 (the standard single-step primal update).
        admm_lr (float): learning rate of the ADMM mask optimizer. Default 0.1.
        rho (float): ADMM penalty / pruning rate. Default 0.01.
        lambda_tv (float): total-variation (smoothness) weight. Default 1e-3
            (the paper uses 1e-3 for MNIST and 1e-4 for FashionMNIST).
        mask_init ({"normalized","ones","random"}): mask initialization.
        s1 ({"l0","l1"}): sparsity constraint type. Default "l0".
        use_s0 (bool): if True, restrict the explanation to non-background
            pixels (values above the image minimum). If False, use every pixel.
        return_full (bool): if True, also return the raw ``(1, 1, H, W)`` map
            and metadata as a dict.
        progress (bool): if True, show a tqdm progress bar over ADMM iterations.

    Returns:
        torch.Tensor of shape ``(H, W)`` with the attribution scores, or, if
        ``return_full=True``, a dict with keys ``attribution`` (H, W),
        ``attribution_full`` (1, 1, H, W), ``label`` and ``indices``.
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # --- normalize input shape to (1, 1, H, W) ---
    x = image.detach().to(device).float()
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected a single image; got shape {tuple(image.shape)}")

    # --- target label ---
    if label is None:
        with torch.no_grad():
            label = int(model(x).argmax(dim=1).item())
    y = torch.tensor([int(label)], device=device)

    # --- per-pixel intensity map (H, W) ---
    # The mask is a single spatial map shared across channels, so support
    # selection and initialization use one scalar per pixel: the mean over
    # channels (identical to the pixel value for a 1-channel image).
    intensity = x[0].mean(dim=0)

    # --- S0: restrict support to non-background pixels ---
    threshold = -np.inf if not use_s0 else intensity.min()
    mask_bool = intensity > threshold
    M_0 = intensity[mask_bool].reshape(-1, 1)
    indices = mask_bool.nonzero()

    active_pixels = max(1, int(M_0.shape[0] / sparsity))

    A = _ADMM(
        M_initial=M_0, indices=indices, x=x, y=y, device=device,
        active_pixels=active_pixels, admm_lr=admm_lr, rho=rho,
        lambda_tv=lambda_tv, initialization=mask_init, s1=s1,
    )
    # ADMM initialization (Eq. 9 for M1, M2)
    A.M1 = _M1_update(A.M, A.U1, A.active_pixels, A.device, A.s1)
    A.M2 = _M2_update(A.M, A.U2, A.device)

    iters = range(max_iteration)
    if progress:
        from tqdm import tqdm
        iters = tqdm(iters, desc="SSplain ADMM")
    for _ in iters:
        _admm_step(A, model, grad_steps=grad_steps)

    # --- assemble the full-resolution attribution map ---
    m = A.M
    background = m.min().item() if m.min() < 0 else 0.0
    _, _, H, W = x.shape
    full = torch.zeros([1, 1, H, W], device=device) + background
    full[0, 0, indices[:, 0], indices[:, 1]] = m[:, 0]

    attribution = full[0, 0].detach()
    if return_full:
        return {
            "attribution": attribution,
            "attribution_full": full.detach(),
            "label": int(label),
            "indices": indices.detach(),
        }
    return attribution


def visualize(
    image,
    attribution,
    *,
    label=None,
    cmap="viridis",
    overlay=False,
    alpha=0.6,
    title=None,
    save_path=None,
    show=True,
):
    """Visualize an image next to (or overlaid with) its SSplain attribution.

    Args:
        image (torch.Tensor): the image, shape ``(H, W)``/``(1,H,W)``/``(1,1,H,W)``.
        attribution (torch.Tensor): ``(H, W)`` map from ``ssplain``.
        label (int, optional): shown in the title if given.
        cmap (str): matplotlib colormap for the attribution.
        overlay (bool): if True, overlay the attribution on the image in a
            single panel instead of showing two panels.
        alpha (float): overlay transparency (used when ``overlay=True``).
        title (str, optional): overrides the default title.
        save_path (str, optional): if given, save the figure there.
        show (bool): call ``plt.show()``. Set False in headless/batch runs.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    # normalize the image to (H, W) grayscale or (H, W, 3) RGB for display
    img = image.detach().cpu().float()
    if img.dim() == 4:          # (1, C, H, W)
        img = img[0]
    if img.dim() == 3:          # (C, H, W)
        img = img[0] if img.shape[0] == 1 else img.permute(1, 2, 0)
    img_kw = {"cmap": "gray"} if img.dim() == 2 else {}
    if img.dim() == 3:          # clip RGB to a valid display range
        img = img.clamp(0, 1)
    attr = attribution.detach().cpu().float()

    if overlay:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img, **img_kw)
        im = ax.imshow(attr, cmap=cmap, alpha=alpha)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img, **img_kw)
        axes[0].set_title("input")
        axes[0].axis("off")
        im = axes[1].imshow(attr, cmap=cmap)
        axes[1].set_title("SSplain")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if title is None and label is not None:
        title = f"SSplain (class {label})"
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig