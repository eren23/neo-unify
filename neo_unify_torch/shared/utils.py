"""Visualization and helper utilities (PyTorch)."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "outputs"
WEIGHTS_DIR = "weights"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_weights_dir():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)


def get_device():
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_numpy(x):
    """Convert torch tensor to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def save_image_grid(images, filename, title=None, nrow=8):
    """Save a grid of images.

    Args:
        images: tensor of shape (N, C, H, W) in [0, 1] (NCHW)
        filename: output filename (saved in OUTPUT_DIR)
        title: optional title
        nrow: images per row
    """
    ensure_output_dir()
    images = to_numpy(images)
    images = np.clip(images, 0.0, 1.0)

    # NCHW -> NHWC for matplotlib
    if images.ndim == 4 and images.shape[1] in (1, 3):
        images = np.transpose(images, (0, 2, 3, 1))

    n = len(images)
    ncol = nrow
    nrows = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 1.2, nrows * 1.2))
    if nrows == 1:
        axes = [axes] if ncol == 1 else [axes]
    if ncol == 1:
        axes = [[ax] for ax in axes]

    for i in range(nrows):
        for j in range(ncol):
            ax = axes[i][j] if nrows > 1 else axes[0][j]
            idx = i * ncol + j
            if idx < n:
                ax.imshow(images[idx], interpolation="nearest")
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_comparison_grid(image_sets, method_names, class_names, filename="comparison.png"):
    """Side-by-side comparison of N methods.

    Args:
        image_sets: list of tensors, each (n_classes * samples_per_class, C, H, W) NCHW
        method_names: list of method name strings
        class_names: list of class name strings
        filename: output filename
    """
    ensure_output_dir()
    image_sets_np = []
    for imgs in image_sets:
        imgs = to_numpy(imgs)
        # NCHW -> NHWC
        if imgs.ndim == 4 and imgs.shape[1] in (1, 3):
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        image_sets_np.append(imgs)

    n_methods = len(method_names)
    n_classes = len(class_names)
    samples = min(4, len(image_sets_np[0]) // n_classes)

    fig, axes = plt.subplots(
        n_classes, n_methods * samples,
        figsize=(n_methods * samples * 1.2, n_classes * 1.2),
    )

    for cls_idx in range(n_classes):
        for method_idx, (imgs, name) in enumerate(zip(image_sets_np, method_names)):
            for s in range(samples):
                col = method_idx * samples + s
                img_idx = cls_idx * samples + s
                axes[cls_idx][col].imshow(
                    np.clip(imgs[img_idx], 0, 1), interpolation="nearest"
                )
                axes[cls_idx][col].axis("off")
                if cls_idx == 0 and s == 0:
                    axes[cls_idx][col].set_title(name, fontsize=8)

        axes[cls_idx][0].set_ylabel(
            class_names[cls_idx], fontsize=8, rotation=0, labelpad=30, va="center"
        )

    title = " vs ".join(method_names)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_losses(losses, filename, title="Training Loss"):
    """Plot and save training loss curve."""
    ensure_output_dir()
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_dual_losses(losses_a, losses_b, label_a, label_b, filename, title="Training Losses"):
    """Plot two loss curves on the same figure."""
    ensure_output_dir()
    fig, ax1 = plt.subplots(figsize=(8, 4))

    color_a = "tab:blue"
    color_b = "tab:orange"
    ax1.plot(losses_a, color=color_a, alpha=0.7, label=label_a)
    ax1.set_xlabel("Step")
    ax1.set_ylabel(label_a, color=color_a)

    ax2 = ax1.twinx()
    ax2.plot(losses_b, color=color_b, alpha=0.7, label=label_b)
    ax2.set_ylabel(label_b, color=color_b)

    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
