"""Train Mini Neo-Unify with alternating understanding/generation mini-batches (PyTorch)."""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math

from neo_unify_torch.shared.data import generate_dataset, CLASSES
from neo_unify_torch.phase_c.model import NeoUnifyModel
from neo_unify_torch.shared.utils import (
    save_image_grid, plot_losses, plot_dual_losses,
    WEIGHTS_DIR, ensure_weights_dir, get_device,
)


def make_lr_lambda(warmup_steps, total_steps, base_lr=3e-4, min_lr=1e-6):
    """Warmup + cosine decay schedule as a lambda for LambdaLR."""
    def lr_lambda(step):
        if step < warmup_steps:
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def train():
    print("=" * 60)
    print("Phase C: Training Mini Neo-Unify (MoT) [PyTorch]")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Generate data
    print("\nGenerating synthetic dataset...")
    images, labels = generate_dataset(samples_per_class=3000)
    images = images.to(device)
    labels = labels.to(device)
    print(f"Dataset: {images.shape} images, {labels.shape} labels")

    # Model
    model = NeoUnifyModel(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Neo-Unify parameters: {n_params:,}")

    # Optimizer with warmup + cosine decay
    num_epochs = 200
    batch_size = 64
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size
    total_steps = num_epochs * steps_per_epoch * 2  # 2 updates per step

    warmup_steps = steps_per_epoch * 5 * 2
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps)
    )

    # Training: alternating mini-batch updates
    und_losses = []
    gen_losses = []
    t0 = time.time()

    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_und_loss = 0.0
        epoch_gen_loss = 0.0

        model.train()
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch_images = images[idx]
            batch_labels = labels[idx]

            # 1. Understanding update (cross-entropy)
            logits = model.forward_understand(batch_images)
            und_loss = F.cross_entropy(logits, batch_labels)

            optimizer.zero_grad()
            und_loss.backward()
            optimizer.step()
            scheduler.step()

            # 2. Generation update (flow matching MSE)
            B = batch_images.shape[0]
            t = torch.rand(B, device=device)
            noise = torch.randn_like(batch_images)
            t_expanded = t.reshape(B, 1, 1, 1)
            x_t = (1 - t_expanded) * noise + t_expanded * batch_images
            v_target = batch_images - noise
            v_pred = model.forward_generate(x_t, t, batch_labels)
            gen_loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            gen_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_und_loss += und_loss.item()
            epoch_gen_loss += gen_loss.item()
            und_losses.append(und_loss.item())
            gen_losses.append(gen_loss.item())

        avg_und = epoch_und_loss / steps_per_epoch
        avg_gen = epoch_gen_loss / steps_per_epoch

        if (epoch + 1) % 15 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Und CE: {avg_und:.4f} | "
                f"Gen MSE: {avg_gen:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Plot losses
    plot_dual_losses(
        und_losses, gen_losses,
        "Understanding CE", "Generation MSE",
        "neo_unify_losses_torch.png",
        title="Neo-Unify Training Losses (PyTorch)",
    )
    plot_losses(und_losses, "neo_unify_und_loss_torch.png", title="Neo-Unify Understanding Loss (PyTorch)")
    plot_losses(gen_losses, "neo_unify_gen_loss_torch.png", title="Neo-Unify Generation Loss (PyTorch)")

    # Save model weights
    ensure_weights_dir()
    torch.save(model.state_dict(), f"{WEIGHTS_DIR}/neo_unify_torch.pt")
    print(f"Saved: {WEIGHTS_DIR}/neo_unify_torch.pt")

    return model


if __name__ == "__main__":
    train()
