"""Exp CFG: Joint gradient training with label dropout, EMA, and longer training (PyTorch)."""

import torch
import torch.nn.functional as F
import numpy as np
import time
import copy
import math

from neo_unify_torch.shared.data import generate_dataset, CLASSES
from neo_unify_torch.exp_cfg.model import NeoUnifyModel
from neo_unify_torch.shared.utils import (
    save_image_grid, plot_losses, plot_dual_losses,
    WEIGHTS_DIR, ensure_weights_dir, get_device,
)


def make_lr_lambda(warmup_steps, total_steps, base_lr=3e-4, min_lr=1e-6):
    def lr_lambda(step):
        if step < warmup_steps:
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def train():
    print("=" * 60)
    print("Exp CFG: Joint Gradient + CFG + EMA [PyTorch]")
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
    print(f"Parameters: {n_params:,}")

    # Optimizer with warmup + cosine decay
    num_epochs = 200
    batch_size = 64
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size
    total_steps = num_epochs * steps_per_epoch  # single update per step

    warmup_steps = steps_per_epoch * 5
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps)
    )

    num_classes = 6
    label_dropout_rate = 0.1
    gen_weight = 3.0

    # Initialize EMA model
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999

    # Training
    t0 = time.time()

    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples, device=device)

        model.train()
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch_images = images[idx]
            batch_labels = labels[idx]

            # Label dropout for generation: 10% -> null class (index num_classes)
            B = batch_labels.shape[0]
            dropout_mask = torch.rand(B, device=device) < label_dropout_rate
            gen_labels = torch.where(
                dropout_mask,
                torch.full((B,), num_classes, dtype=torch.long, device=device),
                batch_labels,
            )

            # Understanding loss: cross-entropy
            logits = model.forward_understand(batch_images)
            und_loss = F.cross_entropy(logits, batch_labels)

            # Generation loss: flow matching MSE (with label-dropped class labels)
            t = torch.rand(B, device=device)
            noise = torch.randn_like(batch_images)
            t_expanded = t.reshape(B, 1, 1, 1)
            x_t = (1 - t_expanded) * noise + t_expanded * batch_images
            v_target = batch_images - noise
            v_pred = model.forward_generate(x_t, t, gen_labels)
            gen_loss = F.mse_loss(v_pred, v_target)

            # Reconstruction loss
            recon = model.forward_reconstruct(batch_images)
            recon_loss = F.mse_loss(recon, batch_images)

            total_loss = und_loss + gen_weight * gen_loss + 1.0 * recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # EMA update
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(ema_decay).add_(model_p, alpha=1 - ema_decay)

        if (epoch + 1) % 30 == 0 or epoch == 0:
            # Log individual losses
            with torch.no_grad():
                sample_idx = perm[:256]
                sample_images = images[sample_idx]
                sample_labels = labels[sample_idx]

                logits = model.forward_understand(sample_images)
                und_loss_val = F.cross_entropy(logits, sample_labels)

                t_sample = torch.rand(256, device=device)
                noise = torch.randn_like(sample_images)
                t_exp = t_sample.reshape(256, 1, 1, 1)
                x_t = (1 - t_exp) * noise + t_exp * sample_images
                v_target = sample_images - noise
                v_pred = model.forward_generate(x_t, t_sample, sample_labels)
                gen_loss_val = F.mse_loss(v_pred, v_target)

                recon = model.forward_reconstruct(sample_images)
                recon_loss_val = F.mse_loss(recon, sample_images)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Und CE: {und_loss_val.item():.4f} | "
                f"Gen MSE: {gen_loss_val.item():.6f} | "
                f"Recon MSE: {recon_loss_val.item():.6f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Save main weights
    ensure_weights_dir()
    torch.save(model.state_dict(), f"{WEIGHTS_DIR}/exp_cfg_torch.pt")
    print(f"Saved: {WEIGHTS_DIR}/exp_cfg_torch.pt")

    # Save EMA weights
    torch.save(ema_model.state_dict(), f"{WEIGHTS_DIR}/exp_cfg_ema_torch.pt")
    print(f"Saved: {WEIGHTS_DIR}/exp_cfg_ema_torch.pt")

    return model


if __name__ == "__main__":
    train()
