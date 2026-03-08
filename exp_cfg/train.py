"""Exp CFG: Joint gradient training with label dropout, EMA, and longer training."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import copy

from shared.data import generate_dataset, CLASSES
from exp_cfg.model import NeoUnifyModel
from shared.utils import (
    save_image_grid, plot_losses, plot_dual_losses,
    WEIGHTS_DIR, ensure_weights_dir,
)


def ema_update(ema_weights, model_weights, decay=0.999):
    """Update EMA weights in-place."""
    new_ema = {}
    for key in ema_weights:
        new_ema[key] = decay * ema_weights[key] + (1 - decay) * model_weights[key]
    return new_ema


def train():
    print("=" * 60)
    print("Exp CFG: Joint Gradient + CFG + EMA + 300 Epochs")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic dataset...")
    images, labels = generate_dataset(samples_per_class=3000)
    print(f"Dataset: {images.shape} images, {labels.shape} labels")

    # Model
    model = NeoUnifyModel(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    )
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")

    # Optimizer with warmup + cosine decay
    num_epochs = 200
    batch_size = 64
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size
    total_steps = num_epochs * steps_per_epoch  # single update per step

    warmup_steps = steps_per_epoch * 5
    lr_schedule = optim.schedulers.cosine_decay(3e-4, total_steps - warmup_steps)
    warmup = optim.schedulers.linear_schedule(1e-6, 3e-4, warmup_steps)
    schedule = optim.schedulers.join_schedules([warmup, lr_schedule], [warmup_steps])
    optimizer = optim.Adam(learning_rate=schedule)

    num_classes = 6
    label_dropout_rate = 0.1
    gen_weight = 3.0

    # Joint loss function
    def joint_loss_fn(model, x, class_labels, gen_class_labels):
        # Understanding loss: cross-entropy
        logits = model.forward_understand(x)
        B = logits.shape[0]
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        und_loss = -mx.mean(log_probs[mx.arange(B), class_labels])

        # Generation loss: flow matching MSE (with label-dropped class labels)
        t = mx.random.uniform(shape=(B,))
        noise = mx.random.normal(x.shape)
        t_expanded = t.reshape(B, 1, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * x
        v_target = x - noise
        v_pred = model.forward_generate(x_t, t, gen_class_labels)
        gen_loss = mx.mean((v_pred - v_target) ** 2)

        # Reconstruction loss: understand pathway -> per-patch decode -> MSE
        recon = model.forward_reconstruct(x)
        recon_loss = mx.mean((recon - x) ** 2)

        total_loss = und_loss + gen_weight * gen_loss + 1.0 * recon_loss
        return total_loss

    loss_and_grad = nn.value_and_grad(model, joint_loss_fn)

    # Initialize EMA weights
    ema_weights = dict(nn.utils.tree_flatten(model.parameters()))
    # MLX lazy computation trigger
    mx.eval(model.parameters(), optimizer.state)  # noqa: S307

    # Training
    und_losses = []
    gen_losses = []
    t0 = time.time()

    for epoch in range(num_epochs):
        perm = mx.array(np.random.permutation(n_samples))
        epoch_und_loss = 0.0
        epoch_gen_loss = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch_images = images[idx]
            batch_labels = labels[idx]

            # Label dropout for generation: 10% -> null class (index num_classes)
            B = batch_labels.shape[0]
            dropout_mask = mx.random.uniform(shape=(B,)) < label_dropout_rate
            gen_labels = mx.where(dropout_mask, mx.full((B,), num_classes, dtype=mx.int32), batch_labels)

            # Joint forward + backward
            total_loss, grads = loss_and_grad(model, batch_images, batch_labels, gen_labels)
            optimizer.update(model, grads)
            # MLX lazy computation trigger
            mx.eval(model.parameters(), optimizer.state)  # noqa: S307

            # Update EMA
            model_weights = dict(nn.utils.tree_flatten(model.parameters()))
            ema_weights = ema_update(ema_weights, model_weights)
            # MLX lazy computation trigger
            mx.eval(list(ema_weights.values()))  # noqa: S307

        if (epoch + 1) % 30 == 0 or epoch == 0:
            # Compute actual individual losses for logging
            sample_idx = perm[:256]
            sample_images = images[sample_idx]
            sample_labels = labels[sample_idx]

            logits = model.forward_understand(sample_images)
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            und_loss_val = -mx.mean(log_probs[mx.arange(256), sample_labels])

            t_sample = mx.random.uniform(shape=(256,))
            noise = mx.random.normal(sample_images.shape)
            t_exp = t_sample.reshape(256, 1, 1, 1)
            x_t = (1 - t_exp) * noise + t_exp * sample_images
            v_target = sample_images - noise
            v_pred = model.forward_generate(x_t, t_sample, sample_labels)
            gen_loss_val = mx.mean((v_pred - v_target) ** 2)

            recon = model.forward_reconstruct(sample_images)
            recon_loss_val = mx.mean((recon - sample_images) ** 2)
            # MLX lazy computation trigger - evaluates MLX arrays, not Python eval
            mx.eval(und_loss_val, gen_loss_val, recon_loss_val)  # noqa: S307

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
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(f"{WEIGHTS_DIR}/exp_cfg.npz", **weights)
    print(f"Saved: {WEIGHTS_DIR}/exp_cfg.npz")

    # Save EMA weights
    mx.savez(f"{WEIGHTS_DIR}/exp_cfg_ema.npz", **ema_weights)
    print(f"Saved: {WEIGHTS_DIR}/exp_cfg_ema.npz")

    return model


if __name__ == "__main__":
    train()
