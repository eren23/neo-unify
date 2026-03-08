"""Exp Latent: Flow matching in VQ-VAE latent space with joint training."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

from shared.data import generate_dataset, CLASSES
from exp_latent.model import NeoUnifyLatentModel
from phase_a.vqvae import VQVAE
from shared.utils import (
    save_image_grid, plot_losses, plot_dual_losses,
    WEIGHTS_DIR, ensure_weights_dir,
)


def ema_update(ema_weights, model_weights, decay=0.999):
    new_ema = {}
    for key in ema_weights:
        new_ema[key] = decay * ema_weights[key] + (1 - decay) * model_weights[key]
    return new_ema


def load_vqvae():
    """Load frozen VQ-VAE for encoding/decoding."""
    vqvae = VQVAE(num_embeddings=256, embedding_dim=64, commitment_cost=0.25)
    weights = mx.load(f"{WEIGHTS_DIR}/vqvae.npz")
    vqvae.load_weights(list(weights.items()))
    print("Loaded frozen VQ-VAE from weights/vqvae.npz")
    return vqvae


def pre_encode_dataset(vqvae, images, batch_size=256):
    """Encode all images to continuous latent space (before quantization)."""
    all_z = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i : i + batch_size]
        z_e = vqvae.encoder(batch)  # (B, 4, 4, 64) continuous latent
        # MLX lazy eval
        mx.eval(z_e)  # noqa: S307
        all_z.append(z_e)
    z_all = mx.concatenate(all_z, axis=0)
    print(f"Pre-encoded dataset: {z_all.shape} (latent space)")
    return z_all


def train():
    print("=" * 60)
    print("Exp Latent: Flow Matching in VQ-VAE Latent Space")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic dataset...")
    images, labels = generate_dataset(samples_per_class=3000)
    print(f"Dataset: {images.shape} images, {labels.shape} labels")

    # Load frozen VQ-VAE and pre-encode
    vqvae = load_vqvae()
    z_all = pre_encode_dataset(vqvae, images)

    # Model
    model = NeoUnifyLatentModel(
        patch_size=4, image_size=16, channels=3, latent_dim=64,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    )
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")

    # Optimizer
    num_epochs = 200
    batch_size = 64
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size
    total_steps = num_epochs * steps_per_epoch

    warmup_steps = steps_per_epoch * 5
    lr_schedule = optim.schedulers.cosine_decay(3e-4, total_steps - warmup_steps)
    warmup = optim.schedulers.linear_schedule(1e-6, 3e-4, warmup_steps)
    schedule = optim.schedulers.join_schedules([warmup, lr_schedule], [warmup_steps])
    optimizer = optim.Adam(learning_rate=schedule)

    num_classes = 6
    label_dropout_rate = 0.1
    gen_weight = 3.0

    # Joint loss: understanding on pixels + generation on latents
    def joint_loss_fn(model, x_pixels, z_0, class_labels, gen_class_labels):
        B = x_pixels.shape[0]

        # Understanding loss: cross-entropy on raw pixels
        logits = model.forward_understand(x_pixels)
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        und_loss = -mx.mean(log_probs[mx.arange(B), class_labels])

        # Generation loss: flow matching MSE in latent space
        t = mx.random.uniform(shape=(B,))
        z_noise = mx.random.normal(z_0.shape)
        t_expanded = t.reshape(B, 1, 1, 1)
        z_t = (1 - t_expanded) * z_noise + t_expanded * z_0
        v_target = z_0 - z_noise
        v_pred = model.forward_generate(z_t, t, gen_class_labels)
        gen_loss = mx.mean((v_pred - v_target) ** 2)

        # Reconstruction loss: understand pathway -> per-patch decode -> MSE
        recon = model.forward_reconstruct(x_pixels)
        recon_loss = mx.mean((recon - x_pixels) ** 2)

        total_loss = und_loss + gen_weight * gen_loss + 1.0 * recon_loss
        return total_loss

    loss_and_grad = nn.value_and_grad(model, joint_loss_fn)

    # Initialize EMA
    ema_weights = dict(nn.utils.tree_flatten(model.parameters()))
    # MLX lazy eval
    mx.eval(model.parameters(), optimizer.state)  # noqa: S307

    t0 = time.time()

    for epoch in range(num_epochs):
        perm = mx.array(np.random.permutation(n_samples))

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch_images = images[idx]
            batch_z = z_all[idx]
            batch_labels = labels[idx]

            # Label dropout
            B = batch_labels.shape[0]
            dropout_mask = mx.random.uniform(shape=(B,)) < label_dropout_rate
            gen_labels = mx.where(dropout_mask, mx.full((B,), num_classes, dtype=mx.int32), batch_labels)

            total_loss, grads = loss_and_grad(model, batch_images, batch_z, batch_labels, gen_labels)
            optimizer.update(model, grads)
            # MLX lazy eval
            mx.eval(model.parameters(), optimizer.state)  # noqa: S307

            # EMA update
            model_weights = dict(nn.utils.tree_flatten(model.parameters()))
            ema_weights = ema_update(ema_weights, model_weights)
            # MLX lazy eval
            mx.eval(list(ema_weights.values()))  # noqa: S307

        if (epoch + 1) % 30 == 0 or epoch == 0:
            # Compute individual losses for logging
            sample_idx = perm[:256]
            sample_images = images[sample_idx]
            sample_z = z_all[sample_idx]
            sample_labels = labels[sample_idx]

            logits = model.forward_understand(sample_images)
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            und_loss_val = -mx.mean(log_probs[mx.arange(256), sample_labels])

            t_sample = mx.random.uniform(shape=(256,))
            z_noise = mx.random.normal(sample_z.shape)
            t_exp = t_sample.reshape(256, 1, 1, 1)
            z_t = (1 - t_exp) * z_noise + t_exp * sample_z
            v_target = sample_z - z_noise
            v_pred = model.forward_generate(z_t, t_sample, sample_labels)
            gen_loss_val = mx.mean((v_pred - v_target) ** 2)

            recon = model.forward_reconstruct(sample_images)
            recon_loss_val = mx.mean((recon - sample_images) ** 2)
            # MLX lazy computation - mx.eval triggers MLX array evaluation, not Python's eval
            mx.eval(und_loss_val, gen_loss_val, recon_loss_val)  # noqa: S307

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Und CE: {und_loss_val.item():.4f} | "
                f"Gen MSE (latent): {gen_loss_val.item():.4f} | "
                f"Recon MSE: {recon_loss_val.item():.6f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Save weights
    ensure_weights_dir()
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(f"{WEIGHTS_DIR}/exp_latent.npz", **weights)
    print(f"Saved: {WEIGHTS_DIR}/exp_latent.npz")

    mx.savez(f"{WEIGHTS_DIR}/exp_latent_ema.npz", **ema_weights)
    print(f"Saved: {WEIGHTS_DIR}/exp_latent_ema.npz")

    return model


if __name__ == "__main__":
    train()
