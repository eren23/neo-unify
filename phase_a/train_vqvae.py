"""Stage 1: Train VQ-VAE tokenizer on synthetic 16x16 images."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

from shared.data import generate_dataset, CLASSES
from phase_a.vqvae import VQVAE
from shared.utils import save_image_grid, plot_losses, WEIGHTS_DIR, ensure_weights_dir


def train():
    print("=" * 60)
    print("Phase A - Stage 1: Training VQ-VAE Tokenizer")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic dataset...")
    images, labels = generate_dataset(samples_per_class=1000)
    print(f"Dataset: {images.shape} images, {labels.shape} labels")

    # Save sample real images
    save_image_grid(images[:64], "real_samples.png", title="Real Training Samples")

    # Model
    model = VQVAE(num_embeddings=256, embedding_dim=64, commitment_cost=0.25)
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"VQ-VAE parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=3e-4)

    # Loss function
    def loss_fn(model, x):
        x_recon, indices, vq_loss = model(x)
        recon_loss = mx.mean((x - x_recon) ** 2)
        total_loss = recon_loss + vq_loss
        return total_loss, (recon_loss, vq_loss, indices)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    batch_size = 64
    num_epochs = 80
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size

    losses = []
    t0 = time.time()

    for epoch in range(num_epochs):
        # Shuffle
        perm = mx.array(np.random.permutation(n_samples))
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_vq = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch = images[idx]

            (total_loss, (recon_loss, vq_loss, indices)), grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            # Force computation (MLX is lazy)
            mx.eval(model.parameters(), optimizer.state)  # noqa: S307

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_vq += vq_loss.item()
            losses.append(total_loss.item())

        avg_loss = epoch_loss / steps_per_epoch
        avg_recon = epoch_recon / steps_per_epoch
        avg_vq = epoch_vq / steps_per_epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            all_indices = model.encode(images[:512])
            mx.eval(all_indices)  # noqa: S307
            used, total = model.codebook_utilization(all_indices)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} (recon: {avg_recon:.4f}, vq: {avg_vq:.4f}) | "
                f"Codebook: {used}/{total} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Plot losses
    plot_losses(losses, "vqvae_loss.png", title="VQ-VAE Training Loss")

    # Reconstruction quality
    test_images = images[:64]
    recon, _, _ = model(test_images)
    mx.eval(recon)  # noqa: S307
    save_image_grid(recon, "vqvae_reconstructions.png", title="VQ-VAE Reconstructions")
    save_image_grid(test_images, "vqvae_originals.png", title="Originals (for comparison)")

    # Save model weights
    ensure_weights_dir()
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(f"{WEIGHTS_DIR}/vqvae.npz", **weights)
    print(f"Saved: {WEIGHTS_DIR}/vqvae.npz")

    # Encode full dataset for transformer training
    print("\nEncoding full dataset...")
    all_indices = []
    for i in range(0, n_samples, 256):
        batch = images[i : min(i + 256, n_samples)]
        idx = model.encode(batch)
        mx.eval(idx)  # noqa: S307
        all_indices.append(idx)

    all_indices = mx.concatenate(all_indices, axis=0)
    all_indices_flat = all_indices.reshape(n_samples, -1)
    mx.eval(all_indices_flat)  # noqa: S307

    np.savez(
        f"{WEIGHTS_DIR}/encoded_dataset.npz",
        indices=np.array(all_indices_flat),
        labels=np.array(labels),
    )
    print(f"Saved: {WEIGHTS_DIR}/encoded_dataset.npz ({all_indices_flat.shape})")

    # Final codebook stats
    used, total = model.codebook_utilization(all_indices)
    print(f"Final codebook utilization: {used}/{total} ({100*used/total:.1f}%)")

    return model


if __name__ == "__main__":
    train()
