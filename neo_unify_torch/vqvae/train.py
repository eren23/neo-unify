"""Train VQ-VAE tokenizer on synthetic 16x16 images (PyTorch)."""

import torch
import torch.nn.functional as F
import numpy as np
import time

from neo_unify_torch.shared.data import generate_dataset, CLASSES
from neo_unify_torch.vqvae.model import VQVAE
from neo_unify_torch.shared.utils import (
    save_image_grid, plot_losses, WEIGHTS_DIR, ensure_weights_dir, get_device,
)


def train():
    print("=" * 60)
    print("VQ-VAE Training (PyTorch)")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Generate data
    print("\nGenerating synthetic dataset...")
    images, labels = generate_dataset(samples_per_class=1000)
    images = images.to(device)
    labels = labels.to(device)
    print(f"Dataset: {images.shape} images, {labels.shape} labels")

    # Save sample real images
    save_image_grid(images[:64], "real_samples_torch.png", title="Real Training Samples")

    # Model
    model = VQVAE(num_embeddings=256, embedding_dim=64, commitment_cost=0.25).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VQ-VAE parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training
    batch_size = 64
    num_epochs = 80
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size

    losses = []
    t0 = time.time()

    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_vq = 0.0

        model.train()
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch = images[idx]

            x_recon, indices, vq_loss = model(batch)
            recon_loss = F.mse_loss(x_recon, batch)
            total_loss = recon_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_vq += vq_loss.item()
            losses.append(total_loss.item())

        avg_loss = epoch_loss / steps_per_epoch
        avg_recon = epoch_recon / steps_per_epoch
        avg_vq = epoch_vq / steps_per_epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                all_indices = model.encode(images[:512])
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
    plot_losses(losses, "vqvae_loss_torch.png", title="VQ-VAE Training Loss (PyTorch)")

    # Reconstruction quality
    with torch.no_grad():
        test_images = images[:64]
        recon, _, _ = model(test_images)
    save_image_grid(recon, "vqvae_reconstructions_torch.png", title="VQ-VAE Reconstructions (PyTorch)")
    save_image_grid(test_images, "vqvae_originals_torch.png", title="Originals (for comparison)")

    # Save model weights
    ensure_weights_dir()
    torch.save(model.state_dict(), f"{WEIGHTS_DIR}/vqvae_torch.pt")
    print(f"Saved: {WEIGHTS_DIR}/vqvae_torch.pt")

    # Final codebook stats
    with torch.no_grad():
        all_indices = model.encode(images)
    used, total = model.codebook_utilization(all_indices)
    print(f"Final codebook utilization: {used}/{total} ({100*used/total:.1f}%)")

    return model


if __name__ == "__main__":
    train()
