"""VQ-VAE visual tokenizer for 16x16 images in MLX.

Encoder: Conv layers downsample to 4x4 latent grid (16 tokens)
Vector Quantizer: 256-entry codebook, 64-dim embeddings
Decoder: ConvTranspose layers upsample back to 16x16
"""

import mlx.core as mx
import mlx.nn as nn


class Encoder(nn.Module):
    """16x16x3 -> 4x4x64 latent grid."""

    def __init__(self, latent_dim=64):
        super().__init__()
        # 16x16x3 -> 8x8x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        # 8x8x32 -> 4x4x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        # 4x4x64 -> 4x4x latent_dim (project to codebook dim)
        self.conv3 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1)

    def __call__(self, x):
        # x: (B, 16, 16, 3) - MLX uses NHWC
        x = nn.relu(self.gn1(self.conv1(x)))
        x = nn.relu(self.gn2(self.conv2(x)))
        x = self.conv3(x)  # (B, 4, 4, latent_dim)
        return x


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through estimator."""

    def __init__(self, num_embeddings=256, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: (num_embeddings, embedding_dim)
        self.codebook = mx.random.normal((num_embeddings, embedding_dim)) * 0.1

    def __call__(self, z_e):
        """Quantize encoder output.

        Args:
            z_e: (B, H, W, D) encoder output

        Returns:
            z_q: (B, H, W, D) quantized output (with straight-through gradient)
            indices: (B, H, W) codebook indices
            vq_loss: commitment + codebook loss
        """
        B, H, W, D = z_e.shape
        flat = z_e.reshape(-1, D)  # (B*H*W, D)

        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e^T
        z_sq = (flat * flat).sum(axis=-1, keepdims=True)  # (N, 1)
        e_sq = (self.codebook * self.codebook).sum(axis=-1, keepdims=True).T  # (1, K)
        distances = z_sq + e_sq - 2 * flat @ self.codebook.T  # (N, K)

        indices = mx.argmin(distances, axis=-1)  # (N,)
        z_q = self.codebook[indices]  # (N, D)
        z_q = z_q.reshape(B, H, W, D)
        indices = indices.reshape(B, H, W)

        # Losses
        codebook_loss = mx.mean((mx.stop_gradient(z_e) - z_q) ** 2)
        commitment_loss = mx.mean((z_e - mx.stop_gradient(z_q)) ** 2)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: gradients pass through as if z_q = z_e
        z_q = z_e + mx.stop_gradient(z_q - z_e)

        return z_q, indices, vq_loss

    def decode_indices(self, indices):
        """Map indices back to embeddings."""
        shape = indices.shape
        flat = indices.reshape(-1)
        z_q = self.codebook[flat]
        return z_q.reshape(*shape, self.embedding_dim)


class Decoder(nn.Module):
    """4x4x64 -> 16x16x3."""

    def __init__(self, latent_dim=64):
        super().__init__()
        # 4x4 -> 8x8
        self.conv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)
        # 8x8 -> 16x16
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        # Final projection
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def __call__(self, z_q):
        # z_q: (B, 4, 4, latent_dim) in NHWC
        x = nn.relu(self.gn1(self.conv1(z_q)))
        x = nn.relu(self.gn2(self.conv2(x)))
        x = mx.sigmoid(self.conv3(x))  # Output in [0, 1]
        return x


class VQVAE(nn.Module):
    """Complete VQ-VAE model."""

    def __init__(self, num_embeddings=256, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(latent_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=embedding_dim)

    def __call__(self, x):
        """Forward pass.

        Returns:
            x_recon: reconstructed images
            indices: codebook indices (B, 4, 4)
            vq_loss: quantization loss
        """
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, indices, vq_loss

    def encode(self, x):
        """Encode images to codebook indices."""
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer(z_e)
        return indices

    def decode(self, indices):
        """Decode codebook indices to images."""
        z_q = self.quantizer.decode_indices(indices)
        return self.decoder(z_q)

    def codebook_utilization(self, indices):
        """Check how many codebook entries are being used."""
        import numpy as np
        flat = np.array(indices.reshape(-1))
        unique = np.unique(flat)
        return len(unique), self.quantizer.num_embeddings


if __name__ == "__main__":
    model = VQVAE()
    x = mx.random.uniform(shape=(4, 16, 16, 3))
    x_recon, indices, vq_loss = model(x)
    mx.eval(x_recon, indices, vq_loss)  # MLX lazy evaluation trigger
    print(f"Input:  {x.shape}")
    print(f"Recon:  {x_recon.shape}")
    print(f"Indices: {indices.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")
