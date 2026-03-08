"""VQ-VAE visual tokenizer for 16x16 images in PyTorch.

Encoder: Conv layers downsample to 4x4 latent grid (16 tokens)
Vector Quantizer: 256-entry codebook, 64-dim embeddings
Decoder: ConvTranspose layers upsample back to 16x16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """(B,3,16,16) -> (B,64,4,4) latent grid."""

    def __init__(self, latent_dim=64):
        super().__init__()
        # 16x16 -> 8x8
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        # 8x8 -> 4x4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        # 4x4 -> 4x4
        self.conv3 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: (B, 3, 16, 16) NCHW
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.conv3(x)  # (B, latent_dim, 4, 4)
        return x


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through estimator."""

    def __init__(self, num_embeddings=256, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.codebook.weight, std=0.1)

    def forward(self, z_e):
        """Quantize encoder output.

        Args:
            z_e: (B, D, H, W) encoder output (NCHW)

        Returns:
            z_q: (B, D, H, W) quantized output (with straight-through gradient)
            indices: (B, H, W) codebook indices
            vq_loss: commitment + codebook loss
        """
        # Permute to (B, H, W, D) for distance computation
        z_e_perm = z_e.permute(0, 2, 3, 1)  # (B, H, W, D)
        B, H, W, D = z_e_perm.shape
        flat = z_e_perm.reshape(-1, D)  # (B*H*W, D)

        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e^T
        z_sq = (flat ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
        e_sq = (self.codebook.weight ** 2).sum(dim=-1, keepdim=True).T  # (1, K)
        distances = z_sq + e_sq - 2 * flat @ self.codebook.weight.T  # (N, K)

        indices = distances.argmin(dim=-1)  # (N,)
        z_q_flat = self.codebook(indices)  # (N, D)
        z_q_perm = z_q_flat.reshape(B, H, W, D)
        indices = indices.reshape(B, H, W)

        # Losses
        codebook_loss = F.mse_loss(z_e_perm.detach(), z_q_perm)
        commitment_loss = F.mse_loss(z_e_perm, z_q_perm.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_perm = z_e_perm + (z_q_perm - z_e_perm).detach()

        # Permute back to (B, D, H, W)
        z_q = z_q_perm.permute(0, 3, 1, 2)

        return z_q, indices, vq_loss

    def decode_indices(self, indices):
        """Map indices back to embeddings. Returns (B, D, H, W)."""
        B, H, W = indices.shape
        flat = indices.reshape(-1)
        z_q = self.codebook(flat)  # (B*H*W, D)
        z_q = z_q.reshape(B, H, W, self.embedding_dim)
        return z_q.permute(0, 3, 1, 2)  # (B, D, H, W)


class Decoder(nn.Module):
    """(B,64,4,4) -> (B,3,16,16)."""

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

    def forward(self, z_q):
        # z_q: (B, latent_dim, 4, 4) NCHW
        x = F.relu(self.gn1(self.conv1(z_q)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))  # Output in [0, 1]
        return x


class VQVAE(nn.Module):
    """Complete VQ-VAE model."""

    def __init__(self, num_embeddings=256, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(latent_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=embedding_dim)

    def forward(self, x):
        """Forward pass.

        Returns:
            x_recon: reconstructed images (B, 3, 16, 16)
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
        flat = indices.detach().cpu().numpy().reshape(-1)
        unique = np.unique(flat)
        return len(unique), self.quantizer.num_embeddings


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VQVAE().to(device)
    x = torch.rand(4, 3, 16, 16, device=device)
    x_recon, indices, vq_loss = model(x)
    print(f"Input:  {x.shape}")
    print(f"Recon:  {x_recon.shape}")
    print(f"Indices: {indices.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
