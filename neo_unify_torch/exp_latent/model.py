"""Exp Latent: Neo-Unify with generation in VQ-VAE latent space (B,64,4,4) (PyTorch)."""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class MoTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, cond_dim):
        super().__init__()
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.attn = BidirectionalAttention(hidden_dim, num_heads)
        self.ln_und = nn.LayerNorm(hidden_dim)
        self.ffn_und = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ln_gen = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn_gen = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.gen_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )

    def forward(self, x, task, cond=None):
        x = x + self.attn(self.ln_attn(x))
        if task == "understand":
            x = x + self.ffn_und(self.ln_und(x))
        else:
            mod = self.gen_modulation(cond)
            shift, scale, gate = mod.chunk(3, dim=-1)
            h = self.ln_gen(x)
            h = h * (1 + scale[:, None, :]) + shift[:, None, :]
            x = x + gate[:, None, :] * self.ffn_gen(h)
        return x


class NeoUnifyLatentModel(nn.Module):
    """Neo-Unify with latent-space generation.

    Understanding: raw pixels (B,3,16,16) -> patches -> MoT -> class logits
    Generation: latent (B,64,4,4) -> 16 patches of dim 64 -> MoT -> velocity in latent space
    """

    def __init__(
        self,
        patch_size=4,
        image_size=16,
        channels=3,
        latent_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=6,
        num_classes=6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2  # 16
        self.num_classes = num_classes

        pixel_patch_dim = patch_size * patch_size * channels  # 48

        # Understanding pathway: pixel patches
        self.und_patch_proj = nn.Linear(pixel_patch_dim, hidden_dim)
        self.und_pos_emb = nn.Embedding(self.num_patches, hidden_dim)

        # Generation pathway: latent patches (each 1x1x64 = 64-dim)
        self.gen_patch_proj = nn.Linear(latent_dim, hidden_dim)
        self.gen_pos_emb = nn.Embedding(self.num_patches, hidden_dim)

        # Generation conditioning
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        # +1 for null class (CFG)
        self.class_emb = nn.Embedding(num_classes + 1, hidden_dim)

        # MoT backbone (shared)
        self.blocks = nn.ModuleList([
            MoTBlock(hidden_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

        # Understanding head
        self.und_ln = nn.LayerNorm(hidden_dim)
        self.und_head = nn.Linear(hidden_dim, num_classes)

        # Generation head: project back to latent patch dim
        self.gen_ln = nn.LayerNorm(hidden_dim)
        self.gen_head = nn.Linear(hidden_dim, latent_dim)

        # Reconstruction head (reuses understand pathway)
        self.recon_ln = nn.LayerNorm(hidden_dim)
        self.recon_head = nn.Linear(hidden_dim, pixel_patch_dim)

    def patchify_pixels(self, x):
        """Patchify pixel images: (B, C, H, W) -> (B, 16, 48)."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H//p, W//p, p, p, C)
        return x.reshape(B, self.num_patches, -1)

    def patchify_latent(self, z):
        """Patchify latent: (B, D, 4, 4) -> (B, 16, D). Permute NCHW to sequence."""
        B = z.shape[0]
        # (B, D, H, W) -> (B, H, W, D) -> (B, 16, D)
        return z.permute(0, 2, 3, 1).reshape(B, self.num_patches, self.latent_dim)

    def unpatchify_latent(self, patches):
        """Unpatchify latent: (B, 16, D) -> (B, D, 4, 4)."""
        B = patches.shape[0]
        # (B, 16, D) -> (B, 4, 4, D) -> (B, D, 4, 4)
        return patches.reshape(B, 4, 4, self.latent_dim).permute(0, 3, 1, 2)

    def unpatchify_pixels(self, patches):
        """Unpatchify pixel patches: (B, 16, 48) -> (B, C, H, W)."""
        B = patches.shape[0]
        p = self.patch_size
        h = w = self.image_size // p
        patches = patches.reshape(B, h, w, p, p, self.channels)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        return patches.reshape(B, self.channels, self.image_size, self.image_size)

    def forward_understand(self, images):
        """Understanding: pixel images (B,3,16,16) -> class logits."""
        patches = self.patchify_pixels(images)
        x = self.und_patch_proj(patches)
        x = x + self.und_pos_emb(torch.arange(self.num_patches, device=images.device))
        for block in self.blocks:
            x = block(x, task="understand")
        x = x.mean(dim=1)
        x = self.und_ln(x)
        return self.und_head(x)

    def forward_reconstruct(self, images):
        """Reconstruction: pixel image -> understand pathway -> per-patch decode -> image."""
        patches = self.patchify_pixels(images)
        x = self.und_patch_proj(patches)
        x = x + self.und_pos_emb(torch.arange(self.num_patches, device=images.device))
        for block in self.blocks:
            x = block(x, task="understand")
        x = self.recon_ln(x)
        x_patches = self.recon_head(x)
        return torch.sigmoid(self.unpatchify_pixels(x_patches))

    def forward_generate(self, z_t, t, class_labels):
        """Generation: noisy latent + time + class -> velocity in latent space.

        Args:
            z_t: (B, 64, 4, 4) noisy latent (NCHW)
            t: (B,) timestep
            class_labels: (B,) class indices (num_classes = null class for CFG)

        Returns:
            v: (B, 64, 4, 4) predicted velocity (NCHW)
        """
        patches = self.patchify_latent(z_t)
        x = self.gen_patch_proj(patches)
        x = x + self.gen_pos_emb(torch.arange(self.num_patches, device=z_t.device))

        t_emb = self.time_emb(t)
        c_emb = self.class_emb(class_labels)
        cond = t_emb + c_emb

        for block in self.blocks:
            x = block(x, task="generate", cond=cond)

        x = self.gen_ln(x)
        v_patches = self.gen_head(x)
        return self.unpatchify_latent(v_patches)
