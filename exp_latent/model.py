"""Exp Latent: Neo-Unify with generation in VQ-VAE latent space (4x4x64)."""

import mlx.core as mx
import mlx.nn as nn
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

    def __call__(self, t):
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        half_dim = self.dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half_dim) / half_dim)
        args = t * freqs
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        return self.mlp(emb)


class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
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
        self.ln_gen = nn.LayerNorm(hidden_dim, affine=False)
        self.ffn_gen = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.gen_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )

    def __call__(self, x, task, cond=None):
        x = x + self.attn(self.ln_attn(x))
        if task == "understand":
            x = x + self.ffn_und(self.ln_und(x))
        else:
            mod = self.gen_modulation(cond)
            shift, scale, gate = mx.split(mod, 3, axis=-1)
            h = self.ln_gen(x)
            h = h * (1 + scale[:, None, :]) + shift[:, None, :]
            x = x + gate[:, None, :] * self.ffn_gen(h)
        return x


class NeoUnifyLatentModel(nn.Module):
    """Neo-Unify with latent-space generation.

    Understanding: raw pixels -> patches -> MoT -> class logits (same as baseline)
    Generation: 4x4x64 latent -> 16 patches of dim 64 -> MoT -> velocity in latent space
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
        self.blocks = [
            MoTBlock(hidden_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ]

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
        """Patchify pixel images: (B, 16, 16, 3) -> (B, 16, 48)."""
        B, H, W, C = x.shape
        p = self.patch_size
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, self.num_patches, -1)

    def patchify_latent(self, z):
        """Patchify latent: (B, 4, 4, 64) -> (B, 16, 64). Each 1x1 position is a patch."""
        B = z.shape[0]
        return z.reshape(B, self.num_patches, self.latent_dim)

    def unpatchify_latent(self, patches):
        """Unpatchify latent: (B, 16, 64) -> (B, 4, 4, 64)."""
        B = patches.shape[0]
        return patches.reshape(B, 4, 4, self.latent_dim)

    def unpatchify_pixels(self, patches):
        """Unpatchify pixel patches: (B, 16, 48) -> (B, 16, 16, 3)."""
        B = patches.shape[0]
        p = self.patch_size
        h = w = self.image_size // p
        patches = patches.reshape(B, h, w, p, p, self.channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        return patches.reshape(B, self.image_size, self.image_size, self.channels)

    def forward_understand(self, images):
        """Understanding: pixel images -> class logits."""
        patches = self.patchify_pixels(images)
        x = self.und_patch_proj(patches)
        x = x + self.und_pos_emb(mx.arange(self.num_patches))
        for block in self.blocks:
            x = block(x, task="understand")
        x = mx.mean(x, axis=1)
        x = self.und_ln(x)
        return self.und_head(x)

    def forward_reconstruct(self, images):
        """Reconstruction: pixel image -> understand pathway -> per-patch decode -> image."""
        patches = self.patchify_pixels(images)
        x = self.und_patch_proj(patches)
        x = x + self.und_pos_emb(mx.arange(self.num_patches))
        for block in self.blocks:
            x = block(x, task="understand")
        x = self.recon_ln(x)
        x_patches = self.recon_head(x)
        return mx.sigmoid(self.unpatchify_pixels(x_patches))

    def forward_generate(self, z_t, t, class_labels):
        """Generation: noisy latent + time + class -> velocity in latent space.

        Args:
            z_t: (B, 4, 4, 64) noisy latent
            t: (B,) timestep
            class_labels: (B,) class indices (num_classes = null class for CFG)

        Returns:
            v: (B, 4, 4, 64) predicted velocity
        """
        patches = self.patchify_latent(z_t)
        x = self.gen_patch_proj(patches)
        x = x + self.gen_pos_emb(mx.arange(self.num_patches))

        t_emb = self.time_emb(t)
        c_emb = self.class_emb(class_labels)
        cond = t_emb + c_emb

        for block in self.blocks:
            x = block(x, task="generate", cond=cond)

        x = self.gen_ln(x)
        v_patches = self.gen_head(x)
        return self.unpatchify_latent(v_patches)
