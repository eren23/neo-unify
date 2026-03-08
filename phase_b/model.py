"""Flow matching transformer for continuous image generation (Neo-Unify style).

Patchifies 16x16 images into 4x4 patches (16 patches, 48-dim each).
Bidirectional transformer with time + class conditioning.
Predicts velocity field for flow matching ODE.
"""

import mlx.core as mx
import mlx.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for flow matching timestep t in [0, 1]."""

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
    """Full (non-causal) self-attention."""

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


class AdaLNBlock(nn.Module):
    """Transformer block with Adaptive Layer Norm (AdaLN) conditioning."""

    def __init__(self, hidden_dim, num_heads, cond_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim, affine=False)
        self.attn = BidirectionalAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim, affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        # AdaLN: project condition to 6 modulation params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )

    def __call__(self, x, cond):
        mod = self.adaLN_modulation(cond)
        shift1, scale1, gate1, shift2, scale2, gate2 = mx.split(
            mod.reshape(x.shape[0], 6, -1).transpose(1, 0, 2), 6, axis=0
        )
        shift1 = shift1.squeeze(0)
        scale1 = scale1.squeeze(0)
        gate1 = gate1.squeeze(0)
        shift2 = shift2.squeeze(0)
        scale2 = scale2.squeeze(0)
        gate2 = gate2.squeeze(0)

        h = self.ln1(x)
        h = h * (1 + scale1[:, None, :]) + shift1[:, None, :]
        x = x + gate1[:, None, :] * self.attn(h)

        h = self.ln2(x)
        h = h * (1 + scale2[:, None, :]) + shift2[:, None, :]
        x = x + gate2[:, None, :] * self.mlp(h)

        return x


class FlowMatchingTransformer(nn.Module):
    """Patch-based flow matching model for 16x16 image generation."""

    def __init__(
        self,
        patch_size=4,
        image_size=16,
        channels=3,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2

        patch_dim = patch_size * patch_size * channels

        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.num_patches, hidden_dim)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        self.blocks = [
            AdaLNBlock(hidden_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ]

        self.final_ln = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, patch_dim)

    def patchify(self, x):
        B, H, W, C = x.shape
        p = self.patch_size
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        patches = x.reshape(B, self.num_patches, -1)
        return patches

    def unpatchify(self, patches):
        B = patches.shape[0]
        p = self.patch_size
        h = w = self.image_size // p

        patches = patches.reshape(B, h, w, p, p, self.channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        x = patches.reshape(B, self.image_size, self.image_size, self.channels)
        return x

    def __call__(self, x_t, t, class_labels):
        patches = self.patchify(x_t)
        x = self.patch_proj(patches)
        positions = mx.arange(self.num_patches)
        x = x + self.pos_emb(positions)

        t_emb = self.time_emb(t)
        c_emb = self.class_emb(class_labels)
        cond = t_emb + c_emb

        for block in self.blocks:
            x = block(x, cond)

        x = self.final_ln(x)
        v_patches = self.output_proj(x)
        v = self.unpatchify(v_patches)
        return v


if __name__ == "__main__":
    model = FlowMatchingTransformer()
    x = mx.random.normal((4, 16, 16, 3))
    t = mx.random.uniform(shape=(4,))
    labels = mx.array([0, 1, 2, 3])
    v = model(x, t, labels)
    mx.eval(v)
    print(f"Input:    {x.shape}")
    print(f"Velocity: {v.shape}")
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")
