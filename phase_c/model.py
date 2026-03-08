"""Mini Neo-Unify: Unified model with Mixture-of-Transformer (MoT) backbone.

ONE model that does BOTH understanding (image -> class) AND generation (class -> image).
- Encoder-free: raw pixels, patchified (like Phase B)
- MoT backbone: shared self-attention + task-specific expert FFNs
- Hybrid loss: cross-entropy (understanding) + flow matching MSE (generation)
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


class MoTBlock(nn.Module):
    """Mixture-of-Transformer block: shared attention + task-specific expert FFNs.

    - Shared: LayerNorm + BidirectionalAttention (unmodulated, task-agnostic)
    - Understanding FFN: standard pre-norm MLP (no conditioning)
    - Generation FFN: AdaLN-modulated MLP (conditioned on time + class)
    - Routing: hard task flag (no learned gating needed)
    """

    def __init__(self, hidden_dim, num_heads, cond_dim):
        super().__init__()
        # Shared attention
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.attn = BidirectionalAttention(hidden_dim, num_heads)

        # Understanding expert FFN (standard pre-norm)
        self.ln_und = nn.LayerNorm(hidden_dim)
        self.ffn_und = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        # Generation expert FFN (AdaLN-modulated)
        self.ln_gen = nn.LayerNorm(hidden_dim, affine=False)
        self.ffn_gen = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        # AdaLN modulation for generation FFN: shift, scale, gate (3 * hidden_dim)
        self.gen_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )

    def __call__(self, x, task, cond=None):
        """
        Args:
            x: (B, T, D) patch sequence
            task: "understand" or "generate"
            cond: (B, D) conditioning vector (only used for generation)
        """
        # Shared attention (unmodulated)
        x = x + self.attn(self.ln_attn(x))

        # Task-specific FFN
        if task == "understand":
            x = x + self.ffn_und(self.ln_und(x))
        else:
            # AdaLN-modulated generation FFN
            mod = self.gen_modulation(cond)  # (B, 3*D)
            shift, scale, gate = mx.split(mod, 3, axis=-1)
            h = self.ln_gen(x)
            h = h * (1 + scale[:, None, :]) + shift[:, None, :]
            x = x + gate[:, None, :] * self.ffn_gen(h)

        return x


class NeoUnifyModel(nn.Module):
    """Mini Neo-Unify: unified understanding + generation model.

    Architecture:
        Input: 16x16 image -> patchify -> 16 patches -> Linear -> + pos_emb
        Backbone: 6x MoTBlock (shared attn + task-specific expert FFNs)
        Understanding head: mean_pool -> LN -> Linear(hidden, 6) -> class logits
        Generation head: LN -> Linear(hidden, 48) -> unpatchify -> velocity
    """

    def __init__(
        self,
        patch_size=4,
        image_size=16,
        channels=3,
        hidden_dim=128,
        num_heads=4,
        num_layers=6,
        num_classes=6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2  # 16
        self.num_classes = num_classes

        patch_dim = patch_size * patch_size * channels  # 48

        # Shared input projection
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.num_patches, hidden_dim)

        # Generation conditioning
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        # MoT backbone
        self.blocks = [
            MoTBlock(hidden_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ]

        # Understanding head: mean pool -> class logits
        self.und_ln = nn.LayerNorm(hidden_dim)
        self.und_head = nn.Linear(hidden_dim, num_classes)

        # Generation head: project back to patch space
        self.gen_ln = nn.LayerNorm(hidden_dim)
        self.gen_head = nn.Linear(hidden_dim, patch_dim)

    def patchify(self, x):
        B, H, W, C = x.shape
        p = self.patch_size
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, self.num_patches, -1)

    def unpatchify(self, patches):
        B = patches.shape[0]
        p = self.patch_size
        h = w = self.image_size // p
        patches = patches.reshape(B, h, w, p, p, self.channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        return patches.reshape(B, self.image_size, self.image_size, self.channels)

    def _encode_patches(self, images):
        """Shared patch encoding for both tasks."""
        patches = self.patchify(images)
        x = self.patch_proj(patches)
        x = x + self.pos_emb(mx.arange(self.num_patches))
        return x

    def forward_understand(self, images):
        """Understanding pathway: image -> class logits.

        Args:
            images: (B, 16, 16, 3) input images

        Returns:
            logits: (B, num_classes) class logits
        """
        x = self._encode_patches(images)

        for block in self.blocks:
            x = block(x, task="understand")

        # Mean pool over patches -> classification
        x = mx.mean(x, axis=1)  # (B, hidden_dim)
        x = self.und_ln(x)
        logits = self.und_head(x)  # (B, num_classes)
        return logits

    def forward_generate(self, x_t, t, class_labels):
        """Generation pathway: noisy image + time + class -> velocity.

        Args:
            x_t: (B, 16, 16, 3) noisy image at time t
            t: (B,) timestep in [0, 1]
            class_labels: (B,) class indices

        Returns:
            v: (B, 16, 16, 3) predicted velocity field
        """
        x = self._encode_patches(x_t)

        # Conditioning
        t_emb = self.time_emb(t)
        c_emb = self.class_emb(class_labels)
        cond = t_emb + c_emb

        for block in self.blocks:
            x = block(x, task="generate", cond=cond)

        # Project back to image space
        x = self.gen_ln(x)
        v_patches = self.gen_head(x)
        v = self.unpatchify(v_patches)
        return v


if __name__ == "__main__":
    model = NeoUnifyModel()
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Neo-Unify parameters: {n_params:,}")

    # Test understanding
    images = mx.random.uniform(shape=(4, 16, 16, 3))
    logits = model.forward_understand(images)
    mx.eval(logits)  # MLX lazy computation trigger
    print(f"Understanding: {images.shape} -> logits {logits.shape}")

    # Test generation
    x_t = mx.random.normal((4, 16, 16, 3))
    t = mx.random.uniform(shape=(4,))
    labels = mx.array([0, 1, 2, 3])
    v = model.forward_generate(x_t, t, labels)
    mx.eval(v)  # MLX lazy computation trigger
    print(f"Generation: {x_t.shape} -> velocity {v.shape}")
