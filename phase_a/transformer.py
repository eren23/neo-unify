"""Tiny class-conditional GPT for autoregressive image token prediction.

Input: [CLASS_TOKEN] + [16 image tokens from VQ-VAE 4x4 grid]
Output: next-token prediction over 256 codebook entries
"""

import mlx.core as mx
import mlx.nn as nn
import math


class CausalSelfAttention(nn.Module):
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

        # Causal mask
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ImageGPT(nn.Module):
    """Class-conditional autoregressive transformer for image tokens.

    Sequence: [class_emb, tok_0, tok_1, ..., tok_15]
    Predicts: [tok_0, tok_1, ..., tok_15] (shifted by 1)
    """

    def __init__(
        self,
        vocab_size=256,
        num_classes=6,
        seq_len=16,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # Token embedding for codebook indices
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        # Class embedding
        self.cls_emb = nn.Embedding(num_classes, hidden_dim)
        # Positional embedding for full sequence (1 class + seq_len tokens)
        self.pos_emb = nn.Embedding(1 + seq_len, hidden_dim)

        self.blocks = [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def __call__(self, tokens, class_labels):
        """Forward pass.

        Args:
            tokens: (B, seq_len) codebook indices
            class_labels: (B,) class indices

        Returns:
            logits: (B, seq_len, vocab_size) - predictions for each position
        """
        B, T = tokens.shape

        # Embed class token: (B, 1, hidden)
        cls_emb = self.cls_emb(class_labels).reshape(B, 1, -1)

        # Embed image tokens: (B, T, hidden)
        tok_emb = self.tok_emb(tokens)

        # Concatenate: [class, tok_0, ..., tok_{T-1}]
        x = mx.concatenate([cls_emb, tok_emb], axis=1)  # (B, 1+T, hidden)

        # Add positional embeddings
        positions = mx.arange(1 + T)
        x = x + self.pos_emb(positions)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, 1+T, vocab_size)

        # We predict tokens 0..T-1 from positions 0..T-1 (class, tok_0, ..., tok_{T-2})
        # So logits at position i predicts token i
        logits = logits[:, :-1, :]  # (B, T, vocab_size) - drop last position

        return logits

    def generate(self, class_labels, temperature=1.0, top_k=0):
        """Autoregressive generation.

        Args:
            class_labels: (B,) class indices
            temperature: sampling temperature
            top_k: if > 0, only sample from top-k tokens

        Returns:
            tokens: (B, seq_len) generated codebook indices
        """
        B = class_labels.shape[0]
        tokens = []

        for i in range(self.seq_len):
            if len(tokens) == 0:
                # First step: only class embedding
                cls_emb = self.cls_emb(class_labels).reshape(B, 1, -1)
                x = cls_emb + self.pos_emb(mx.array([0]))
            else:
                # Build sequence so far
                curr_tokens = mx.stack(tokens, axis=1)  # (B, i)
                cls_emb = self.cls_emb(class_labels).reshape(B, 1, -1)
                tok_emb = self.tok_emb(curr_tokens)
                x = mx.concatenate([cls_emb, tok_emb], axis=1)
                positions = mx.arange(1 + i)
                x = x + self.pos_emb(positions)

            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.head(x[:, -1, :])  # (B, vocab_size)

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_vals = mx.sort(logits, axis=-1)[:, -top_k:]
                threshold = top_vals[:, 0:1]
                logits = mx.where(logits < threshold, mx.full(logits.shape, -1e9), logits)

            # Sample
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))  # (B,)
            tokens.append(next_token)
            mx.eval(next_token)  # MLX lazy evaluation trigger

        return mx.stack(tokens, axis=1)  # (B, seq_len)


if __name__ == "__main__":
    model = ImageGPT()
    tokens = mx.zeros((4, 16), dtype=mx.int32)
    labels = mx.array([0, 1, 2, 3])
    logits = model(tokens, labels)
    mx.eval(logits)  # MLX lazy evaluation trigger
    print(f"Tokens: {tokens.shape}")
    print(f"Logits: {logits.shape}")
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")

    # Test generation
    gen = model.generate(labels, temperature=0.8, top_k=50)
    mx.eval(gen)  # MLX lazy evaluation trigger
    print(f"Generated: {gen.shape}")
