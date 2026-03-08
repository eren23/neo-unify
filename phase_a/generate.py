"""Generate images using the trained AR pipeline (VQ-VAE + Transformer)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from phase_a.vqvae import VQVAE
from phase_a.transformer import ImageGPT
from shared.data import CLASSES, NUM_CLASSES
from shared.utils import save_image_grid, WEIGHTS_DIR


def load_vqvae():
    model = VQVAE(num_embeddings=256, embedding_dim=64, commitment_cost=0.25)
    weights = mx.load(f"{WEIGHTS_DIR}/vqvae.npz")
    model.load_weights(list(weights.items()))
    return model


def load_transformer():
    model = ImageGPT(
        vocab_size=256, num_classes=6, seq_len=16,
        hidden_dim=128, num_heads=4, num_layers=4,
    )
    weights = mx.load(f"{WEIGHTS_DIR}/transformer.npz")
    model.load_weights(list(weights.items()))
    return model


def generate(samples_per_class=8, temperature=0.8, top_k=50):
    print("=" * 60)
    print("Phase A: Autoregressive Generation")
    print("=" * 60)

    vqvae = load_vqvae()
    transformer = load_transformer()
    print("Models loaded.")

    all_images = []

    for cls_idx, cls_name in enumerate(CLASSES):
        print(f"Generating {samples_per_class} '{cls_name}' images...")
        class_labels = mx.full((samples_per_class,), cls_idx, dtype=mx.int32)

        tokens = transformer.generate(class_labels, temperature=temperature, top_k=top_k)
        mx.eval(tokens)  # noqa: S307

        token_grid = tokens.reshape(samples_per_class, 4, 4)
        images = vqvae.decode(token_grid)
        mx.eval(images)  # noqa: S307

        all_images.append(images)

        save_image_grid(
            images, f"ar_gen_{cls_name}.png",
            title=f"AR Generated: {cls_name}", nrow=samples_per_class,
        )

    all_images = mx.concatenate(all_images, axis=0)
    save_image_grid(
        all_images, "ar_generated_all.png",
        title="AR Generated (all classes)", nrow=samples_per_class,
    )

    return all_images


if __name__ == "__main__":
    generate()
