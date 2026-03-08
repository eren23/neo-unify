"""Side-by-side comparison of Phase A (AR), Phase B (Flow), and Phase C (Neo-Unify)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from shared.data import generate_dataset, CLASSES, NUM_CLASSES
from phase_a.generate import load_vqvae, load_transformer
from phase_b.generate import load_flow_model, euler_sample as flow_euler_sample
from phase_c.generate import load_model as load_neo_unify, euler_sample as neo_euler_sample
from shared.utils import save_comparison_grid, save_image_grid, to_numpy


def compare(samples_per_class=4, num_steps=50, temperature=0.8, top_k=50):
    print("=" * 60)
    print("Comparison: AR vs Flow Matching vs Neo-Unify")
    print("=" * 60)

    # Load models
    print("Loading models...")
    vqvae = load_vqvae()
    transformer = load_transformer()
    flow_model = load_flow_model()
    neo_unify = load_neo_unify()

    # Generate real samples
    images, labels = generate_dataset(samples_per_class=1000)

    real_samples = []
    ar_samples = []
    flow_samples = []
    neo_samples = []

    for cls_idx, cls_name in enumerate(CLASSES):
        print(f"\nClass: {cls_name}")

        # Real samples
        cls_mask = np.array(labels) == cls_idx
        cls_indices = np.where(cls_mask)[0]
        real = images[mx.array(cls_indices[:samples_per_class])]
        # Force lazy computation (MLX)
        mx.eval(real)  # noqa: S307
        real_samples.append(real)

        class_labels = mx.full((samples_per_class,), cls_idx, dtype=mx.int32)

        # AR generation
        tokens = transformer.generate(class_labels, temperature=temperature, top_k=top_k)
        # Force lazy computation (MLX)
        mx.eval(tokens)  # noqa: S307
        token_grid = tokens.reshape(samples_per_class, 4, 4)
        ar_imgs = vqvae.decode(token_grid)
        # Force lazy computation (MLX)
        mx.eval(ar_imgs)  # noqa: S307
        ar_samples.append(ar_imgs)

        # Flow generation
        flow_imgs = flow_euler_sample(flow_model, class_labels, num_steps=num_steps)
        flow_imgs = mx.clip(flow_imgs, 0.0, 1.0)
        # Force lazy computation (MLX)
        mx.eval(flow_imgs)  # noqa: S307
        flow_samples.append(flow_imgs)

        # Neo-Unify generation
        neo_imgs = neo_euler_sample(neo_unify, class_labels, num_steps=num_steps)
        neo_imgs = mx.clip(neo_imgs, 0.0, 1.0)
        # Force lazy computation (MLX)
        mx.eval(neo_imgs)  # noqa: S307
        neo_samples.append(neo_imgs)

        print(f"  Real: mean={real.mean().item():.3f}")
        print(f"  AR:   mean={ar_imgs.mean().item():.3f}")
        print(f"  Flow: mean={flow_imgs.mean().item():.3f}")
        print(f"  Neo:  mean={neo_imgs.mean().item():.3f}")

    # Stack all
    real_all = mx.concatenate(real_samples, axis=0)
    ar_all = mx.concatenate(ar_samples, axis=0)
    flow_all = mx.concatenate(flow_samples, axis=0)
    neo_all = mx.concatenate(neo_samples, axis=0)

    # 4-column comparison
    save_comparison_grid(
        [real_all, ar_all, flow_all, neo_all],
        ["Real", "AR", "Flow", "Neo-Unify"],
        CLASSES,
        "comparison.png",
    )

    # Neo-Unify classification accuracy
    print("\n--- Neo-Unify Classification ---")
    correct = 0
    total = 0
    batch_size = 256
    for i in range(0, images.shape[0], batch_size):
        batch_images = images[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        logits = neo_unify.forward_understand(batch_images)
        preds = mx.argmax(logits, axis=-1)
        # Force lazy computation (MLX)
        mx.eval(preds)  # noqa: S307
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.shape[0]
    print(f"Neo-Unify classification accuracy: {correct}/{total} = {correct/total:.1%}")

    print("\nComparison complete! Check outputs/ directory.")


if __name__ == "__main__":
    compare()
