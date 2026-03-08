"""Generate images and evaluate classification using trained Neo-Unify model."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from phase_c.model import NeoUnifyModel
from shared.data import generate_dataset, CLASSES, NUM_CLASSES
from shared.utils import save_image_grid, WEIGHTS_DIR


def load_model():
    model = NeoUnifyModel(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    )
    weights = mx.load(f"{WEIGHTS_DIR}/neo_unify.npz")
    model.load_weights(list(weights.items()))
    return model


def euler_sample(model, class_labels, num_steps=50):
    """Generate images via Euler ODE integration using the generation pathway."""
    B = class_labels.shape[0]
    dt = 1.0 / num_steps

    x = mx.random.normal((B, 16, 16, 3))

    for i in range(num_steps):
        t = mx.full((B,), i * dt)
        v = model.forward_generate(x, t, class_labels)
        x = x + v * dt
        # Force lazy computation (MLX)
        mx.eval(x)  # noqa: S307

    return x


def generate(samples_per_class=8, num_steps=50):
    print("=" * 60)
    print("Phase C: Neo-Unify Generation + Understanding")
    print("=" * 60)
    print(f"ODE steps: {num_steps}")

    model = load_model()
    print("Model loaded.")

    # --- Generation ---
    print("\n--- Generation ---")
    all_images = []

    for cls_idx, cls_name in enumerate(CLASSES):
        print(f"Generating {samples_per_class} '{cls_name}' images...")
        class_labels = mx.full((samples_per_class,), cls_idx, dtype=mx.int32)

        images = euler_sample(model, class_labels, num_steps=num_steps)
        images = mx.clip(images, 0.0, 1.0)
        # Force lazy computation (MLX)
        mx.eval(images)  # noqa: S307

        all_images.append(images)

        save_image_grid(
            images, f"neounify_gen_{cls_name}.png",
            title=f"Neo-Unify Generated: {cls_name}", nrow=samples_per_class,
        )

    all_images = mx.concatenate(all_images, axis=0)
    save_image_grid(
        all_images, "neounify_generated_all.png",
        title="Neo-Unify Generated (all classes)", nrow=samples_per_class,
    )

    # --- Understanding (Classification) ---
    print("\n--- Understanding (Classification) ---")
    images, labels = generate_dataset(samples_per_class=1000)

    correct = 0
    total = 0
    batch_size = 256

    for i in range(0, images.shape[0], batch_size):
        batch_images = images[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        logits = model.forward_understand(batch_images)
        preds = mx.argmax(logits, axis=-1)
        # Force lazy computation (MLX)
        mx.eval(preds)  # noqa: S307
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.shape[0]

    accuracy = correct / total
    print(f"Classification accuracy: {correct}/{total} = {accuracy:.1%}")

    # Per-class accuracy
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = np.array(labels) == cls_idx
        cls_images = images[mx.array(np.where(mask)[0])]
        cls_labels_arr = labels[mx.array(np.where(mask)[0])]
        logits = model.forward_understand(cls_images)
        preds = mx.argmax(logits, axis=-1)
        # Force lazy computation (MLX)
        mx.eval(preds)  # noqa: S307
        cls_acc = (preds == cls_labels_arr).sum().item() / cls_labels_arr.shape[0]
        print(f"  {cls_name}: {cls_acc:.1%}")

    return all_images


if __name__ == "__main__":
    generate()
