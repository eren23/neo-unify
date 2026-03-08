"""Generate images using flow matching ODE sampling."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from phase_b.model import FlowMatchingTransformer
from shared.data import CLASSES, NUM_CLASSES
from shared.utils import save_image_grid, WEIGHTS_DIR


def load_flow_model():
    model = FlowMatchingTransformer(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=4, num_classes=6,
    )
    weights = mx.load(f"{WEIGHTS_DIR}/flow.npz")
    model.load_weights(list(weights.items()))
    return model


def euler_sample(model, class_labels, num_steps=50):
    """Generate images via Euler ODE integration from t=0 (noise) to t=1 (data)."""
    B = class_labels.shape[0]
    dt = 1.0 / num_steps

    x = mx.random.normal((B, 16, 16, 3))

    for i in range(num_steps):
        t = mx.full((B,), i * dt)
        v = model(x, t, class_labels)
        x = x + v * dt
        mx.eval(x)

    return x


def generate(samples_per_class=8, num_steps=50):
    print("=" * 60)
    print("Phase B: Flow Matching Generation")
    print("=" * 60)
    print(f"ODE steps: {num_steps}")

    model = load_flow_model()
    print("Model loaded.")

    all_images = []

    for cls_idx, cls_name in enumerate(CLASSES):
        print(f"Generating {samples_per_class} '{cls_name}' images...")
        class_labels = mx.full((samples_per_class,), cls_idx, dtype=mx.int32)

        images = euler_sample(model, class_labels, num_steps=num_steps)
        images = mx.clip(images, 0.0, 1.0)
        mx.eval(images)

        all_images.append(images)

        save_image_grid(
            images, f"flow_gen_{cls_name}.png",
            title=f"Flow Generated: {cls_name}", nrow=samples_per_class,
        )

    all_images = mx.concatenate(all_images, axis=0)
    save_image_grid(
        all_images, "flow_generated_all.png",
        title="Flow Generated (all classes)", nrow=samples_per_class,
    )

    # Compare step counts
    print("\nComparing step counts...")
    test_labels = mx.full((4,), 0, dtype=mx.int32)
    for steps in [10, 20, 50]:
        imgs = euler_sample(model, test_labels, num_steps=steps)
        imgs = mx.clip(imgs, 0.0, 1.0)
        mx.eval(imgs)
        save_image_grid(
            imgs, f"flow_steps_{steps}.png",
            title=f"Flow (spiral, {steps} steps)", nrow=4,
        )

    return all_images


if __name__ == "__main__":
    generate()
