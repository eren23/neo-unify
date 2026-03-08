"""Synthetic 16x16 RGB dataset generator for unified multimodal experiments."""

import mlx.core as mx
import numpy as np
import math


CLASSES = ["spiral", "triangle", "circle", "cross", "checkerboard", "gradient"]
NUM_CLASSES = len(CLASSES)


def generate_spiral(rng):
    """Archimedean spiral of colored pixels on black."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    color = rng.random(3).astype(np.float32) * 0.5 + 0.5
    n_points = rng.integers(60, 120)
    direction = rng.choice([-1, 1])
    offset = rng.random() * 2 * math.pi
    for i in range(n_points):
        t = i / n_points * 3 * math.pi
        r = t / (3 * math.pi) * 7
        x = int(8 + r * math.cos(direction * t + offset))
        y = int(8 + r * math.sin(direction * t + offset))
        if 0 <= x < 16 and 0 <= y < 16:
            fade = 0.5 + 0.5 * (i / n_points)
            img[y, x] = color * fade
    return img


def generate_triangle(rng):
    """Filled triangle with random color."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    color = rng.random(3).astype(np.float32) * 0.5 + 0.5

    # Random triangle vertices within bounds
    cx, cy = rng.integers(5, 11), rng.integers(5, 11)
    size = rng.integers(3, 7)

    # Simple equilateral-ish triangle
    pts = [
        (cx, cy - size),
        (cx - size, cy + size // 2),
        (cx + size, cy + size // 2),
    ]

    for y in range(16):
        for x in range(16):
            # Barycentric check
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

            d1 = sign((x, y), pts[0], pts[1])
            d2 = sign((x, y), pts[1], pts[2])
            d3 = sign((x, y), pts[2], pts[0])
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            if not (has_neg and has_pos):
                img[y, x] = color
    return img


def generate_circle(rng):
    """Filled circle with random color."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    color = rng.random(3).astype(np.float32) * 0.5 + 0.5
    cx = rng.integers(4, 12)
    cy = rng.integers(4, 12)
    r = rng.integers(2, 6)
    for y in range(16):
        for x in range(16):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                img[y, x] = color
    return img


def generate_cross(rng):
    """Two diagonal lines forming an X."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    color = rng.random(3).astype(np.float32) * 0.5 + 0.5
    cx, cy = rng.integers(4, 12), rng.integers(4, 12)
    size = rng.integers(3, 7)
    thickness = rng.integers(1, 3)

    for i in range(-size, size + 1):
        for t in range(-thickness + 1, thickness):
            # Diagonal 1
            x, y = cx + i, cy + i + t
            if 0 <= x < 16 and 0 <= y < 16:
                img[y, x] = color
            # Diagonal 2
            x, y = cx + i, cy - i + t
            if 0 <= x < 16 and 0 <= y < 16:
                img[y, x] = color
    return img


def generate_checkerboard(rng):
    """Alternating color pattern."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    color1 = rng.random(3).astype(np.float32) * 0.5 + 0.5
    color2 = rng.random(3).astype(np.float32) * 0.3
    block_size = rng.choice([2, 4])

    for y in range(16):
        for x in range(16):
            if ((x // block_size) + (y // block_size)) % 2 == 0:
                img[y, x] = color1
            else:
                img[y, x] = color2
    return img


def generate_gradient(rng):
    """Linear color gradient."""
    img = np.zeros((16, 16, 3), dtype=np.float32)
    color_start = rng.random(3).astype(np.float32)
    color_end = rng.random(3).astype(np.float32)

    # Random angle (0=horizontal, 1=vertical, 2=diagonal)
    direction = rng.integers(0, 3)

    for y in range(16):
        for x in range(16):
            if direction == 0:
                t = x / 15.0
            elif direction == 1:
                t = y / 15.0
            else:
                t = (x + y) / 30.0
            img[y, x] = color_start * (1 - t) + color_end * t
    return img


GENERATORS = {
    "spiral": generate_spiral,
    "triangle": generate_triangle,
    "circle": generate_circle,
    "cross": generate_cross,
    "checkerboard": generate_checkerboard,
    "gradient": generate_gradient,
}


def generate_dataset(samples_per_class=1000, seed=42):
    """Generate synthetic dataset.

    Returns:
        images: mx.array of shape (N, 16, 16, 3) in [0, 1]
        labels: mx.array of shape (N,) with class indices
    """
    rng = np.random.default_rng(seed)
    all_images = []
    all_labels = []

    for cls_idx, cls_name in enumerate(CLASSES):
        gen_fn = GENERATORS[cls_name]
        for _ in range(samples_per_class):
            img = gen_fn(rng)
            img = np.clip(img, 0.0, 1.0)
            all_images.append(img)
            all_labels.append(cls_idx)

    # Shuffle
    indices = rng.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    images = mx.array(np.stack(all_images))
    labels = mx.array(np.array(all_labels, dtype=np.int32))

    return images, labels


if __name__ == "__main__":
    images, labels = generate_dataset(samples_per_class=100)
    print(f"Dataset: {images.shape}, labels: {labels.shape}")
    print(f"Value range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    for i, name in enumerate(CLASSES):
        count = (labels == i).sum().item()
        print(f"  {name}: {count} samples")
