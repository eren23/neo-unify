"""Generate images and classify using trained Neo-Unify model (PyTorch)."""

import torch
import numpy as np

from neo_unify_torch.phase_c.model import NeoUnifyModel
from neo_unify_torch.shared.data import generate_dataset, CLASSES, NUM_CLASSES
from neo_unify_torch.shared.utils import save_image_grid, WEIGHTS_DIR, get_device


def load_model(device=None):
    if device is None:
        device = get_device()
    model = NeoUnifyModel(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    ).to(device)
    state = torch.load(f"{WEIGHTS_DIR}/neo_unify_torch.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def euler_sample(model, class_labels, num_steps=50):
    """Generate images via Euler ODE integration."""
    device = class_labels.device
    B = class_labels.shape[0]
    dt = 1.0 / num_steps

    x = torch.randn(B, 3, 16, 16, device=device)

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model.forward_generate(x, t, class_labels)
        x = x + v * dt

    return x


def generate(samples_per_class=8, num_steps=50):
    print("=" * 60)
    print("Phase C: Neo-Unify Generation + Understanding [PyTorch]")
    print("=" * 60)
    print(f"ODE steps: {num_steps}")

    device = get_device()
    model = load_model(device)

    # --- Generation ---
    print("\n--- Generation ---")
    all_images = []

    for cls_idx, cls_name in enumerate(CLASSES):
        print(f"Generating {samples_per_class} '{cls_name}' images...")
        class_labels = torch.full((samples_per_class,), cls_idx, dtype=torch.long, device=device)

        images = euler_sample(model, class_labels, num_steps=num_steps)
        images = torch.clamp(images, 0.0, 1.0)

        all_images.append(images)

        save_image_grid(
            images, f"neounify_gen_{cls_name}_torch.png",
            title=f"Neo-Unify Generated: {cls_name}", nrow=samples_per_class,
        )

    all_images = torch.cat(all_images, dim=0)
    save_image_grid(
        all_images, "neounify_generated_all_torch.png",
        title="Neo-Unify Generated (all classes)", nrow=samples_per_class,
    )

    # --- Understanding (Classification) ---
    print("\n--- Understanding (Classification) ---")
    test_images, test_labels = generate_dataset(samples_per_class=1000)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    correct = 0
    total = 0
    batch_size = 256

    with torch.no_grad():
        for i in range(0, test_images.shape[0], batch_size):
            batch_images = test_images[i : i + batch_size]
            batch_labels = test_labels[i : i + batch_size]
            logits = model.forward_understand(batch_images)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.shape[0]

    accuracy = correct / total
    print(f"Classification accuracy: {correct}/{total} = {accuracy:.1%}")

    # Per-class accuracy
    labels_np = test_labels.cpu().numpy()
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = labels_np == cls_idx
        cls_indices = torch.from_numpy(np.where(mask)[0]).to(device)
        cls_images = test_images[cls_indices]
        cls_labels = test_labels[cls_indices]
        with torch.no_grad():
            logits = model.forward_understand(cls_images)
            preds = logits.argmax(dim=-1)
        cls_acc = (preds == cls_labels).sum().item() / cls_labels.shape[0]
        print(f"  {cls_name}: {cls_acc:.1%}")

    # --- Generation MSE ---
    print("\n--- Generation MSE ---")
    total_mse = 0.0
    n_mse = 0
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = labels_np == cls_idx
        cls_real = test_images[torch.from_numpy(np.where(mask)[0]).to(device)]
        real_mean = cls_real.mean(dim=0, keepdim=True)
        cls_gen = all_images[cls_idx * samples_per_class : (cls_idx + 1) * samples_per_class]
        mse = torch.nn.functional.mse_loss(cls_gen, real_mean.expand_as(cls_gen)).item()
        total_mse += mse
        n_mse += 1
        print(f"  {cls_name}: MSE vs class mean = {mse:.4f}")
    print(f"  Average: {total_mse / n_mse:.4f}")

    return all_images


if __name__ == "__main__":
    generate()
