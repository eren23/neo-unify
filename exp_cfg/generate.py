"""Exp CFG: Generation with RK2 solver, CFG inference, and guidance scale sweep."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from exp_cfg.model import NeoUnifyModel
from shared.data import generate_dataset, CLASSES, NUM_CLASSES
from shared.utils import save_image_grid, WEIGHTS_DIR


def load_model(use_ema=True):
    model = NeoUnifyModel(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    )
    weight_file = f"{WEIGHTS_DIR}/exp_cfg_ema.npz" if use_ema else f"{WEIGHTS_DIR}/exp_cfg.npz"
    weights = mx.load(weight_file)
    model.load_weights(list(weights.items()))
    print(f"Loaded weights: {weight_file}")
    return model


def rk2_cfg_sample(model, class_labels, num_steps=100, guidance_scale=3.0):
    """Generate images via RK2 midpoint ODE integration with CFG."""
    B = class_labels.shape[0]
    dt = 1.0 / num_steps
    null_labels = mx.full((B,), NUM_CLASSES, dtype=mx.int32)

    x = mx.random.normal((B, 16, 16, 3))

    for i in range(num_steps):
        t = mx.full((B,), i * dt)
        t_mid = mx.full((B,), (i + 0.5) * dt)

        # RK2 step 1: evaluate at current point
        v_cond = model.forward_generate(x, t, class_labels)
        v_uncond = model.forward_generate(x, t, null_labels)
        v1 = v_uncond + guidance_scale * (v_cond - v_uncond)

        # RK2 step 2: evaluate at midpoint
        x_mid = x + v1 * (dt / 2)
        v_cond_mid = model.forward_generate(x_mid, t_mid, class_labels)
        v_uncond_mid = model.forward_generate(x_mid, t_mid, null_labels)
        v2 = v_uncond_mid + guidance_scale * (v_cond_mid - v_uncond_mid)

        x = x + v2 * dt
        # MLX lazy computation trigger
        mx.eval(x)  # noqa: S307

    return x


def generate(samples_per_class=8, num_steps=100):
    print("=" * 60)
    print("Exp CFG: Generation with RK2 + CFG")
    print("=" * 60)

    model = load_model(use_ema=True)

    # --- Guidance scale sweep ---
    guidance_scales = [1.0, 2.0, 3.0, 5.0]

    for gs in guidance_scales:
        print(f"\n--- Guidance scale = {gs} ---")
        all_images = []

        # Use same seed for each scale for fair comparison
        mx.random.seed(42)

        for cls_idx, cls_name in enumerate(CLASSES):
            class_labels = mx.full((samples_per_class,), cls_idx, dtype=mx.int32)
            images = rk2_cfg_sample(model, class_labels, num_steps=num_steps, guidance_scale=gs)
            images = mx.clip(images, 0.0, 1.0)
            # MLX lazy computation trigger
            mx.eval(images)  # noqa: S307
            all_images.append(images)

        all_images_cat = mx.concatenate(all_images, axis=0)
        save_image_grid(
            all_images_cat, f"exp_cfg_gs{gs:.1f}.png",
            title=f"Exp CFG (guidance={gs:.1f})", nrow=samples_per_class,
        )

    # --- Classification accuracy (using main weights, not EMA) ---
    print("\n--- Understanding (Classification) ---")
    model_main = load_model(use_ema=False)
    images, labels = generate_dataset(samples_per_class=1000)

    correct = 0
    total = 0
    batch_size = 256

    for i in range(0, images.shape[0], batch_size):
        batch_images = images[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        logits = model_main.forward_understand(batch_images)
        preds = mx.argmax(logits, axis=-1)
        # MLX lazy computation trigger
        mx.eval(preds)  # noqa: S307
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.shape[0]

    accuracy = correct / total
    print(f"Classification accuracy: {correct}/{total} = {accuracy:.1%}")

    for cls_idx, cls_name in enumerate(CLASSES):
        mask = np.array(labels) == cls_idx
        cls_images = images[mx.array(np.where(mask)[0])]
        cls_labels_arr = labels[mx.array(np.where(mask)[0])]
        logits = model_main.forward_understand(cls_images)
        preds = mx.argmax(logits, axis=-1)
        # MLX lazy computation - mx.eval triggers MLX array evaluation, not Python's eval
        mx.eval(preds)  # noqa: S307
        cls_acc = (preds == cls_labels_arr).sum().item() / cls_labels_arr.shape[0]
        print(f"  {cls_name}: {cls_acc:.1%}")

    # --- Reconstruction visualization ---
    print("\n--- Reconstruction ---")
    recon_per_class = 8
    recon_rows = []
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = np.array(labels) == cls_idx
        cls_indices = np.where(mask)[0][:recon_per_class]
        cls_images = images[mx.array(cls_indices)]
        recon = model_main.forward_reconstruct(cls_images)
        # mx.eval triggers MLX array materialization, not Python's eval
        mx.eval(recon)  # noqa: S307
        recon_rows.append(cls_images)
        recon_rows.append(recon)

    recon_mse = mx.mean((recon_rows[1] - recon_rows[0]) ** 2)
    # mx.eval triggers MLX array materialization, not Python's eval
    mx.eval(recon_mse)  # noqa: S307
    print(f"Sample recon MSE: {recon_mse.item():.6f}")

    recon_grid = mx.concatenate(recon_rows, axis=0)
    save_image_grid(
        recon_grid, "exp_cfg_recon.png",
        title="Exp CFG Reconstruction (orig / recon per class)",
        nrow=recon_per_class,
    )


if __name__ == "__main__":
    generate()
