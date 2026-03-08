"""Side-by-side comparison of Phase C, Exp CFG, and Exp Latent (PyTorch)."""

import torch
import torch.nn.functional as F
import numpy as np

from neo_unify_torch.shared.data import generate_dataset, CLASSES, NUM_CLASSES
from neo_unify_torch.phase_c.generate import load_model as load_phase_c, euler_sample
from neo_unify_torch.exp_cfg.generate import load_model as load_exp_cfg, rk2_cfg_sample
from neo_unify_torch.exp_latent.generate import (
    load_model as load_exp_latent, load_vqvae, rk2_cfg_latent_sample,
)
from neo_unify_torch.shared.utils import save_comparison_grid, save_image_grid, get_device


def compare(samples_per_class=4, num_steps_euler=50, num_steps_rk2=100, guidance_scale=3.0):
    print("=" * 60)
    print("Comparison: Phase C vs Exp CFG vs Exp Latent [PyTorch]")
    print("=" * 60)

    device = get_device()

    # Load models
    print("Loading models...")
    phase_c_model = load_phase_c(device)
    cfg_model = load_exp_cfg(device, use_ema=True)
    latent_model = load_exp_latent(device, use_ema=True)
    vqvae = load_vqvae(device)

    # Generate real samples
    images, labels = generate_dataset(samples_per_class=1000)
    images = images.to(device)
    labels = labels.to(device)

    real_samples = []
    phase_c_samples = []
    cfg_samples = []
    latent_samples = []

    torch.manual_seed(42)

    labels_np = labels.cpu().numpy()
    for cls_idx, cls_name in enumerate(CLASSES):
        print(f"\nClass: {cls_name}")

        # Real samples
        cls_mask = labels_np == cls_idx
        cls_indices = torch.from_numpy(np.where(cls_mask)[0][:samples_per_class]).to(device)
        real = images[cls_indices]
        real_samples.append(real)

        class_labels = torch.full((samples_per_class,), cls_idx, dtype=torch.long, device=device)

        # Phase C generation (Euler)
        pc_imgs = euler_sample(phase_c_model, class_labels, num_steps=num_steps_euler)
        pc_imgs = torch.clamp(pc_imgs, 0.0, 1.0)
        phase_c_samples.append(pc_imgs)

        # Exp CFG generation (RK2 + CFG)
        cfg_imgs = rk2_cfg_sample(cfg_model, class_labels, num_steps=num_steps_rk2, guidance_scale=guidance_scale)
        cfg_imgs = torch.clamp(cfg_imgs, 0.0, 1.0)
        cfg_samples.append(cfg_imgs)

        # Exp Latent generation (RK2 + CFG in latent space)
        z_gen = rk2_cfg_latent_sample(latent_model, class_labels, num_steps=num_steps_rk2, guidance_scale=guidance_scale)
        with torch.no_grad():
            lat_imgs = vqvae.decoder(z_gen)
        lat_imgs = torch.clamp(lat_imgs, 0.0, 1.0)
        latent_samples.append(lat_imgs)

        print(f"  Real: mean={real.mean().item():.3f}")
        print(f"  Phase C: mean={pc_imgs.mean().item():.3f}")
        print(f"  Exp CFG: mean={cfg_imgs.mean().item():.3f}")
        print(f"  Exp Latent: mean={lat_imgs.mean().item():.3f}")

    # Stack all
    real_all = torch.cat(real_samples, dim=0)
    pc_all = torch.cat(phase_c_samples, dim=0)
    cfg_all = torch.cat(cfg_samples, dim=0)
    lat_all = torch.cat(latent_samples, dim=0)

    save_comparison_grid(
        [real_all, pc_all, cfg_all, lat_all],
        ["Real", "Phase C", "Exp CFG", "Exp Latent"],
        CLASSES,
        "comparison_torch.png",
    )

    # --- Generation MSE comparison ---
    print("\n--- Generation MSE (vs real class means) ---")
    test_images, test_labels = generate_dataset(samples_per_class=1000)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    for name, gen_imgs in [("Phase C", pc_all), ("Exp CFG", cfg_all), ("Exp Latent", lat_all)]:
        # Compute MSE against real images (using class means as proxy)
        total_mse = 0.0
        for cls_idx in range(NUM_CLASSES):
            cls_mask = test_labels.cpu().numpy() == cls_idx
            cls_real = test_images[torch.from_numpy(np.where(cls_mask)[0]).to(device)]
            real_mean = cls_real.mean(dim=0, keepdim=True)
            cls_gen = gen_imgs[cls_idx * samples_per_class : (cls_idx + 1) * samples_per_class]
            total_mse += F.mse_loss(cls_gen, real_mean.expand_as(cls_gen)).item()
        avg_mse = total_mse / NUM_CLASSES
        print(f"  {name}: avg class-mean MSE = {avg_mse:.4f}")

    # Phase C classification accuracy
    print("\n--- Phase C Classification ---")
    correct = 0
    total = 0
    batch_size = 256
    with torch.no_grad():
        for i in range(0, test_images.shape[0], batch_size):
            batch_images = test_images[i : i + batch_size]
            batch_labels = test_labels[i : i + batch_size]
            logits = phase_c_model.forward_understand(batch_images)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.shape[0]
    print(f"Phase C classification accuracy: {correct}/{total} = {correct/total:.1%}")

    print("\nComparison complete! Check outputs/ directory.")


if __name__ == "__main__":
    compare()
