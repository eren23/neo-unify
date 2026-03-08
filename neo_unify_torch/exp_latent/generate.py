"""Exp Latent: Generation in latent space with RK2 + CFG, decoded through VQ-VAE (PyTorch)."""

import torch
import numpy as np

from neo_unify_torch.exp_latent.model import NeoUnifyLatentModel
from neo_unify_torch.vqvae.model import VQVAE
from neo_unify_torch.shared.data import generate_dataset, CLASSES, NUM_CLASSES
from neo_unify_torch.shared.utils import save_image_grid, WEIGHTS_DIR, get_device


def load_model(device=None, use_ema=True):
    if device is None:
        device = get_device()
    model = NeoUnifyLatentModel(
        patch_size=4, image_size=16, channels=3, latent_dim=64,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    ).to(device)
    weight_file = f"{WEIGHTS_DIR}/exp_latent_ema_torch.pt" if use_ema else f"{WEIGHTS_DIR}/exp_latent_torch.pt"
    state = torch.load(weight_file, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded weights: {weight_file}")
    return model


def load_vqvae(device=None):
    if device is None:
        device = get_device()
    vqvae = VQVAE(num_embeddings=256, embedding_dim=64, commitment_cost=0.25).to(device)
    state = torch.load(f"{WEIGHTS_DIR}/vqvae_torch.pt", map_location=device, weights_only=True)
    vqvae.load_state_dict(state)
    print("Loaded frozen VQ-VAE decoder")
    return vqvae


@torch.no_grad()
def rk2_cfg_latent_sample(model, class_labels, num_steps=100, guidance_scale=3.0):
    """Generate latents via RK2 midpoint ODE with CFG, in (B,64,4,4) latent space."""
    device = class_labels.device
    B = class_labels.shape[0]
    dt = 1.0 / num_steps
    null_labels = torch.full((B,), NUM_CLASSES, dtype=torch.long, device=device)

    z = torch.randn(B, 64, 4, 4, device=device)

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        t_mid = torch.full((B,), (i + 0.5) * dt, device=device)

        # RK2 step 1
        v_cond = model.forward_generate(z, t, class_labels)
        v_uncond = model.forward_generate(z, t, null_labels)
        v1 = v_uncond + guidance_scale * (v_cond - v_uncond)

        # RK2 step 2: midpoint
        z_mid = z + v1 * (dt / 2)
        v_cond_mid = model.forward_generate(z_mid, t_mid, class_labels)
        v_uncond_mid = model.forward_generate(z_mid, t_mid, null_labels)
        v2 = v_uncond_mid + guidance_scale * (v_cond_mid - v_uncond_mid)

        z = z + v2 * dt

    return z


def generate(samples_per_class=8, num_steps=100):
    print("=" * 60)
    print("Exp Latent: Latent-Space Generation with RK2 + CFG [PyTorch]")
    print("=" * 60)

    device = get_device()
    model = load_model(device, use_ema=True)
    vqvae = load_vqvae(device)

    # --- Guidance scale sweep ---
    guidance_scales = [1.0, 2.0, 3.0, 5.0]

    for gs in guidance_scales:
        print(f"\n--- Guidance scale = {gs} ---")
        all_images = []

        torch.manual_seed(42)

        for cls_idx, cls_name in enumerate(CLASSES):
            class_labels = torch.full((samples_per_class,), cls_idx, dtype=torch.long, device=device)

            # Generate in latent space
            z_gen = rk2_cfg_latent_sample(model, class_labels, num_steps=num_steps, guidance_scale=gs)

            # Decode through frozen VQ-VAE decoder
            with torch.no_grad():
                images = vqvae.decoder(z_gen)
            images = torch.clamp(images, 0.0, 1.0)
            all_images.append(images)

        all_images_cat = torch.cat(all_images, dim=0)
        save_image_grid(
            all_images_cat, f"exp_latent_gs{gs:.1f}_torch.png",
            title=f"Exp Latent (guidance={gs:.1f})", nrow=samples_per_class,
        )

    # --- Classification accuracy (using main weights) ---
    print("\n--- Understanding (Classification) ---")
    model_main = load_model(device, use_ema=False)
    images, labels = generate_dataset(samples_per_class=1000)
    images = images.to(device)
    labels = labels.to(device)

    correct = 0
    total = 0
    batch_size = 256

    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch_images = images[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            logits = model_main.forward_understand(batch_images)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.shape[0]

    accuracy = correct / total
    print(f"Classification accuracy: {correct}/{total} = {accuracy:.1%}")

    labels_np = labels.cpu().numpy()
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = labels_np == cls_idx
        cls_indices = torch.from_numpy(np.where(mask)[0]).to(device)
        cls_images = images[cls_indices]
        cls_labels = labels[cls_indices]
        with torch.no_grad():
            logits = model_main.forward_understand(cls_images)
            preds = logits.argmax(dim=-1)
        cls_acc = (preds == cls_labels).sum().item() / cls_labels.shape[0]
        print(f"  {cls_name}: {cls_acc:.1%}")

    # --- Reconstruction visualization ---
    print("\n--- Reconstruction ---")
    recon_per_class = 8
    recon_rows = []
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = labels_np == cls_idx
        cls_indices = np.where(mask)[0][:recon_per_class]
        cls_images = images[torch.from_numpy(cls_indices).to(device)]
        with torch.no_grad():
            recon = model_main.forward_reconstruct(cls_images)
        recon_rows.append(cls_images)
        recon_rows.append(recon)

    recon_mse = torch.nn.functional.mse_loss(recon_rows[1], recon_rows[0])
    print(f"Sample recon MSE: {recon_mse.item():.6f}")

    recon_grid = torch.cat(recon_rows, dim=0)
    save_image_grid(
        recon_grid, "exp_latent_recon_torch.png",
        title="Exp Latent Reconstruction (orig / recon per class)",
        nrow=recon_per_class,
    )


if __name__ == "__main__":
    generate()
