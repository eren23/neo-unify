"""Tests for generation pipelines."""

import torch
import pytest

from neo_unify_torch.phase_c.model import NeoUnifyModel as PhaseC
from neo_unify_torch.exp_cfg.model import NeoUnifyModel as ExpCFG
from neo_unify_torch.exp_latent.model import NeoUnifyLatentModel
from neo_unify_torch.vqvae.model import VQVAE


@pytest.fixture
def device():
    return torch.device("cpu")


class TestEulerSample:
    def test_output_shape(self, device):
        model = PhaseC().to(device)
        class_labels = torch.tensor([0, 1, 2, 3], device=device)

        # Inline euler sample with few steps for speed
        B = class_labels.shape[0]
        num_steps = 5
        dt = 1.0 / num_steps
        x = torch.randn(B, 3, 16, 16, device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=device)
                v = model.forward_generate(x, t, class_labels)
                x = x + v * dt

        assert x.shape == (4, 3, 16, 16)
        assert torch.isfinite(x).all(), "Generated images should be finite"

    def test_clipped_range(self, device):
        model = PhaseC().to(device)
        B = 4
        num_steps = 5
        dt = 1.0 / num_steps
        labels = torch.zeros(B, dtype=torch.long, device=device)
        x = torch.randn(B, 3, 16, 16, device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=device)
                v = model.forward_generate(x, t, labels)
                x = x + v * dt
        x = torch.clamp(x, 0.0, 1.0)
        assert x.min() >= 0.0, "Clipped values should be >= 0"
        assert x.max() <= 1.0, "Clipped values should be <= 1"


class TestRK2CFGSample:
    def test_output_shape(self, device):
        model = ExpCFG().to(device)
        B = 4
        num_steps = 3
        dt = 1.0 / num_steps
        class_labels = torch.tensor([0, 1, 2, 3], device=device)
        null_labels = torch.full((B,), 6, dtype=torch.long, device=device)
        guidance_scale = 2.0

        x = torch.randn(B, 3, 16, 16, device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=device)
                t_mid = torch.full((B,), (i + 0.5) * dt, device=device)

                v_cond = model.forward_generate(x, t, class_labels)
                v_uncond = model.forward_generate(x, t, null_labels)
                v1 = v_uncond + guidance_scale * (v_cond - v_uncond)

                x_mid = x + v1 * (dt / 2)
                v_cond_mid = model.forward_generate(x_mid, t_mid, class_labels)
                v_uncond_mid = model.forward_generate(x_mid, t_mid, null_labels)
                v2 = v_uncond_mid + guidance_scale * (v_cond_mid - v_uncond_mid)

                x = x + v2 * dt

        assert x.shape == (4, 3, 16, 16)
        assert torch.isfinite(x).all(), "RK2+CFG output should be finite"


class TestLatentGeneration:
    def test_latent_generate_shape(self, device):
        model = NeoUnifyLatentModel().to(device)
        B = 4
        num_steps = 3
        dt = 1.0 / num_steps
        class_labels = torch.tensor([0, 1, 2, 3], device=device)
        null_labels = torch.full((B,), 6, dtype=torch.long, device=device)
        guidance_scale = 2.0

        z = torch.randn(B, 64, 4, 4, device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=device)
                t_mid = torch.full((B,), (i + 0.5) * dt, device=device)

                v_cond = model.forward_generate(z, t, class_labels)
                v_uncond = model.forward_generate(z, t, null_labels)
                v1 = v_uncond + guidance_scale * (v_cond - v_uncond)

                z_mid = z + v1 * (dt / 2)
                v_cond_mid = model.forward_generate(z_mid, t_mid, class_labels)
                v_uncond_mid = model.forward_generate(z_mid, t_mid, null_labels)
                v2 = v_uncond_mid + guidance_scale * (v_cond_mid - v_uncond_mid)

                z = z + v2 * dt

        assert z.shape == (4, 64, 4, 4), f"Expected (4, 64, 4, 4), got {z.shape}"
        assert torch.isfinite(z).all(), "Latent generation should be finite"

    def test_latent_decode_to_image(self, device):
        vqvae = VQVAE().to(device)
        z = torch.randn(4, 64, 4, 4, device=device)
        with torch.no_grad():
            images = vqvae.decoder(z)
        assert images.shape == (4, 3, 16, 16)
        assert torch.isfinite(images).all()

    def test_full_pipeline(self, device):
        """Latent gen -> VQ-VAE decode -> clipped image."""
        latent_model = NeoUnifyLatentModel().to(device)
        vqvae = VQVAE().to(device)

        B = 2
        labels = torch.zeros(B, dtype=torch.long, device=device)

        # Quick 2-step generation
        z = torch.randn(B, 64, 4, 4, device=device)
        with torch.no_grad():
            t = torch.full((B,), 0.0, device=device)
            v = latent_model.forward_generate(z, t, labels)
            z = z + v * 0.5

            t = torch.full((B,), 0.5, device=device)
            v = latent_model.forward_generate(z, t, labels)
            z = z + v * 0.5

            images = vqvae.decoder(z)
            images = torch.clamp(images, 0.0, 1.0)

        assert images.shape == (2, 3, 16, 16)
        assert images.min() >= 0.0
        assert images.max() <= 1.0
