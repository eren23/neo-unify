"""Tests for model architectures."""

import torch
import pytest
from neo_unify_torch.phase_c.model import NeoUnifyModel as PhaseC
from neo_unify_torch.exp_cfg.model import NeoUnifyModel as ExpCFG
from neo_unify_torch.exp_latent.model import NeoUnifyLatentModel
from neo_unify_torch.vqvae.model import VQVAE


@pytest.fixture
def device():
    return torch.device("cpu")


class TestPhaseC:
    def test_understand_shape(self, device):
        model = PhaseC().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        logits = model.forward_understand(x)
        assert logits.shape == (4, 6), f"Expected (4, 6), got {logits.shape}"

    def test_generate_shape(self, device):
        model = PhaseC().to(device)
        x_t = torch.randn(4, 3, 16, 16, device=device)
        t = torch.rand(4, device=device)
        labels = torch.tensor([0, 1, 2, 3], device=device)
        v = model.forward_generate(x_t, t, labels)
        assert v.shape == (4, 3, 16, 16), f"Expected (4, 3, 16, 16), got {v.shape}"

    def test_patchify_roundtrip(self, device):
        model = PhaseC().to(device)
        x = torch.rand(2, 3, 16, 16, device=device)
        patches = model.patchify(x)
        assert patches.shape == (2, 16, 48), f"Expected (2, 16, 48), got {patches.shape}"
        x_recon = model.unpatchify(patches)
        assert x_recon.shape == x.shape
        assert torch.allclose(x, x_recon, atol=1e-6), "Patchify/unpatchify roundtrip failed"

    def test_param_count(self, device):
        model = PhaseC().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be ~2.4M (allow 10% tolerance)
        assert 2_100_000 < n_params < 2_700_000, f"Expected ~2.4M params, got {n_params:,}"


class TestExpCFG:
    def test_understand_shape(self, device):
        model = ExpCFG().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        logits = model.forward_understand(x)
        assert logits.shape == (4, 6)

    def test_generate_shape(self, device):
        model = ExpCFG().to(device)
        x_t = torch.randn(4, 3, 16, 16, device=device)
        t = torch.rand(4, device=device)
        labels = torch.tensor([0, 1, 2, 3], device=device)
        v = model.forward_generate(x_t, t, labels)
        assert v.shape == (4, 3, 16, 16)

    def test_reconstruct_shape(self, device):
        model = ExpCFG().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        recon = model.forward_reconstruct(x)
        assert recon.shape == (4, 3, 16, 16)
        assert recon.min() >= 0.0, "Reconstruction should be >= 0 (sigmoid output)"
        assert recon.max() <= 1.0, "Reconstruction should be <= 1 (sigmoid output)"

    def test_null_class(self, device):
        model = ExpCFG().to(device)
        x_t = torch.randn(4, 3, 16, 16, device=device)
        t = torch.rand(4, device=device)
        null_labels = torch.full((4,), 6, dtype=torch.long, device=device)
        v = model.forward_generate(x_t, t, null_labels)
        assert v.shape == (4, 3, 16, 16), "Should handle null class (index 6)"

    def test_param_count(self, device):
        model = ExpCFG().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        assert 2_100_000 < n_params < 2_700_000, f"Expected ~2.4M params, got {n_params:,}"


class TestExpLatent:
    def test_understand_shape(self, device):
        model = NeoUnifyLatentModel().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        logits = model.forward_understand(x)
        assert logits.shape == (4, 6)

    def test_generate_shape(self, device):
        model = NeoUnifyLatentModel().to(device)
        z_t = torch.randn(4, 64, 4, 4, device=device)
        t = torch.rand(4, device=device)
        labels = torch.tensor([0, 1, 2, 3], device=device)
        v = model.forward_generate(z_t, t, labels)
        assert v.shape == (4, 64, 4, 4), f"Expected (4, 64, 4, 4), got {v.shape}"

    def test_reconstruct_shape(self, device):
        model = NeoUnifyLatentModel().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        recon = model.forward_reconstruct(x)
        assert recon.shape == (4, 3, 16, 16)

    def test_patchify_latent_roundtrip(self, device):
        model = NeoUnifyLatentModel().to(device)
        z = torch.rand(2, 64, 4, 4, device=device)
        patches = model.patchify_latent(z)
        assert patches.shape == (2, 16, 64)
        z_recon = model.unpatchify_latent(patches)
        assert z_recon.shape == z.shape
        assert torch.allclose(z, z_recon, atol=1e-6), "Latent patchify roundtrip failed"

    def test_patchify_pixels_roundtrip(self, device):
        model = NeoUnifyLatentModel().to(device)
        x = torch.rand(2, 3, 16, 16, device=device)
        patches = model.patchify_pixels(x)
        assert patches.shape == (2, 16, 48)
        x_recon = model.unpatchify_pixels(patches)
        assert torch.allclose(x, x_recon, atol=1e-6), "Pixel patchify roundtrip failed"

    def test_param_count(self, device):
        model = NeoUnifyLatentModel().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        assert 2_100_000 < n_params < 2_700_000, f"Expected ~2.4M params, got {n_params:,}"


class TestVQVAE:
    def test_forward_shape(self, device):
        model = VQVAE().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        x_recon, indices, vq_loss = model(x)
        assert x_recon.shape == (4, 3, 16, 16), f"Expected (4, 3, 16, 16), got {x_recon.shape}"
        assert indices.shape == (4, 4, 4), f"Expected (4, 4, 4), got {indices.shape}"
        assert vq_loss.ndim == 0, "VQ loss should be scalar"

    def test_encode_decode(self, device):
        model = VQVAE().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        indices = model.encode(x)
        assert indices.shape == (4, 4, 4)
        decoded = model.decode(indices)
        assert decoded.shape == (4, 3, 16, 16)

    def test_output_range(self, device):
        model = VQVAE().to(device)
        x = torch.rand(4, 3, 16, 16, device=device)
        x_recon, _, _ = model(x)
        assert x_recon.min() >= 0.0, "Recon should be >= 0 (sigmoid)"
        assert x_recon.max() <= 1.0, "Recon should be <= 1 (sigmoid)"

    def test_param_count(self, device):
        model = VQVAE().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        # VQ-VAE is smaller (~187K)
        assert 150_000 < n_params < 250_000, f"Expected ~187K params, got {n_params:,}"

    def test_codebook_utilization(self, device):
        model = VQVAE().to(device)
        x = torch.rand(32, 3, 16, 16, device=device)
        with torch.no_grad():
            indices = model.encode(x)
        used, total = model.codebook_utilization(indices)
        assert used > 0, "Should use at least some codebook entries"
        assert total == 256, "Total should be 256"
