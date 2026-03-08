"""Tests for training logic (smoke tests)."""

import torch
import torch.nn.functional as F
import pytest
import math

from neo_unify_torch.phase_c.model import NeoUnifyModel as PhaseC
from neo_unify_torch.exp_cfg.model import NeoUnifyModel as ExpCFG
from neo_unify_torch.exp_latent.model import NeoUnifyLatentModel
from neo_unify_torch.vqvae.model import VQVAE
from neo_unify_torch.shared.data import generate_dataset


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_dataset(device):
    images, labels = generate_dataset(samples_per_class=10, seed=42)
    return images.to(device), labels.to(device)


class TestPhaseCTraining:
    def test_one_step_understand(self, device, small_dataset):
        images, labels = small_dataset
        model = PhaseC().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        logits = model.forward_understand(images[:8])
        loss = F.cross_entropy(logits, labels[:8])

        assert torch.isfinite(loss), "Loss should be finite"
        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Gradients should be finite"

        optimizer.step()

    def test_one_step_generate(self, device, small_dataset):
        images, labels = small_dataset
        model = PhaseC().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B = 8
        batch = images[:B]
        batch_labels = labels[:B]
        t = torch.rand(B, device=device)
        noise = torch.randn_like(batch)
        t_exp = t.reshape(B, 1, 1, 1)
        x_t = (1 - t_exp) * noise + t_exp * batch
        v_target = batch - noise
        v_pred = model.forward_generate(x_t, t, batch_labels)
        loss = F.mse_loss(v_pred, v_target)

        assert torch.isfinite(loss), "Loss should be finite"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_loss_decreases(self, device, small_dataset):
        images, labels = small_dataset
        model = PhaseC().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            logits = model.forward_understand(images[:16])
            loss = F.cross_entropy(logits, labels[:16])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestExpCFGTraining:
    def test_one_step_joint(self, device, small_dataset):
        images, labels = small_dataset
        model = ExpCFG().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B = 8
        batch = images[:B]
        batch_labels = labels[:B]

        # Joint loss
        logits = model.forward_understand(batch)
        und_loss = F.cross_entropy(logits, batch_labels)

        t = torch.rand(B, device=device)
        noise = torch.randn_like(batch)
        t_exp = t.reshape(B, 1, 1, 1)
        x_t = (1 - t_exp) * noise + t_exp * batch
        v_target = batch - noise
        v_pred = model.forward_generate(x_t, t, batch_labels)
        gen_loss = F.mse_loss(v_pred, v_target)

        recon = model.forward_reconstruct(batch)
        recon_loss = F.mse_loss(recon, batch)

        total = und_loss + 3.0 * gen_loss + 1.0 * recon_loss

        assert torch.isfinite(total), "Joint loss should be finite"
        optimizer.zero_grad()
        total.backward()
        optimizer.step()


class TestExpLatentTraining:
    def test_one_step(self, device, small_dataset):
        images, labels = small_dataset
        model = NeoUnifyLatentModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B = 8
        batch = images[:B]
        batch_labels = labels[:B]

        # Simulate latent space (random for test)
        batch_z = torch.randn(B, 64, 4, 4, device=device)

        logits = model.forward_understand(batch)
        und_loss = F.cross_entropy(logits, batch_labels)

        t = torch.rand(B, device=device)
        z_noise = torch.randn_like(batch_z)
        t_exp = t.reshape(B, 1, 1, 1)
        z_t = (1 - t_exp) * z_noise + t_exp * batch_z
        v_target = batch_z - z_noise
        v_pred = model.forward_generate(z_t, t, batch_labels)
        gen_loss = F.mse_loss(v_pred, v_target)

        recon = model.forward_reconstruct(batch)
        recon_loss = F.mse_loss(recon, batch)

        total = und_loss + 3.0 * gen_loss + 1.0 * recon_loss

        assert torch.isfinite(total), "Joint loss should be finite"
        optimizer.zero_grad()
        total.backward()
        optimizer.step()


class TestVQVAETraining:
    def test_one_step(self, device, small_dataset):
        images, _ = small_dataset
        model = VQVAE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = images[:8]
        x_recon, indices, vq_loss = model(batch)
        recon_loss = F.mse_loss(x_recon, batch)
        total = recon_loss + vq_loss

        assert torch.isfinite(total), "Loss should be finite"
        optimizer.zero_grad()
        total.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Gradients should be finite"

        optimizer.step()


class TestLRSchedule:
    def test_warmup_increases(self):
        from neo_unify_torch.phase_c.train import make_lr_lambda
        lr_fn = make_lr_lambda(warmup_steps=100, total_steps=1000)
        lrs = [lr_fn(i) for i in range(100)]
        # LR should generally increase during warmup
        assert lrs[50] > lrs[0], "LR should increase during warmup"
        assert lrs[99] > lrs[50], "LR should continue increasing"

    def test_cosine_decreases(self):
        from neo_unify_torch.phase_c.train import make_lr_lambda
        lr_fn = make_lr_lambda(warmup_steps=100, total_steps=1000)
        # After warmup, LR should decrease
        lr_warmup_end = lr_fn(100)
        lr_late = lr_fn(900)
        assert lr_late < lr_warmup_end, "LR should decrease after warmup"
