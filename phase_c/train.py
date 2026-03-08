"""Train Mini Neo-Unify with alternating understanding/generation mini-batches."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

from shared.data import generate_dataset, CLASSES
from phase_c.model import NeoUnifyModel
from shared.utils import (
    save_image_grid, plot_losses, plot_dual_losses,
    WEIGHTS_DIR, ensure_weights_dir,
)


def train():
    print("=" * 60)
    print("Phase C: Training Mini Neo-Unify (MoT)")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic dataset...")
    images, labels = generate_dataset(samples_per_class=3000)
    print(f"Dataset: {images.shape} images, {labels.shape} labels")

    # Model
    model = NeoUnifyModel(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=6, num_classes=6,
    )
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Neo-Unify parameters: {n_params:,}")

    # Optimizer with warmup + cosine decay
    num_epochs = 200
    batch_size = 64
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size
    total_steps = num_epochs * steps_per_epoch * 2  # 2 updates per step

    warmup_steps = steps_per_epoch * 5 * 2
    lr_schedule = optim.schedulers.cosine_decay(3e-4, total_steps - warmup_steps)
    warmup = optim.schedulers.linear_schedule(1e-6, 3e-4, warmup_steps)
    schedule = optim.schedulers.join_schedules([warmup, lr_schedule], [warmup_steps])
    optimizer = optim.Adam(learning_rate=schedule)

    # Understanding loss: cross-entropy
    def und_loss_fn(model, x, class_labels):
        logits = model.forward_understand(x)
        B = logits.shape[0]
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        loss = -mx.mean(log_probs[mx.arange(B), class_labels])
        return loss

    # Generation loss: flow matching MSE
    def gen_loss_fn(model, x_0, class_labels):
        B = x_0.shape[0]
        t = mx.random.uniform(shape=(B,))
        noise = mx.random.normal(x_0.shape)
        t_expanded = t.reshape(B, 1, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * x_0
        v_target = x_0 - noise
        v_pred = model.forward_generate(x_t, t, class_labels)
        loss = mx.mean((v_pred - v_target) ** 2)
        return loss

    und_loss_and_grad = nn.value_and_grad(model, und_loss_fn)
    gen_loss_and_grad = nn.value_and_grad(model, gen_loss_fn)

    # Training: alternating mini-batch updates
    und_losses = []
    gen_losses = []
    t0 = time.time()

    for epoch in range(num_epochs):
        perm = mx.array(np.random.permutation(n_samples))
        epoch_und_loss = 0.0
        epoch_gen_loss = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch_images = images[idx]
            batch_labels = labels[idx]

            # 1. Understanding update
            und_loss, und_grads = und_loss_and_grad(model, batch_images, batch_labels)
            optimizer.update(model, und_grads)
            # Force lazy computation (MLX)
            mx.eval(model.parameters(), optimizer.state)  # noqa: S307

            # 2. Generation update
            gen_loss, gen_grads = gen_loss_and_grad(model, batch_images, batch_labels)
            optimizer.update(model, gen_grads)
            # Force lazy computation (MLX)
            mx.eval(model.parameters(), optimizer.state)  # noqa: S307

            epoch_und_loss += und_loss.item()
            epoch_gen_loss += gen_loss.item()
            und_losses.append(und_loss.item())
            gen_losses.append(gen_loss.item())

        avg_und = epoch_und_loss / steps_per_epoch
        avg_gen = epoch_gen_loss / steps_per_epoch

        if (epoch + 1) % 15 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Und CE: {avg_und:.4f} | "
                f"Gen MSE: {avg_gen:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Plot losses
    plot_dual_losses(
        und_losses, gen_losses,
        "Understanding CE", "Generation MSE",
        "neo_unify_losses.png",
        title="Neo-Unify Training Losses",
    )
    plot_losses(und_losses, "neo_unify_und_loss.png", title="Neo-Unify Understanding Loss")
    plot_losses(gen_losses, "neo_unify_gen_loss.png", title="Neo-Unify Generation Loss")

    # Save model weights
    ensure_weights_dir()
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(f"{WEIGHTS_DIR}/neo_unify.npz", **weights)
    print(f"Saved: {WEIGHTS_DIR}/neo_unify.npz")

    return model


if __name__ == "__main__":
    train()
