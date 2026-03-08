"""Train flow matching model on synthetic 16x16 images."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

from shared.data import generate_dataset, CLASSES
from phase_b.model import FlowMatchingTransformer
from shared.utils import save_image_grid, plot_losses, WEIGHTS_DIR, ensure_weights_dir


def train():
    print("=" * 60)
    print("Phase B: Training Flow Matching Transformer")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic dataset...")
    images, labels = generate_dataset(samples_per_class=1000)
    print(f"Dataset: {images.shape} images, {labels.shape} labels")

    # Model
    model = FlowMatchingTransformer(
        patch_size=4, image_size=16, channels=3,
        hidden_dim=128, num_heads=4, num_layers=4, num_classes=6,
    )
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Flow model parameters: {n_params:,}")

    # Optimizer
    num_epochs = 120
    batch_size = 64
    n_samples = images.shape[0]
    steps_per_epoch = n_samples // batch_size
    total_steps = num_epochs * steps_per_epoch

    warmup_steps = steps_per_epoch * 5
    lr_schedule = optim.schedulers.cosine_decay(3e-4, total_steps - warmup_steps)
    warmup = optim.schedulers.linear_schedule(1e-6, 3e-4, warmup_steps)
    schedule = optim.schedulers.join_schedules([warmup, lr_schedule], [warmup_steps])
    optimizer = optim.Adam(learning_rate=schedule)

    # Flow matching loss
    def loss_fn(model, x_0, class_labels):
        B = x_0.shape[0]
        t = mx.random.uniform(shape=(B,))
        noise = mx.random.normal(x_0.shape)
        t_expanded = t.reshape(B, 1, 1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * x_0
        v_target = x_0 - noise
        v_pred = model(x_t, t, class_labels)
        loss = mx.mean((v_pred - v_target) ** 2)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    losses = []
    t0 = time.time()

    for epoch in range(num_epochs):
        perm = mx.array(np.random.permutation(n_samples))
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch_images = images[idx]
            batch_labels = labels[idx]

            loss, grads = loss_and_grad(model, batch_images, batch_labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            losses.append(loss.item())

        avg_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % 15 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Plot losses
    plot_losses(losses, "flow_loss.png", title="Flow Matching Training Loss")

    # Save model weights
    ensure_weights_dir()
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(f"{WEIGHTS_DIR}/flow.npz", **weights)
    print(f"Saved: {WEIGHTS_DIR}/flow.npz")

    return model


if __name__ == "__main__":
    train()
