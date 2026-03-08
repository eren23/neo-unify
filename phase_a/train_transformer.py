"""Stage 2: Train class-conditional autoregressive transformer on encoded tokens."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

from phase_a.transformer import ImageGPT
from shared.utils import plot_losses, WEIGHTS_DIR, ensure_weights_dir


def train():
    print("=" * 60)
    print("Phase A - Stage 2: Training Autoregressive Transformer")
    print("=" * 60)

    # Load encoded dataset
    data = np.load(f"{WEIGHTS_DIR}/encoded_dataset.npz")
    indices = mx.array(data["indices"].astype(np.int32))
    labels = mx.array(data["labels"].astype(np.int32))
    print(f"Loaded: {indices.shape[0]} sequences of {indices.shape[1]} tokens")
    print(f"Token range: [{indices.min().item()}, {indices.max().item()}]")

    # Model
    model = ImageGPT(
        vocab_size=256, num_classes=6, seq_len=16,
        hidden_dim=128, num_heads=4, num_layers=4,
    )
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"ImageGPT parameters: {n_params:,}")

    # Optimizer with warmup + cosine decay
    num_epochs = 150
    batch_size = 128
    n_samples = indices.shape[0]
    steps_per_epoch = n_samples // batch_size
    total_steps = num_epochs * steps_per_epoch

    warmup_steps = steps_per_epoch * 5
    lr_schedule = optim.schedulers.cosine_decay(3e-4, total_steps - warmup_steps)
    warmup = optim.schedulers.linear_schedule(1e-6, 3e-4, warmup_steps)
    schedule = optim.schedulers.join_schedules([warmup, lr_schedule], [warmup_steps])
    optimizer = optim.Adam(learning_rate=schedule)

    # Loss function
    def loss_fn(model, tokens, class_labels):
        logits = model(tokens, class_labels)
        targets = tokens
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        loss = -mx.mean(log_probs[mx.arange(B * T), targets_flat])
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
            batch_tokens = indices[idx]
            batch_labels = labels[idx]

            loss, grads = loss_and_grad(model, batch_tokens, batch_labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)  # noqa: S307

            epoch_loss += loss.item()
            losses.append(loss.item())

        avg_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            bpt = avg_loss / np.log(2)
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"BPT: {bpt:.2f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # Plot losses
    plot_losses(losses, "transformer_loss.png", title="AR Transformer Training Loss")

    # Save model weights
    ensure_weights_dir()
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.savez(f"{WEIGHTS_DIR}/transformer.npz", **weights)
    print(f"Saved: {WEIGHTS_DIR}/transformer.npz")

    return model


if __name__ == "__main__":
    train()
