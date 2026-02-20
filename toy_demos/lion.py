"""
Adam demo: numpy hand-written vs PyTorch autograd, side by side.

Trains the SAME 3-layer MLP on y = x^2 + 2x + 1 with BOTH implementations,
using identical initialization, data order, and hyperparameters.
Verifies that hand-written Adam matches torch.optim.Adam exactly.

Architecture:  x -> Linear(1,8) -> ReLU -> Linear(8,8) -> ReLU -> Linear(8,1) -> y_pred
Loss:          MSE = mean((y_pred - y_true)^2)
Optimizer:     Adam (adaptive moment estimation)

Run:  python toy_demos/adam.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════
# Shared setup
# ═══════════════════════════════════════════════════════════════════════
np.random.seed(42)

x_all = np.linspace(1.0, 10.0, 200).reshape(-1, 1)  # (200, 1)
y_all = x_all**2 + 2 * x_all + 1                     # (200, 1)

batch_size = 20
epochs = 500

in_ch = 1
hidden_ch = 8
out_ch = 1

lr = 1e-2 
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
adam_type = "lion"
weight_decay = 0.01

# He initialization (shared by both implementations)
w1_init = np.random.randn(in_ch, hidden_ch).astype(np.float32) * np.sqrt(2.0 / in_ch)
b1_init = np.zeros((1, hidden_ch), dtype=np.float32)
w2_init = np.random.randn(hidden_ch, hidden_ch).astype(np.float32) * np.sqrt(2.0 / hidden_ch)
b2_init = np.zeros((1, hidden_ch), dtype=np.float32)
w3_init = np.random.randn(hidden_ch, out_ch).astype(np.float32) * np.sqrt(2.0 / hidden_ch)
b3_init = np.zeros((1, out_ch), dtype=np.float32)

# Pre-generate shuffle indices so both use the same data order
shuffle_indices = np.array([np.random.permutation(200) for _ in range(epochs)])

# ═══════════════════════════════════════════════════════════════════════
# Part 1: Numpy (hand-written forward + backward + Lion)
# ═══════════════════════════════════════════════════════════════════════

# Copy init so the two runs don't share mutable state
w1 = w1_init.copy()
b1 = b1_init.copy()
w2 = w2_init.copy()
b2 = b2_init.copy()
w3 = w3_init.copy()
b3 = b3_init.copy()


def np_forward(x):
    """Returns (y_pred, cache) where cache stores intermediates for backward."""
    z1 = x @ w1 + b1          # (B, 1) @ (1, 8) + (1, 8) = (B, 8)
    a1 = np.maximum(0, z1)    # ReLU                        (B, 8)
    z2 = a1 @ w2 + b2         # (B, 8) @ (8, 8) + (1, 8) = (B, 8)
    a2 = np.maximum(0, z2)    # ReLU                        (B, 8)
    z3 = a2 @ w3 + b3         # (B, 8) @ (8, 1) + (1, 1) = (B, 1)
    cache = (x, z1, a1, z2, a2)
    return z3, cache


def np_backward(y_pred, y_true, cache):
    """Compute gradients of MSE loss w.r.t. all parameters via chain rule.

    Chain rule walkthrough:
        L = mean((y_pred - y_true)^2)

        dL/dz3 = 2/B * (y_pred - y_true)                                 (B, 1)

        Layer 3: z3 = a2 @ w3 + b3
            dL/dw3 = a2.T @ dL/dz3                                       (8, 1)
            dL/db3 = sum(dL/dz3, axis=0)                                  (1, 1)
            dL/da2 = dL/dz3 @ w3.T                                       (B, 8)

        ReLU: a2 = max(0, z2)
            dL/dz2 = dL/da2 * (z2 > 0)                                   (B, 8)

        Layer 2: z2 = a1 @ w2 + b2
            dL/dw2 = a1.T @ dL/dz2                                       (8, 8)
            dL/db2 = sum(dL/dz2, axis=0)                                  (1, 8)
            dL/da1 = dL/dz2 @ w2.T                                       (B, 8)

        ReLU: a1 = max(0, z1)
            dL/dz1 = dL/da1 * (z1 > 0)                                   (B, 8)

        Layer 1: z1 = x @ w1 + b1
            dL/dw1 = x.T @ dL/dz1                                        (1, 8)
            dL/db1 = sum(dL/dz1, axis=0)                                  (1, 8)
    """
    x, z1, a1, z2, a2 = cache
    B = x.shape[0]

    dz3 = (2.0 / B) * (y_pred - y_true)

    dw3 = a2.T @ dz3
    db3 = dz3.sum(axis=0, keepdims=True)
    da2 = dz3 @ w3.T

    dz2 = da2 * (z2 > 0)

    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ w2.T

    dz1 = da1 * (z1 > 0)

    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    return dw1, db1, dw2, db2, dw3, db3

def lion_update(params, grads, state, lr, beta1, beta2, epsilon, weight_decay=0.0):
    """Adam optimizer: adaptive learning rate via first & second moment estimates.

    state dict holds:
        "m": first moment  (EMA of grad),      init to zeros
        "v": second moment (EMA of grad^2),     init to zeros
        "t": step counter (1-indexed for bias correction)
    """
    if "m" not in state:
        state["m"] = [np.zeros_like(p) for p in params]
        state["v"] = [np.zeros_like(p) for p in params]
        state["t"] = 0

    state["t"] += 1
    t = state["t"]

    if weight_decay > 0.0 and adam_type == "adaml2":
        for i in range(len(params)):
            grads[i] += weight_decay * params[i] # weight decay in the gradient

    for i in range(len(params)):
        state["m"][i] = beta1 * state["m"][i] + (1 - beta1) * grads[i]
        state["v"][i] = beta2 * state["v"][i] + (1 - beta2) * grads[i] ** 2

        m_hat = state["m"][i] / (1 - beta1 ** t)
        v_hat = state["v"][i] / (1 - beta2 ** t)

        if weight_decay > 0.0 and adam_type == "adamw":
            params[i] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * params[i])
        else:
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)


adam_state = {}

print("=" * 60)
print("Part 1: Numpy (hand-written backprop)")
print("=" * 60)

np_loss_history = []

for epoch in range(epochs):
    indices = shuffle_indices[epoch]
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, len(x_all), batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = x_all[batch_idx].astype(np.float32)
        y_batch = y_all[batch_idx].astype(np.float32)

        y_pred, cache = np_forward(x_batch)
        batch_loss = np.mean((y_pred - y_batch) ** 2)
        epoch_loss += batch_loss
        n_batches += 1

        dw1, db1, dw2, db2, dw3, db3 = np_backward(y_pred, y_batch, cache)

        params = [w1, b1, w2, b2, w3, b3]
        grads = [dw1, db1, dw2, db2, dw3, db3]
        adam_update(params, grads, adam_state, lr, beta1, beta2, epsilon, weight_decay, adam_type)
    
    avg_loss = epoch_loss / n_batches
    np_loss_history.append(avg_loss)

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# Part 2: PyTorch (autograd + torch.optim.SGD)
# ═══════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=8, out_ch=1):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden_ch)
        self.fc2 = nn.Linear(hidden_ch, hidden_ch)
        self.fc3 = nn.Linear(hidden_ch, out_ch)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model = MLP(in_ch=in_ch, hidden_ch=hidden_ch, out_ch=out_ch)

# Load the SAME initial weights (nn.Linear stores weight as (out, in))
with torch.no_grad():
    model.fc1.weight.copy_(torch.from_numpy(w1_init.T))
    model.fc1.bias.copy_(torch.from_numpy(b1_init.flatten()))
    model.fc2.weight.copy_(torch.from_numpy(w2_init.T))
    model.fc2.bias.copy_(torch.from_numpy(b2_init.flatten()))
    model.fc3.weight.copy_(torch.from_numpy(w3_init.T))
    model.fc3.bias.copy_(torch.from_numpy(b3_init.flatten()))

model.to(DEVICE)

if adam_type == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
elif adam_type == "adaml2":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
elif adam_type == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
else:
    raise ValueError(f"Unknown Adam type: {adam_type}")


x_all_t = torch.from_numpy(x_all.astype(np.float32)).to(DEVICE)
y_all_t = torch.from_numpy(y_all.astype(np.float32)).to(DEVICE)

print()
print("=" * 60)
print("Part 2: PyTorch (autograd)")
print("=" * 60)

pt_loss_history = []

for epoch in range(epochs):
    indices = shuffle_indices[epoch]
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, len(x_all), batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = x_all_t[batch_idx]
        y_batch = y_all_t[batch_idx]

        y_pred = model(x_batch)
        batch_loss = torch.mean((y_pred - y_batch) ** 2)
        epoch_loss += batch_loss.item()
        n_batches += 1

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / n_batches
    pt_loss_history.append(avg_loss)

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Comparison (max difference per epoch)")
print("=" * 60)
diffs = [abs(a - b) for a, b in zip(np_loss_history, pt_loss_history)]
print(f"  Max diff:  {max(diffs):.2e}")
print(f"  Mean diff: {np.mean(diffs):.2e}")
if max(diffs) < 1.0:
    print("  PASS: numpy backprop matches PyTorch autograd!")
else:
    print("  MISMATCH: check your backward() implementation.")

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss curves overlay (offset PyTorch slightly so both are visible)
axes[0].plot(np_loss_history, label="Numpy", linewidth=2.5, alpha=0.8)
axes[0].plot(pt_loss_history, label="PyTorch", linewidth=1.5, linestyle="--", color="red")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training Loss (two lines overlap = correct!)")
axes[0].set_yscale("log")
axes[0].legend()

# Difference
axes[1].plot(diffs)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("|numpy_loss - pytorch_loss|")
axes[1].set_title("Loss Difference (should be ~0)")
axes[1].set_yscale("log")

# Fit quality (numpy version)
y_pred_all, _ = np_forward(x_all.astype(np.float32))
y_pred_all_t = model(x_all_t).detach().cpu().numpy()
axes[2].scatter(x_all, y_all, s=8, alpha=0.5, label="Ground truth")
axes[2].plot(x_all, y_pred_all_t, color="red", linewidth=3, alpha=0.6, label="torch prediction")
axes[2].plot(x_all, y_pred_all,   color="blue", linewidth=1.5, linestyle="--", label="numpy prediction")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_title("Fit: y = x² + 2x + 1 (two lines overlap = correct!)")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"toy_demos/{adam_type}_result.png", dpi=150)
plt.show()
print(f"Saved to toy_demos/{adam_type}_result.png")
