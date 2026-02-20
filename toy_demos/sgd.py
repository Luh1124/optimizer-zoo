"""
SGD demo: numpy hand-written vs PyTorch autograd, side by side.

Supports three SGD variants via the `sgd_type` variable:
    - "vanilla":   param -= lr * grad
    - "momentum":  v = mu * v + grad;  param -= lr * v
    - "nesterov":  v = mu * v + grad;  param -= lr * (grad + mu * v)

Architecture:  x -> Linear(1,8) -> ReLU -> Linear(8,8) -> ReLU -> Linear(8,1) -> y_pred
Loss:          MSE = mean((y_pred - y_true)^2)

Run:  python toy_demos/sgd.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
np.random.seed(42)

x_all = np.linspace(1.0, 10.0, 200).reshape(-1, 1)  # (200, 1)
y_all = x_all**2 + 2 * x_all + 1                     # (200, 1)

batch_size = 20
epochs = 500

in_ch = 1
hidden_ch = 8
out_ch = 1
lr = 1e-4
mu = 0.95
sgd_type = "nesterov"  # "vanilla", "momentum", or "nesterov"

# He initialization (shared by both implementations)
w1_init = np.random.randn(in_ch, hidden_ch).astype(np.float32) * np.sqrt(2.0 / in_ch)
b1_init = np.zeros((1, hidden_ch), dtype=np.float32)
w2_init = np.random.randn(hidden_ch, hidden_ch).astype(np.float32) * np.sqrt(2.0 / hidden_ch)
b2_init = np.zeros((1, hidden_ch), dtype=np.float32)
w3_init = np.random.randn(hidden_ch, out_ch).astype(np.float32) * np.sqrt(2.0 / hidden_ch)
b3_init = np.zeros((1, out_ch), dtype=np.float32)

shuffle_indices = np.array([np.random.permutation(200) for _ in range(epochs)])


# ═══════════════════════════════════════════════════════════════════════
# Network forward / backward (shared by all variants)
# ═══════════════════════════════════════════════════════════════════════
w1 = w1_init.copy()
b1 = b1_init.copy()
w2 = w2_init.copy()
b2 = b2_init.copy()
w3 = w3_init.copy()
b3 = b3_init.copy()


def np_forward(x):
    z1 = x @ w1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2 + b2
    a2 = np.maximum(0, z2)
    z3 = a2 @ w3 + b3
    return z3, (x, z1, a1, z2, a2)


def np_backward(y_pred, y_true, cache):
    x, z1, a1, z2, a2 = cache
    B = x.shape[0]

    dz3 = (2.0 / B) * (y_pred - y_true)

    dw3 = a2.T @ dz3
    db3 = dz3.sum(axis=0, keepdims=True)
    dz2 = (dz3 @ w3.T) * (z2 > 0)

    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    dz1 = (dz2 @ w2.T) * (z1 > 0)

    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    return dw1, db1, dw2, db2, dw3, db3


# ═══════════════════════════════════════════════════════════════════════
# SGD update functions
# ═══════════════════════════════════════════════════════════════════════
def sgd_update(params, grads, state, lr, mu=0.0, nesterov=False):
    """Unified SGD step covering vanilla, momentum, and nesterov.

    Vanilla (mu=0):
        param -= lr * grad

    Momentum (mu>0, nesterov=False):
        v = mu * v + grad          (first step: v = grad)
        param -= lr * v

    Nesterov (mu>0, nesterov=True):
        v = mu * v + grad          (first step: v = grad)
        param -= lr * (grad + mu * v)
    """
    if mu == 0:
        for i in range(len(params)):
            params[i] -= lr * grads[i]
        return

    if "v" not in state:
        state["v"] = [None] * len(params)

    for i in range(len(params)):
        if state["v"][i] is None:
            state["v"][i] = grads[i].copy()
        else:
            state["v"][i] = mu * state["v"][i] + grads[i]

        if nesterov:
            params[i] -= lr * (grads[i] + mu * state["v"][i])
        else:
            params[i] -= lr * state["v"][i]


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Numpy
# ═══════════════════════════════════════════════════════════════════════
sgd_state = {}

# Determine effective mu and nesterov flag from sgd_type
effective_mu = 0.0 if sgd_type == "vanilla" else mu
effective_nesterov = sgd_type == "nesterov"

print("=" * 60)
print(f"Part 1: Numpy ({sgd_type}, lr={lr}, mu={effective_mu})")
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
        epoch_loss += np.mean((y_pred - y_batch) ** 2)
        n_batches += 1

        grads = list(np_backward(y_pred, y_batch, cache))
        params = [w1, b1, w2, b2, w3, b3]
        sgd_update(params, grads, sgd_state, lr, effective_mu, effective_nesterov)

    avg_loss = epoch_loss / n_batches
    np_loss_history.append(avg_loss)

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# Part 2: PyTorch
# ═══════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden_ch)
        self.fc2 = nn.Linear(hidden_ch, hidden_ch)
        self.fc3 = nn.Linear(hidden_ch, out_ch)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model = MLP()
with torch.no_grad():
    model.fc1.weight.copy_(torch.from_numpy(w1_init.T))
    model.fc1.bias.copy_(torch.from_numpy(b1_init.flatten()))
    model.fc2.weight.copy_(torch.from_numpy(w2_init.T))
    model.fc2.bias.copy_(torch.from_numpy(b2_init.flatten()))
    model.fc3.weight.copy_(torch.from_numpy(w3_init.T))
    model.fc3.bias.copy_(torch.from_numpy(b3_init.flatten()))
model.to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(), lr=lr,
    momentum=effective_mu, nesterov=effective_nesterov,
)

x_all_t = torch.from_numpy(x_all.astype(np.float32)).to(DEVICE)
y_all_t = torch.from_numpy(y_all.astype(np.float32)).to(DEVICE)

print()
print("=" * 60)
print(f"Part 2: PyTorch ({sgd_type})")
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
    print("  PASS: numpy matches PyTorch!")
else:
    print("  MISMATCH: check your implementation.")

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(np_loss_history, label="Numpy", linewidth=2.5, alpha=0.8)
axes[0].plot(pt_loss_history, label="PyTorch", linewidth=1.5, linestyle="--", color="red")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title(f"Training Loss ({sgd_type})")
axes[0].set_yscale("log")
axes[0].legend()

axes[1].plot(diffs)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("|numpy - pytorch|")
axes[1].set_title("Loss Difference")
axes[1].set_yscale("log")

y_pred_all, _ = np_forward(x_all.astype(np.float32))
y_pred_all_t = model(x_all_t).detach().cpu().numpy()
axes[2].scatter(x_all, y_all, s=8, alpha=0.5, label="Ground truth")
axes[2].plot(x_all, y_pred_all_t, color="red", linewidth=3, alpha=0.6, label="torch")
axes[2].plot(x_all, y_pred_all, color="blue", linewidth=1.5, linestyle="--", label="numpy")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_title(f"Fit ({sgd_type})")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"toy_demos/sgd_{sgd_type}_result.png", dpi=150)
plt.show()
print(f"Saved to toy_demos/sgd_{sgd_type}_result.png")
