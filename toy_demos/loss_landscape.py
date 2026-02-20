"""
Loss landscape visualization comparing multiple optimizers.

Method (Li et al., 2018 "Visualizing the Loss Landscape of Neural Nets"):
    1. Train with multiple optimizers, all from the same init
    2. Use PCA on the combined trajectories to find the 2 directions
       that capture the most variance in parameter space
    3. Evaluate loss on a grid in this 2D plane
    4. Overlay all trajectories on the same landscape

Run:  python toy_demos/loss_landscape.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Network & data setup
# ═══════════════════════════════════════════════════════════════════════
np.random.seed(42)

x_all = np.linspace(1.0, 10.0, 200).reshape(-1, 1).astype(np.float32)
y_all = (x_all**2 + 2 * x_all + 1).astype(np.float32)

in_ch, hidden_ch, out_ch = 1, 8, 1
epochs = 500
batch_size = 20

w1_init = np.random.randn(in_ch, hidden_ch).astype(np.float32) * np.sqrt(2.0 / in_ch)
b1_init = np.zeros((1, hidden_ch), dtype=np.float32)
w2_init = np.random.randn(hidden_ch, hidden_ch).astype(np.float32) * np.sqrt(2.0 / hidden_ch)
b2_init = np.zeros((1, hidden_ch), dtype=np.float32)
w3_init = np.random.randn(hidden_ch, out_ch).astype(np.float32) * np.sqrt(2.0 / hidden_ch)
b3_init = np.zeros((1, out_ch), dtype=np.float32)

INIT_PARAMS = [w1_init, b1_init, w2_init, b2_init, w3_init, b3_init]

shuffle_indices = np.array([np.random.permutation(200) for _ in range(epochs)])

SHAPES = [
    (in_ch, hidden_ch), (1, hidden_ch),
    (hidden_ch, hidden_ch), (1, hidden_ch),
    (hidden_ch, out_ch), (1, out_ch),
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def flatten(param_list):
    return np.concatenate([p.ravel() for p in param_list])


def unflatten(vec):
    params, offset = [], 0
    for shape in SHAPES:
        size = int(np.prod(shape))
        params.append(vec[offset : offset + size].reshape(shape))
        offset += size
    return params


def compute_loss(params):
    w1, b1, w2, b2, w3, b3 = params
    a1 = np.maximum(0, x_all @ w1 + b1)
    a2 = np.maximum(0, a1 @ w2 + b2)
    y_pred = a2 @ w3 + b3
    return np.mean((y_pred - y_all) ** 2)


def compute_grads(x, y_true, params):
    w1, b1, w2, b2, w3, b3 = params
    B = x.shape[0]
    z1 = x @ w1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ w2 + b2
    a2 = np.maximum(0, z2)
    y_pred = a2 @ w3 + b3

    dz3 = (2.0 / B) * (y_pred - y_true)
    dw3 = a2.T @ dz3
    db3 = dz3.sum(axis=0, keepdims=True)
    dz2 = (dz3 @ w3.T) * (z2 > 0)
    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    dz1 = (dz2 @ w2.T) * (z1 > 0)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return [dw1, db1, dw2, db2, dw3, db3]


# ═══════════════════════════════════════════════════════════════════════
# Optimizer definitions (each is a callable that returns trajectory)
# ═══════════════════════════════════════════════════════════════════════
def train(optimizer_step, label, **opt_kwargs):
    """Train with a given optimizer step function, return trajectory."""
    params = [p.copy() for p in INIT_PARAMS]
    state = {}
    trajectory = [flatten(params).copy()]

    for epoch in range(epochs):
        indices = shuffle_indices[epoch]
        for start in range(0, len(x_all), batch_size):
            batch_idx = indices[start : start + batch_size]
            grads = compute_grads(x_all[batch_idx], y_all[batch_idx], params)
            optimizer_step(params, grads, state, **opt_kwargs)

        if epoch % 10 == 0 or epoch == epochs - 1:
            trajectory.append(flatten(params).copy())

    final_loss = compute_loss(params)
    print(f"  {label:25s} | Final loss: {final_loss:.4f}")
    return trajectory


def step_vanilla_sgd(params, grads, state, lr=1e-4):
    for i in range(len(params)):
        params[i] -= lr * grads[i]


def step_sgd_momentum(params, grads, state, lr=1e-4, mu=0.9):
    if "v" not in state:
        state["v"] = [None] * len(params)
    for i in range(len(params)):
        if state["v"][i] is None:
            state["v"][i] = grads[i].copy()
        else:
            state["v"][i] = mu * state["v"][i] + grads[i]
        params[i] -= lr * state["v"][i]


def step_sgd_nesterov(params, grads, state, lr=1e-4, mu=0.9):
    if "v" not in state:
        state["v"] = [None] * len(params)
    for i in range(len(params)):
        if state["v"][i] is None:
            state["v"][i] = grads[i].copy()
        else:
            state["v"][i] = mu * state["v"][i] + grads[i]
        params[i] -= lr * (grads[i] + mu * state["v"][i])


# ═══════════════════════════════════════════════════════════════════════
# Run all optimizers
# ═══════════════════════════════════════════════════════════════════════
OPTIMIZERS = [
    ("Vanilla SGD",          step_vanilla_sgd,  dict(lr=1e-4)),
    ("SGD+Momentum (0.9)",   step_sgd_momentum, dict(lr=1e-4, mu=0.9)),
    ("SGD+Momentum (0.95)",  step_sgd_momentum, dict(lr=1e-4, mu=0.95)),
    ("SGD+Nesterov (0.95)",  step_sgd_nesterov, dict(lr=1e-4, mu=0.95)),
]

COLORS = ["gray", "dodgerblue", "red", "lime"]
MARKERS = ["x", "o", "s", "D"]

print("Training with each optimizer...")
trajectories = []
for label, step_fn, kwargs in OPTIMIZERS:
    traj = train(step_fn, label, **kwargs)
    trajectories.append(traj)

# ═══════════════════════════════════════════════════════════════════════
# Build 2D landscape using PCA on combined trajectories
# ═══════════════════════════════════════════════════════════════════════
print("Computing loss landscape...")

theta_init = flatten(INIT_PARAMS)

# Collect all trajectory points for PCA
all_points = []
for traj in trajectories:
    all_points.extend(traj)
all_points = np.array(all_points)

# PCA: find the 2 directions with most variance
center = all_points.mean(axis=0)
centered = all_points - center
cov = centered.T @ centered
eigenvalues, eigenvectors = np.linalg.eigh(cov)
d1 = eigenvectors[:, -1]  # largest variance
d2 = eigenvectors[:, -2]  # second largest

# Project all trajectories
def project(traj):
    alphas = [np.dot(t - center, d1) for t in traj]
    betas = [np.dot(t - center, d2) for t in traj]
    return alphas, betas

# Determine grid range from projected trajectories
all_alphas, all_betas = [], []
for traj in trajectories:
    a, b = project(traj)
    all_alphas.extend(a)
    all_betas.extend(b)

margin = 0.3
a_min, a_max = min(all_alphas), max(all_alphas)
b_min, b_max = min(all_betas), max(all_betas)
a_pad = (a_max - a_min) * margin
b_pad = (b_max - b_min) * margin

alpha_range = np.linspace(a_min - a_pad, a_max + a_pad, 71)
beta_range = np.linspace(b_min - b_pad, b_max + b_pad, 71)

loss_grid = np.zeros((len(beta_range), len(alpha_range)))
for i, beta in enumerate(beta_range):
    for j, alpha in enumerate(alpha_range):
        theta = center + alpha * d1 + beta * d2
        loss_grid[i, j] = compute_loss(unflatten(theta))

# ═══════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 6))

A, B = np.meshgrid(alpha_range, beta_range)
log_loss = np.log10(np.clip(loss_grid, 1e-2, None))

# ── Contour plot ──
ax1 = fig.add_subplot(121)
cf = ax1.contourf(A, B, log_loss, levels=30, cmap="viridis")
ax1.contour(A, B, log_loss, levels=15, colors="white", linewidths=0.3, alpha=0.5)

for idx, (label, _, _) in enumerate(OPTIMIZERS):
    alphas, betas = project(trajectories[idx])
    ax1.plot(alphas, betas, color=COLORS[idx], linewidth=1.5, alpha=0.9, label=label)
    ax1.plot(alphas[0], betas[0], "wo", markersize=6, zorder=5)
    ax1.plot(alphas[-1], betas[-1], marker=MARKERS[idx], color=COLORS[idx],
             markersize=8, zorder=5, markeredgecolor="white", markeredgewidth=0.5)

ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_title("Loss Landscape with Optimizer Trajectories")
ax1.legend(loc="upper right", fontsize=7)
plt.colorbar(cf, ax=ax1, label="log10(MSE)")

# ── 3D surface ──
ax3d = fig.add_subplot(122, projection="3d")
ax3d.plot_surface(A, B, log_loss, cmap="viridis", alpha=0.7, edgecolor="none")

for idx, (label, _, _) in enumerate(OPTIMIZERS):
    alphas, betas = project(trajectories[idx])
    losses = [np.log10(max(compute_loss(unflatten(t)), 1e-2)) for t in trajectories[idx]]
    ax3d.plot(alphas, betas, losses, color=COLORS[idx], linewidth=1.5, alpha=0.9, label=label)

ax3d.set_xlabel("PC1")
ax3d.set_ylabel("PC2")
ax3d.set_zlabel("log10(MSE)")
ax3d.set_title("Loss Surface")
ax3d.view_init(elev=35, azim=-60)
ax3d.legend(loc="upper right", fontsize=6)

plt.tight_layout()
plt.savefig("toy_demos/loss_landscape.png", dpi=150)
plt.show()
print("Saved to toy_demos/loss_landscape.png")
