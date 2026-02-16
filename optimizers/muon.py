"""
Muon Optimizer
==============
Paper: "Muon is Scalable for LLM Training" (Jordan et al., 2025)
       https://arxiv.org/abs/2502.16982
Blog:  https://kellerjordan.github.io/posts/muon/
Also:  "Polar Express" improved orthogonalization (2025)
       https://arxiv.org/abs/2505.16932

Core idea:
    For weight matrices, replace the gradient with its nearest orthogonal
    matrix (in Frobenius norm) before applying momentum. This is the
    "matrix sign" or "polar factor" of the gradient.

    Intuitively: instead of updating weights in the raw gradient direction,
    project the update onto the Stiefel manifold of orthogonal matrices.
    This produces updates with balanced singular values, preventing any
    single direction from dominating.

    The polar factor is computed efficiently via Newton-Schulz iteration
    (no SVD needed), making it practical for large matrices.

Newton-Schulz iteration (5 steps):
    X_0 = G / ||G||_F                          # normalize
    For each step with coefficients (a, b, c):
        A = X @ X^T
        B = b * A + c * A @ A
        X = a * X + B @ X
    Returns X ~ U where G = U S V^T (polar decomposition)

Update rule:
    buf_t = mu * buf_{t-1} + grad_t                    # momentum
    update_t = newton_schulz(grad_t + mu * buf_t)       # Nesterov + orthogonalize
    param_t = param_{t-1} - adjusted_lr * update_t

LR adjustment (rms_norm, from Kimi K2):
    adjusted_lr = lr * 0.2 * sqrt(max(fan_out, fan_in))
    This ensures the element-wise RMS of the update is approximately constant
    regardless of layer size.

Typical usage:
    Muon is applied to 2D weight matrices only. For biases, embeddings,
    and normalization layers, use AdamW as a fallback (not part of this
    implementation -- handle in your training script).

Hyperparameters:
    lr:           Learning rate (typical: 0.01 - 0.02)
    mu:           Momentum factor (typical: 0.95)
    nesterov:     Use Nesterov momentum (typical: True)
    weight_decay: Decoupled weight decay (typical: 0.0 - 0.01)
    ns_steps:     Number of Newton-Schulz iterations (typical: 5)
"""

import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


# Newton-Schulz coefficients (pre-computed for optimal convergence)
NS_COEFFICIENTS = [
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
    (3.4445, -4.7750, 2.0315),
]

# Better coefficients from the Muon paper (quintic polynomial)
NS_COEFFICIENTS_QUINTIC = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Compute the polar factor of G via Newton-Schulz iteration.

    Given G = U S V^T (SVD), returns U V^T (the nearest orthogonal matrix).

    Implementation guide:
        1. Cast to bfloat16 for speed.
        2. If tall matrix (rows > cols), transpose first (work with wide matrix).
        3. Normalize: X = G / ||G||_F
        4. For each of ``steps`` iterations with coefficients (a, b, c):
           A = X @ X^T
           B = b * A + c * (A @ A)
           X = a * X + B @ X
        5. If we transposed in step 2, transpose back.
        6. Return X.

    Args:
        G: Input matrix (2D tensor).
        steps: Number of Newton-Schulz iterations.
        eps: Small constant for numerical stability.

    Returns:
        Polar factor of G (same shape as G).
    """
    # TODO: implement Newton-Schulz iteration
    raise NotImplementedError(
        "Implement Newton-Schulz iteration! See the docstring above."
    )


class Muon(Optimizer):
    """Muon optimizer (orthogonalized SGD with momentum).

    This is the most exciting recent optimizer. Apply it to 2D weight
    matrices; use AdamW for other parameters in your training script.

    Args:
        params: Iterable of 2D parameters.
        lr: Learning rate.
        mu: Momentum factor.
        nesterov: If True, use Nesterov momentum.
        weight_decay: Decoupled weight decay.
        ns_steps: Number of Newton-Schulz iterations.
        adjust_lr: LR adjustment mode ("rms_norm", "spectral_norm", or None).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        adjust_lr: str | None = "rms_norm",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr, mu=mu, nesterov=nesterov, weight_decay=weight_decay,
            ns_steps=ns_steps, adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation guide:
            1. Loop over param_groups, then over params.
            2. Skip params with no gradient.
            3. Lazy-init momentum buffer.
            4. Update momentum: buf = mu * buf + grad
            5. Compute Nesterov update: update = grad + mu * buf
               (or standard: update = buf)
            6. Orthogonalize: update = newton_schulz(update)
            7. Adjust LR based on param shape:
               if rms_norm:      adj_lr = lr * 0.2 * sqrt(max(fan_out, fan_in))
               if spectral_norm: adj_lr = lr * sqrt(fan_out / fan_in)
            8. Apply decoupled weight decay: param -= lr * wd * param
            9. Apply update: param -= adj_lr * update
        """
        # TODO: implement Muon step
        raise NotImplementedError(
            "Implement Muon! Follow the steps in the docstring above."
        )
