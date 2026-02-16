"""
Sophia-H Optimizer
==================
Paper: "Sophia: A Scalable Stochastic Second-order Optimizer for
        Language Model Pre-training" (Liu et al., Stanford, 2023)
       https://arxiv.org/abs/2305.14342

Core idea:
    Use a lightweight diagonal Hessian estimate to clip the Adam-style
    update. This prevents the optimizer from taking excessively large
    steps in directions of high curvature, while keeping the cost
    close to first-order methods.

    The Hessian diagonal is estimated via the Hutchinson estimator:
    diag(H) ~ E[z * (Hz)], where z is a random Rademacher vector.

Update rule:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t
    h_t = beta2 * h_{t-1} + (1 - beta2) * diag_hessian_t   (updated every k steps)
    param_t = param_{t-1} - lr * clip(m_t / max(h_t, rho), 1)

    where clip(x, c) = clamp(x, -c, c) element-wise.

Key differences from Adam:
    - Replaces second moment (grad^2) with actual Hessian diagonal
    - Uses clipping instead of division for stability
    - Hessian is only computed every k steps (e.g., k=10) to save cost
    - The rho parameter controls the maximum update magnitude

Hyperparameters:
    lr:           Learning rate (typical: 1e-4)
    betas:        (beta1, beta2) for momentum and Hessian EMA (typical: (0.965, 0.99))
    rho:          Clipping threshold (typical: 0.04)
    weight_decay: Decoupled weight decay (typical: 0.1)
    hessian_period: How often to update Hessian estimate (typical: 10 steps)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SophiaH(Optimizer):
    """Sophia-H optimizer (second-order clipped updates).

    More advanced than Adam -- requires understanding of Hessian estimation.
    The key insight is that clipping by curvature is more robust than
    dividing by curvature (as Adam does with sqrt(v)).

    Note: This optimizer requires a special training loop because the
    Hessian estimation needs a separate backward pass. See the docstring
    of ``estimate_hessian()`` for details.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        betas: (beta1, beta2) for momentum and Hessian EMA.
        rho: Maximum update magnitude (clipping threshold).
        weight_decay: Decoupled weight decay.
        hessian_period: Steps between Hessian updates.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        hessian_period: int = 10,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho: {rho}")

        defaults = dict(
            lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
            hessian_period=hessian_period,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    def estimate_hessian(self, loss_fn):
        """Estimate diagonal Hessian via Hutchinson's method.

        Call this every ``hessian_period`` steps in your training loop:

            if step % hessian_period == 0:
                def loss_fn():
                    return model(x).loss
                optimizer.estimate_hessian(loss_fn)

        Implementation guide:
            1. Sample Rademacher vector z (random +1/-1) for each param.
            2. Compute loss and backward to get gradients.
            3. Compute Hessian-vector product: Hv = autograd.grad(grad @ z, params)
            4. Diagonal estimate: h = z * Hv
            5. Update EMA: state['hessian'] = beta2 * h_old + (1 - beta2) * h
        """
        # TODO: implement Hutchinson Hessian estimation
        raise NotImplementedError(
            "Implement Hessian estimation! See the docstring above."
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation guide:
            1. Loop over param_groups, then over params.
            2. Skip params with no gradient.
            3. Apply decoupled weight decay: param -= lr * wd * param
            4. Lazy-init state: step, momentum (m), hessian diagonal (h).
            5. Update momentum: m = beta1 * m + (1 - beta1) * grad
            6. Compute clipped update:
               update = clamp(m / max(h, rho), -1, 1)
            7. Apply: param -= lr * update
        """
        # TODO: implement Sophia-H step
        raise NotImplementedError(
            "Implement Sophia-H step! Follow the steps in the docstring above."
        )
