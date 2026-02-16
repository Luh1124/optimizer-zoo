"""
SOAP Optimizer
==============
Paper: "SOAP: Improving and Stabilizing Shampoo using Adam"
       (Vyas et al., 2024)
       https://arxiv.org/abs/2409.11321

Core idea:
    Shampoo uses Kronecker-factored preconditioners to approximate the
    full matrix preconditioner. SOAP improves on Shampoo by running
    Adam in the eigenbasis of the Shampoo preconditioner, combining
    the benefits of both approaches.

    For a weight matrix W of shape (m, n):
    - Maintain left preconditioner L ~ (m, m) and right preconditioner R ~ (n, n)
    - L and R approximate the gradient covariance structure
    - Project gradient into the eigenbasis of (L, R), run Adam there,
      then project back

    The preconditioners are updated periodically (not every step) to
    amortize the cost of eigendecomposition.

Update rule (simplified):
    # Periodically update preconditioners:
    L = beta_prec * L + (1 - beta_prec) * grad @ grad^T
    R = beta_prec * R + (1 - beta_prec) * grad^T @ grad
    Q_L, Q_R = eigenvectors(L), eigenvectors(R)

    # Every step, run Adam in the rotated basis:
    grad_rotated = Q_L^T @ grad @ Q_R
    m, v = adam_update(grad_rotated)  # standard Adam on rotated grad
    update_rotated = m / (sqrt(v) + eps)
    update = Q_L @ update_rotated @ Q_R^T

    param -= lr * update

Key differences from Adam:
    - Captures cross-parameter correlations via Kronecker structure
    - More expensive per step (eigendecomposition), but converges faster
    - Preconditioner update is amortized (every k steps)
    - Memory: O(m^2 + n^2) extra for preconditioners

Hyperparameters:
    lr:              Learning rate (typical: 1e-3)
    betas:           (beta1, beta2) for Adam in rotated basis (typical: (0.9, 0.999))
    eps:             Numerical stability (typical: 1e-8)
    weight_decay:    Decoupled weight decay (typical: 0.01)
    precond_beta:    EMA decay for preconditioners (typical: 0.999)
    precond_period:  Steps between preconditioner updates (typical: 10)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SOAP(Optimizer):
    """SOAP optimizer (Shampoo + Adam in eigenbasis).

    Advanced optimizer -- understand Adam and basic linear algebra
    (eigendecomposition, Kronecker products) before attempting this.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        betas: (beta1, beta2) for Adam in the preconditioned basis.
        eps: Numerical stability term.
        weight_decay: Decoupled weight decay.
        precond_beta: EMA decay rate for preconditioner matrices.
        precond_period: How often to update preconditioners (in steps).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precond_beta: float = 0.999,
        precond_period: int = 10,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            precond_beta=precond_beta, precond_period=precond_period,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation guide:
            1. Increment global step count.
            2. Loop over param_groups, then over params.
            3. Skip params with no gradient.
            4. Apply decoupled weight decay.
            5. Lazy-init state: step, m, v, L (left precond), R (right precond),
               Q_L (left eigenvectors), Q_R (right eigenvectors).
            6. If step % precond_period == 0:
               a. Update L = precond_beta * L + (1 - precond_beta) * grad @ grad^T
               b. Update R = precond_beta * R + (1 - precond_beta) * grad^T @ grad
               c. Compute eigenvectors: Q_L = eigh(L), Q_R = eigh(R)
            7. Rotate gradient: grad_rot = Q_L^T @ grad @ Q_R
            8. Adam update in rotated basis:
               m = beta1 * m + (1 - beta1) * grad_rot
               v = beta2 * v + (1 - beta2) * grad_rot^2
               (apply bias correction)
               update_rot = m_hat / (sqrt(v_hat) + eps)
            9. Rotate back: update = Q_L @ update_rot @ Q_R^T
            10. Apply: param -= lr * update
        """
        # TODO: implement SOAP
        raise NotImplementedError(
            "Implement SOAP! This is an advanced optimizer -- see the docstring above."
        )
