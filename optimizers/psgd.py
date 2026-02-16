"""
PSGD-Kron (Preconditioned SGD with Kronecker-factored preconditioner)
=====================================================================
Paper: "Preconditioned Stochastic Gradient Descent" (Li, 2015)
       https://arxiv.org/abs/1512.04202
       "Curvature-Informed SGD via General Purpose Lie-Group Preconditioners" (Shukla et al., 2024)
       https://arxiv.org/abs/2402.11858

Core idea:
    Maintain a Kronecker-factored preconditioner P = Q_l kron Q_r that
    approximates the inverse Hessian. Update P using a stochastic fitting
    procedure: perturb parameters, observe how the gradient changes, and
    adjust P to make the preconditioned gradient perturbation match the
    parameter perturbation in scale.

    Unlike Shampoo/SOAP which use gradient covariance, PSGD directly
    fits the preconditioner to the local curvature using random probes.

Update rule (simplified):
    # Periodically update preconditioner:
    delta = random perturbation
    grad_delta = grad(params + delta) - grad(params)  # or Hessian-vector product
    # Fit Q_l, Q_r so that (Q_l kron Q_r) @ grad_delta ~ delta

    # Every step:
    precond_grad = Q_l @ grad @ Q_r^T
    param -= lr * precond_grad

Key differences from other preconditioned methods:
    - Fits preconditioner to curvature directly (not gradient covariance)
    - Uses Lie group updates for Q_l, Q_r (stays on the manifold)
    - Can use Hessian-vector products instead of finite differences
    - More theoretically grounded but also more complex

Hyperparameters:
    lr:              Learning rate (typical: 0.01)
    momentum:        Momentum factor (typical: 0.9)
    precond_lr:      Learning rate for preconditioner update (typical: 0.1)
    precond_period:  Steps between preconditioner updates (typical: 10)
    weight_decay:    Decoupled weight decay (typical: 0.0)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class PSGDKron(Optimizer):
    """PSGD with Kronecker-factored preconditioner.

    The most theoretically involved optimizer in this zoo. Understand
    Adam, Shampoo/SOAP concepts, and Lie groups before attempting.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        momentum: Momentum factor.
        precond_lr: Learning rate for preconditioner fitting.
        precond_period: Steps between preconditioner updates.
        weight_decay: Decoupled weight decay.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        precond_lr: float = 0.1,
        precond_period: int = 10,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr, momentum=momentum, precond_lr=precond_lr,
            precond_period=precond_period, weight_decay=weight_decay,
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
            5. Lazy-init state: momentum buffer, Q_l, Q_r (Kronecker factors,
               initialized to identity matrices).
            6. If step % precond_period == 0:
               a. Sample random perturbation delta ~ N(0, I).
               b. Compute (or approximate) Hessian-vector product: Hv = H @ delta
               c. Update Q_l, Q_r via Lie group gradient descent to minimize
                  || (Q_l kron Q_r) @ Hv - delta ||^2
            7. Precondition gradient: precond_grad = Q_l @ grad @ Q_r^T
            8. Momentum: buf = momentum * buf + precond_grad
            9. Apply: param -= lr * buf
        """
        # TODO: implement PSGD-Kron
        raise NotImplementedError(
            "Implement PSGD-Kron! This is the most advanced optimizer in the zoo."
        )
