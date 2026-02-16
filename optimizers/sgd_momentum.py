"""
SGD with Momentum (and Nesterov)
================================
The foundational optimizer. Understanding SGD + momentum is prerequisite
for everything else in this repo.

References:
    - Polyak (1964): "Some methods of speeding up the convergence of iteration methods"
    - Sutskever et al. (2013): "On the importance of initialization and momentum in deep learning"
      https://proceedings.mlr.press/v28/sutskever13.html

Core idea:
    Maintain a velocity buffer that accumulates gradient history,
    providing inertia to escape shallow local minima and accelerate
    convergence along consistent gradient directions.

Update rule (standard momentum):
    v_t = mu * v_{t-1} + grad_t
    param_t = param_{t-1} - lr * v_t

Update rule (Nesterov momentum):
    v_t = mu * v_{t-1} + grad_t
    param_t = param_{t-1} - lr * (grad_t + mu * v_t)

    Nesterov "looks ahead" by evaluating the gradient at the anticipated
    next position, giving better convergence on convex problems.

Weight decay (decoupled, a la AdamW style):
    param_t = param_t - lr * wd * param_{t-1}

Hyperparameters:
    lr:           Learning rate (typical: 0.01 - 0.1)
    mu:           Momentum factor (typical: 0.9 - 0.99)
    weight_decay: Decoupled weight decay (typical: 0 - 1e-4)
    nesterov:     Whether to use Nesterov momentum (default: False)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SGDMomentum(Optimizer):
    """SGD with (optionally Nesterov) momentum and decoupled weight decay.

    This is the simplest optimizer in the zoo. Implement it first to
    understand the Optimizer base class and state management.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        mu: Momentum factor.
        weight_decay: Decoupled weight decay coefficient.
        nesterov: If True, use Nesterov momentum.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        mu: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum value: {mu}")

        defaults = dict(lr=lr, mu=mu, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation guide:
            1. Loop over param_groups, then over params in each group.
            2. Skip params with no gradient.
            3. Apply decoupled weight decay: param -= lr * wd * param
            4. Lazy-init the velocity buffer in self.state[param].
            5. Update velocity: v = mu * v + grad
            6. If nesterov: update = grad + mu * v
               Else:        update = v
            7. Apply: param -= lr * update
        """
        # TODO: implement SGD with momentum
        raise NotImplementedError(
            "Implement SGD with momentum! Follow the steps in the docstring above."
        )
