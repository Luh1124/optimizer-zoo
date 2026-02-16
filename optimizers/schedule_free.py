"""
Schedule-Free AdamW
===================
Paper: "The Road Less Scheduled" (Defazio et al., Meta, 2024)
       https://arxiv.org/abs/2405.15682

Core idea:
    Eliminate the need for learning rate schedules (warmup, cosine decay, etc.)
    by maintaining two sequences of iterates and interpolating between them.

    The key insight: learning rate schedules exist to balance exploration
    (large lr early) and exploitation (small lr late). Schedule-Free
    achieves this automatically by maintaining:
    - z: the "fast" iterate (large steps, like using a big lr)
    - x: the "slow" iterate (averaged, like using a small lr)
    The model weights are set to an interpolation of z and x.

    At evaluation time, use x (the averaged iterate) for best performance.

Update rule:
    # Interpolate to get evaluation point:
    y_t = (1 - beta1) * z_t + beta1 * x_t

    # Gradient is computed at y_t (the model weights during forward pass)
    grad_t = nabla f(y_t)

    # Update z (fast iterate, like Adam without schedule):
    z_{t+1} = z_t - lr * grad_t / (sqrt(v_t) + eps)   # Adam-style

    # Update x (slow iterate, Polyak averaging):
    x_{t+1} = (1 - 1/(t+1)) * x_t + 1/(t+1) * z_{t+1}

    # Set model weights to interpolation for next forward pass:
    model.params = (1 - beta1) * z_{t+1} + beta1 * x_{t+1}

Key differences from Adam:
    - No learning rate schedule needed (no warmup, no cosine decay)
    - Must call optimizer.eval() before evaluation and optimizer.train()
      before training (to switch between x and y iterates)
    - Slightly more memory (stores both z and x)

Hyperparameters:
    lr:           Learning rate (typical: same as Adam peak lr)
    betas:        (beta1, beta2) (typical: (0.9, 0.999))
    eps:          Numerical stability (typical: 1e-8)
    weight_decay: Decoupled weight decay (typical: 0.01)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class ScheduleFreeAdamW(Optimizer):
    """Schedule-Free AdamW optimizer.

    Interesting conceptually -- shows that lr schedules are not fundamental,
    but rather a workaround for limitations of standard optimizers.

    Important: You must call ``.train()`` and ``.eval()`` on this optimizer
    (not just the model) to switch between training and evaluation modes.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (no schedule needed).
        betas: (beta1, beta2) coefficients.
        eps: Numerical stability term.
        weight_decay: Decoupled weight decay.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._step_count = 0

    def eval(self):
        """Switch to evaluation mode: set model params to x (averaged iterate).

        Implementation guide:
            For each param, store current y in state, then set param.data = x.
        """
        # TODO: implement eval mode switch
        raise NotImplementedError("Implement eval mode switch!")

    def train(self):
        """Switch to training mode: restore model params to y (interpolated).

        Implementation guide:
            For each param, restore param.data from the saved y in state.
        """
        # TODO: implement train mode switch
        raise NotImplementedError("Implement train mode switch!")

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation guide:
            1. Increment step count.
            2. Loop over param_groups, then over params.
            3. Skip params with no gradient.
            4. Apply decoupled weight decay to z.
            5. Lazy-init state: z (fast iterate), x (slow iterate), v (second moment).
            6. Adam-style update on z:
               v = beta2 * v + (1 - beta2) * grad^2
               z -= lr * grad / (sqrt(v) + eps)
            7. Polyak averaging for x:
               x = (1 - 1/step) * x + (1/step) * z
            8. Set param to interpolation:
               param.data = (1 - beta1) * z + beta1 * x
        """
        # TODO: implement Schedule-Free AdamW step
        raise NotImplementedError(
            "Implement Schedule-Free AdamW! See the docstring above."
        )
