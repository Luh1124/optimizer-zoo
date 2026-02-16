"""
Lion Optimizer
==============
Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., Google Brain, 2023)
       https://arxiv.org/abs/2302.06675

Core idea:
    Discovered via program search over optimizer update rules.
    Replaces Adam's adaptive scaling with a simple sign() operation
    on an interpolation of momentum and gradient.

    Only 3 lines of core logic -- the simplest "new" optimizer to implement.
    Uses ~50% less memory than Adam (no second moment buffer).

Update rule:
    update_t = sign(beta1 * m_{t-1} + (1 - beta1) * grad_t)
    param_t  = param_{t-1} - lr * update_t - lr * wd * param_{t-1}
    m_t      = beta2 * m_{t-1} + (1 - beta2) * grad_t

    Note the asymmetry: beta1 is used for the update interpolation,
    beta2 is used for the momentum update. Typically beta1 < beta2.

Key differences from Adam:
    - No second moment (v), no sqrt, no epsilon
    - sign() produces uniform magnitude updates (+/- lr per element)
    - Memory: only 1 buffer (m) vs Adam's 2 (m, v)
    - Tends to need smaller lr than Adam (e.g., 3x-10x smaller)
    - Works best with larger weight decay

Hyperparameters:
    lr:           Learning rate (typical: 1e-4, ~3-10x smaller than Adam)
    betas:        (beta1, beta2) for update and momentum (typical: (0.9, 0.99))
    weight_decay: Decoupled weight decay (typical: 0.1 - 1.0, larger than Adam)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion optimizer (EvoLved sIgn mOmeNtum).

    The simplest new optimizer to implement. Start here after Adam
    to see how a fundamentally different update rule can work.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (use ~3-10x smaller than Adam).
        betas: (beta1, beta2). beta1 for update interp, beta2 for momentum.
        weight_decay: Decoupled weight decay (use larger than Adam).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation guide (only 3 key lines!):
            1. Loop over param_groups, then over params.
            2. Skip params with no gradient.
            3. Apply decoupled weight decay: param -= lr * wd * param
            4. Lazy-init momentum buffer m (zeros_like).
            5. Compute update direction:
               update = sign(beta1 * m + (1 - beta1) * grad)
               (or equivalently: use lerp then sign)
            6. Apply update: param -= lr * update
            7. Update momentum for NEXT step:
               m = beta2 * m + (1 - beta2) * grad

        Important: The momentum update (step 7) happens AFTER the param
        update (step 6). This ordering matters!
        """
        # TODO: implement Lion
        raise NotImplementedError(
            "Implement Lion! Only 3 core lines -- see the docstring above."
        )
