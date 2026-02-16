"""
Adam / AdamW Optimizer
======================
Paper: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
       https://arxiv.org/abs/1412.6980
       "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
       https://arxiv.org/abs/1711.05101

Core idea:
    Maintain per-parameter first moment (mean) and second moment (variance)
    estimates of the gradient. Use bias-corrected moments to compute an
    adaptive learning rate for each parameter.

    AdamW decouples weight decay from the gradient-based update, applying
    it directly to the parameters instead of through the gradient.

Update rule:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t           # first moment
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad_t^2         # second moment
    m_hat = m_t / (1 - beta1^t)                             # bias correction
    v_hat = v_t / (1 - beta2^t)                             # bias correction
    param_t = param_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    For AdamW, weight decay is applied separately:
    param_t = param_t - lr * wd * param_{t-1}

Key insight:
    - The second moment v_t acts as a per-parameter learning rate scaler.
    - Parameters with large/noisy gradients get smaller effective lr.
    - Parameters with small/consistent gradients get larger effective lr.
    - This is why Adam works well "out of the box" for most problems.

Hyperparameters:
    lr:           Learning rate (typical: 1e-3 to 3e-4)
    betas:        (beta1, beta2) moment decay rates (typical: (0.9, 0.999))
    eps:          Numerical stability (typical: 1e-8)
    weight_decay: Decoupled weight decay (typical: 0.01 - 0.1)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay).

    This is the most widely used optimizer in deep learning today.
    Understanding it thoroughly is essential before moving to newer optimizers.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        betas: Coefficients for computing running averages of gradient
               and its square (beta1, beta2).
        eps: Term added to denominator for numerical stability.
        weight_decay: Decoupled weight decay coefficient.
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
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Implementation guide:
            1. Loop over param_groups, then over params.
            2. Skip params with no gradient.
            3. Apply decoupled weight decay: param -= lr * wd * param
            4. Lazy-init state: step count, first moment (m), second moment (v).
            5. Increment step count.
            6. Update moments:
               m = beta1 * m + (1 - beta1) * grad
               v = beta2 * v + (1 - beta2) * grad^2
            7. Bias correction:
               m_hat = m / (1 - beta1^step)
               v_hat = v / (1 - beta2^step)
            8. Update param: param -= lr * m_hat / (sqrt(v_hat) + eps)

        Tip: torch.Tensor.lerp_(grad, 1 - beta1) is equivalent to
             m = beta1 * m + (1 - beta1) * grad, and is slightly faster.
        """
        # TODO: implement AdamW
        raise NotImplementedError(
            "Implement AdamW! Follow the steps in the docstring above."
        )
