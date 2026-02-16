"""
Optimizer Zoo - Registry & Unified Export
=========================================

All optimizers follow the standard torch.optim.Optimizer interface.
Use ``get_optimizer(name, params, **kwargs)`` to instantiate by name.

Usage:
    from optimizers import get_optimizer, list_optimizers

    opt = get_optimizer("adam", model.parameters(), lr=1e-3)
    print(list_optimizers())  # ['sgd_momentum', 'adam', 'lion', ...]
"""

from torch.optim.optimizer import Optimizer

from optimizers.adam import AdamW
from optimizers.lion import Lion
from optimizers.muon import Muon
from optimizers.psgd import PSGDKron
from optimizers.schedule_free import ScheduleFreeAdamW
from optimizers.sgd_momentum import SGDMomentum
from optimizers.soap import SOAP
from optimizers.sophia import SophiaH

OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "sgd_momentum": SGDMomentum,
    "adam": AdamW,
    "lion": Lion,
    "sophia": SophiaH,
    "muon": Muon,
    "soap": SOAP,
    "schedule_free": ScheduleFreeAdamW,
    "psgd": PSGDKron,
}


def get_optimizer(name: str, params, **kwargs) -> Optimizer:
    """Instantiate an optimizer by its registry name.

    Args:
        name: One of the keys in OPTIMIZER_REGISTRY.
        params: Model parameters (iterable or param groups).
        **kwargs: Forwarded to the optimizer constructor.

    Returns:
        An optimizer instance.
    """
    if name not in OPTIMIZER_REGISTRY:
        available = ", ".join(sorted(OPTIMIZER_REGISTRY.keys()))
        raise ValueError(f"Unknown optimizer '{name}'. Available: {available}")
    return OPTIMIZER_REGISTRY[name](params, **kwargs)


def list_optimizers() -> list[str]:
    """Return sorted list of available optimizer names."""
    return sorted(OPTIMIZER_REGISTRY.keys())
