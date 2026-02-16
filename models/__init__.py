"""
Benchmark Models
================

Pre-built models for testing optimizer implementations.
These are NOT the learning target -- they are tools for benchmarking.

Usage:
    from models import get_model

    model = get_model("mlp")           # MLP for MNIST
    model = get_model("resnet18")      # ResNet-18 for CIFAR-10
    model = get_model("vit_tiny")      # ViT-Tiny for CIFAR-10
    model = get_model("gpt2_small")    # Small GPT-2 for WikiText-2
"""

from models.gpt2 import GPT2Small
from models.mlp import MLP
from models.resnet import ResNet18
from models.vit import ViTTiny

MODEL_REGISTRY: dict[str, type] = {
    "mlp": MLP,
    "resnet18": ResNet18,
    "vit_tiny": ViTTiny,
    "gpt2_small": GPT2Small,
}


def get_model(name: str, **kwargs):
    """Instantiate a model by its registry name.

    Args:
        name: One of the keys in MODEL_REGISTRY.
        **kwargs: Forwarded to the model constructor.

    Returns:
        A model instance.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of available model names."""
    return sorted(MODEL_REGISTRY.keys())
