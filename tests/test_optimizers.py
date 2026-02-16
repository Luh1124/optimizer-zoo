"""
Optimizer Convergence Tests
===========================

Simple tests to verify that each optimizer implementation is correct.
Uses a quadratic objective f(x) = ||x - target||^2 which every reasonable
optimizer should be able to minimize.

Run with:
    pytest tests/test_optimizers.py -v

When you implement a new optimizer, it should pass these tests automatically.
If it doesn't, there's likely a bug in your implementation.
"""

import pytest
import torch
from torch import nn

from optimizers import OPTIMIZER_REGISTRY, get_optimizer


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class QuadraticModel(nn.Module):
    """A simple model whose loss is ||param - target||^2.

    This is the simplest possible optimization problem. Every optimizer
    should converge to target within a few hundred steps.
    """

    def __init__(self, dim: int = 10):
        super().__init__()
        self.param = nn.Parameter(torch.randn(dim))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return ((self.param - target) ** 2).sum()


class MatrixQuadraticModel(nn.Module):
    """A 2D parameter model: loss = ||W - target||_F^2.

    Useful for testing optimizers that treat 2D params specially (e.g., Muon).
    """

    def __init__(self, rows: int = 8, cols: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rows, cols))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return ((self.weight - target) ** 2).sum()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _check_convergence(
    optimizer_name: str,
    model: nn.Module,
    target: torch.Tensor,
    steps: int = 500,
    tol: float = 0.1,
    **opt_kwargs,
):
    """Run optimizer for `steps` iterations and check convergence.

    Args:
        optimizer_name: Name in the registry.
        model: Model with a forward(target) -> loss method.
        target: Target tensor.
        steps: Number of optimization steps.
        tol: Maximum acceptable final loss.
        **opt_kwargs: Extra kwargs for the optimizer.
    """
    try:
        optimizer = get_optimizer(optimizer_name, model.parameters(), **opt_kwargs)
    except NotImplementedError:
        pytest.skip(f"Optimizer '{optimizer_name}' not yet implemented.")

    initial_loss = None
    for step in range(steps):
        optimizer.zero_grad()
        loss = model(target)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        try:
            optimizer.step()
        except NotImplementedError:
            pytest.skip(f"Optimizer '{optimizer_name}' step() not yet implemented.")

    final_loss = model(target).item()
    assert final_loss < tol, (
        f"{optimizer_name}: final loss {final_loss:.6f} > tolerance {tol}. "
        f"Initial loss was {initial_loss:.6f}. The optimizer may have a bug."
    )
    assert final_loss < initial_loss * 0.01, (
        f"{optimizer_name}: loss only decreased from {initial_loss:.6f} to {final_loss:.6f}. "
        f"Expected at least 100x reduction."
    )


# ---------------------------------------------------------------------------
# Tests for each optimizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("optimizer_name", list(OPTIMIZER_REGISTRY.keys()))
def test_quadratic_convergence(optimizer_name: str):
    """Test that each optimizer can minimize a simple quadratic."""
    torch.manual_seed(42)
    model = QuadraticModel(dim=10)
    target = torch.randn(10)

    # Use conservative hyperparameters that should work for all optimizers
    kwargs = {"lr": 0.01, "weight_decay": 0.0}

    _check_convergence(optimizer_name, model, target, steps=500, tol=0.1, **kwargs)


@pytest.mark.parametrize("optimizer_name", list(OPTIMIZER_REGISTRY.keys()))
def test_matrix_quadratic_convergence(optimizer_name: str):
    """Test convergence on a 2D parameter (important for Muon/SOAP)."""
    torch.manual_seed(42)
    model = MatrixQuadraticModel(rows=8, cols=8)
    target = torch.randn(8, 8)

    kwargs = {"lr": 0.01, "weight_decay": 0.0}

    _check_convergence(optimizer_name, model, target, steps=500, tol=0.5, **kwargs)


def test_weight_decay_effect():
    """Test that weight decay actually shrinks parameters (for any implemented optimizer)."""
    for name in OPTIMIZER_REGISTRY:
        torch.manual_seed(42)
        model = QuadraticModel(dim=10)
        # Set target to zero, so weight decay should help
        target = torch.zeros(10)

        try:
            optimizer = get_optimizer(name, model.parameters(), lr=0.01, weight_decay=0.1)
        except (NotImplementedError, TypeError):
            continue

        initial_norm = model.param.data.norm().item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(target)
            loss.backward()
            try:
                optimizer.step()
            except NotImplementedError:
                break
        else:
            final_norm = model.param.data.norm().item()
            assert final_norm < initial_norm, (
                f"{name}: weight decay did not shrink parameters. "
                f"Norm: {initial_norm:.4f} -> {final_norm:.4f}"
            )


def test_optimizer_registry():
    """Test that the registry is properly set up."""
    from optimizers import list_optimizers

    names = list_optimizers()
    assert len(names) >= 8, f"Expected at least 8 optimizers, got {len(names)}"
    assert "adam" in names
    assert "sgd_momentum" in names
    assert "muon" in names
