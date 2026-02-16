# Optimizer Zoo

A beginner-friendly collection of optimizer implementations for learning and benchmarking.

**The goal**: Implement each optimizer yourself, from scratch, to deeply understand how it works. The repository provides detailed docstrings with math, pseudocode, and step-by-step implementation guides -- you fill in the actual code.

## Learning Roadmap

Recommended order (each builds on the previous):

| # | Optimizer | Difficulty | Key Concept | Paper |
|---|-----------|-----------|-------------|-------|
| 1 | **SGD + Momentum** | Beginner | Velocity buffer, Nesterov lookahead | [Sutskever 2013](https://proceedings.mlr.press/v28/sutskever13.html) |
| 2 | **Adam / AdamW** | Beginner | Adaptive learning rates, bias correction | [Kingma 2014](https://arxiv.org/abs/1412.6980), [Loshchilov 2017](https://arxiv.org/abs/1711.05101) |
| 3 | **Lion** | Beginner | sign() update, program search discovery | [Chen 2023](https://arxiv.org/abs/2302.06675) |
| 4 | **Muon** | Intermediate | Newton-Schulz orthogonalization, polar decomposition | [Jordan 2025](https://arxiv.org/abs/2502.16982) |
| 5 | **Schedule-Free** | Intermediate | Iterate averaging, no lr schedule needed | [Defazio 2024](https://arxiv.org/abs/2405.15682) |
| 6 | **Sophia** | Advanced | Diagonal Hessian estimation, clipped updates | [Liu 2023](https://arxiv.org/abs/2305.14342) |
| 7 | **SOAP** | Advanced | Kronecker preconditioners, eigenbasis rotation | [Vyas 2024](https://arxiv.org/abs/2409.11321) |
| 8 | **PSGD-Kron** | Expert | Lie group preconditioner fitting | [Li 2015](https://arxiv.org/abs/1512.04202) |

## Project Structure

```
optimizer-zoo/
├── optimizers/          # YOUR IMPLEMENTATIONS GO HERE
│   ├── __init__.py      # Registry (get_optimizer, list_optimizers)
│   ├── sgd_momentum.py  # SGD + Momentum + Nesterov
│   ├── adam.py          # Adam / AdamW
│   ├── lion.py          # Lion
│   ├── muon.py          # Muon
│   ├── sophia.py        # Sophia-H
│   ├── soap.py          # SOAP
│   ├── schedule_free.py # Schedule-Free AdamW
│   └── psgd.py          # PSGD-Kron
│
├── models/              # Benchmark networks (pre-built)
│   ├── mlp.py           # MLP for MNIST (~1 min)
│   ├── resnet.py        # ResNet-18 for CIFAR-10 (~5 min)
│   ├── vit.py           # ViT-Tiny for CIFAR-10 (~10 min)
│   └── gpt2.py          # Small GPT-2 for WikiText-2 (~20 min)
│
├── benchmarks/          # Training & comparison scripts
│   ├── train.py         # Unified training entry point
│   ├── compare.py       # Plot comparison charts
│   └── configs/         # Default hyperparameters per task
│
├── tests/               # Correctness verification
│   └── test_optimizers.py
│
└── scripts/
    └── run_all.sh       # Run all benchmarks for one optimizer
```

## Quick Start

### Setup

```bash
cd optimizer-zoo
pip install -e ".[dev]"
```

### Workflow: Implement -> Test -> Benchmark

**Step 1: Read the docstring** in the optimizer file. It contains the math, pseudocode, and step-by-step guide.

**Step 2: Implement** the `step()` method (replace `raise NotImplementedError`).

**Step 3: Test** your implementation:

```bash
# Run convergence test for all implemented optimizers
pytest tests/test_optimizers.py -v

# Run test for a specific optimizer
pytest tests/test_optimizers.py -v -k "adam"
```

**Step 4: Benchmark** against other optimizers:

```bash
# Quick sanity check: MLP on MNIST
python benchmarks/train.py --model mlp --dataset mnist --optimizer adam --lr 1e-3

# Full benchmark: ViT on CIFAR-10
python benchmarks/train.py --config benchmarks/configs/cifar10_vit.yaml --optimizer adam

# Run all benchmarks for one optimizer
./scripts/run_all.sh adam

# Compare results
python benchmarks/compare.py --dir results/
```

### Example: Implementing Lion

1. Open `optimizers/lion.py` and read the docstring
2. The core is just 3 lines:
   ```python
   update = torch.sign(beta1 * m + (1 - beta1) * grad)  # direction
   param -= lr * update                                   # step
   m = beta2 * m + (1 - beta2) * grad                    # update momentum
   ```
3. Run `pytest tests/test_optimizers.py -v -k "lion"` to verify
4. Run `python benchmarks/train.py --model mlp --dataset mnist --optimizer lion --lr 1e-4`

## Benchmark Results

Fill in as you implement each optimizer:

### MNIST + MLP

| Optimizer | LR | Best Test Acc | Train Time |
|-----------|-----|---------------|------------|
| SGD+Momentum | | | |
| AdamW | | | |
| Lion | | | |
| Muon | | | |

### CIFAR-10 + ViT-Tiny

| Optimizer | LR | Best Test Acc | Train Time |
|-----------|-----|---------------|------------|
| AdamW | | | |
| Lion | | | |
| Muon | | | |
| SOAP | | | |

### WikiText-2 + GPT-2

| Optimizer | LR | Best Test PPL | Train Time |
|-----------|-----|---------------|------------|
| AdamW | | | |
| Muon | | | |
| SOAP | | | |

## Tips

- **Start simple**: Implement SGD first to understand the `Optimizer` base class
- **Read PyTorch source**: `torch.optim.Adam` is a great reference for AdamW
- **Use the tests**: They catch most implementation bugs automatically
- **Compare carefully**: Different optimizers need different learning rates. The configs provide reasonable defaults
- **Focus on Transformers**: Muon and SOAP shine on ViT and GPT-2, not so much on MLP/CNN

## License

MIT
