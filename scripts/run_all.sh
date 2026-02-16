#!/usr/bin/env bash
set -euo pipefail

# Run all benchmarks for a given optimizer.
# Usage: ./scripts/run_all.sh <optimizer_name>
# Example: ./scripts/run_all.sh adam

OPTIMIZER="${1:?Usage: $0 <optimizer_name>}"

echo "Running all benchmarks with optimizer: ${OPTIMIZER}"
echo "=============================================="

echo ""
echo "[1/4] MNIST + MLP"
python benchmarks/train.py --config benchmarks/configs/mnist_mlp.yaml --optimizer "${OPTIMIZER}"

echo ""
echo "[2/4] CIFAR-10 + ResNet-18"
python benchmarks/train.py --config benchmarks/configs/cifar10_resnet.yaml --optimizer "${OPTIMIZER}"

echo ""
echo "[3/4] CIFAR-10 + ViT-Tiny"
python benchmarks/train.py --config benchmarks/configs/cifar10_vit.yaml --optimizer "${OPTIMIZER}"

echo ""
echo "[4/4] WikiText-2 + GPT-2 Small"
python benchmarks/train.py --config benchmarks/configs/wikitext2_gpt2.yaml --optimizer "${OPTIMIZER}"

echo ""
echo "All benchmarks complete! Results saved to results/"
echo "Run: python benchmarks/compare.py --dir results/"
