"""
Unified Training Script
=======================

Train any (model, dataset, optimizer) combination from the command line.

Usage:
    # Quick sanity check with MLP on MNIST
    python benchmarks/train.py --model mlp --dataset mnist --optimizer adam --lr 1e-3

    # ViT on CIFAR-10 with Muon
    python benchmarks/train.py --model vit_tiny --dataset cifar10 --optimizer muon --lr 0.02

    # GPT-2 on WikiText-2
    python benchmarks/train.py --model gpt2_small --dataset wikitext2 --optimizer adam --lr 3e-4

    # Load config from YAML (overrides defaults)
    python benchmarks/train.py --config benchmarks/configs/cifar10_vit.yaml --optimizer muon

Results are saved to results/<model>_<dataset>_<optimizer>_<timestamp>.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import get_model
from optimizers import get_optimizer, list_optimizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def get_dataloaders(dataset: str, batch_size: int, num_workers: int = 2):
    """Get train and test dataloaders.

    Returns:
        (train_loader, test_loader, task_type)
        task_type is "classification" or "language_modeling".
    """
    if dataset == "mnist":
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        return train_loader, test_loader, "classification"

    elif dataset == "cifar10":
        from torchvision import datasets, transforms

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10("data", train=False, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        return train_loader, test_loader, "classification"

    elif dataset == "wikitext2":
        try:
            from datasets import load_dataset
            from transformers import GPT2Tokenizer
        except ImportError:
            raise ImportError(
                "WikiText-2 requires: pip install datasets transformers"
            )

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")

        seq_len = 256

        def tokenize_and_chunk(examples):
            tokens = tokenizer(
                examples["text"], truncation=False, padding=False,
            )["input_ids"]
            # Concatenate all tokens and chunk into fixed-length sequences
            all_tokens = [t for seq in tokens for t in seq]
            chunks = [
                all_tokens[i : i + seq_len]
                for i in range(0, len(all_tokens) - seq_len, seq_len)
            ]
            return {"input_ids": chunks}

        train_ds = raw["train"].map(
            tokenize_and_chunk, batched=True, remove_columns=["text"],
        )
        test_ds = raw["validation"].map(
            tokenize_and_chunk, batched=True, remove_columns=["text"],
        )
        train_ds.set_format("torch")
        test_ds.set_format("torch")

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        return train_loader, test_loader, "language_modeling"

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: mnist, cifar10, wikitext2")


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch_classification(model, loader, optimizer, device):
    """Train one epoch for classification tasks. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_classification(model, loader, device):
    """Evaluate classification model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def train_one_epoch_lm(model, loader, optimizer, device):
    """Train one epoch for language modeling. Returns avg_loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)

        optimizer.zero_grad()
        loss = model.compute_loss(input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

    return total_loss / total_tokens


@torch.no_grad()
def evaluate_lm(model, loader, device):
    """Evaluate language model. Returns (avg_loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        loss = model.compute_loss(input_ids)
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import math


def main():
    parser = argparse.ArgumentParser(description="Optimizer Zoo Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--model", type=str, default="mlp", help="Model name.")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer name.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Load config from YAML if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # CLI args override config
    model_name = args.model or config.get("model", "mlp")
    dataset_name = args.dataset or config.get("dataset", "mnist")
    optimizer_name = args.optimizer or config.get("optimizer", "adam")
    lr = args.lr or config.get("lr", 1e-3)
    weight_decay = args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.01)
    batch_size = args.batch_size or config.get("batch_size", 128)
    epochs = args.epochs or config.get("epochs", 10)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Model: {model_name} | Dataset: {dataset_name} | Optimizer: {optimizer_name}")
    logger.info(f"LR: {lr} | WD: {weight_decay} | Batch: {batch_size} | Epochs: {epochs}")
    logger.info(f"Device: {device}")

    # Build model
    model = get_model(model_name).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Build optimizer
    opt_kwargs = dict(lr=lr, weight_decay=weight_decay)
    # Add optimizer-specific kwargs from config
    opt_config = config.get("optimizer_kwargs", {})
    opt_kwargs.update(opt_config)
    optimizer = get_optimizer(optimizer_name, model.parameters(), **opt_kwargs)

    # Build dataloaders
    train_loader, test_loader, task_type = get_dataloaders(dataset_name, batch_size)

    # Training loop
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_params": num_params,
        "history": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if task_type == "classification":
            train_loss, train_acc = train_one_epoch_classification(
                model, train_loader, optimizer, device,
            )
            test_loss, test_acc = evaluate_classification(model, test_loader, device)
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            results["history"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "epoch_time": epoch_time,
            })

        elif task_type == "language_modeling":
            train_loss = train_one_epoch_lm(model, train_loader, optimizer, device)
            test_loss, test_ppl = evaluate_lm(model, test_loader, device)
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} PPL: {test_ppl:.2f} | "
                f"Time: {epoch_time:.1f}s"
            )
            results["history"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_ppl": test_ppl,
                "epoch_time": epoch_time,
            })

    total_time = time.time() - start_time
    results["total_time"] = total_time
    logger.info(f"Training complete in {total_time:.1f}s")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"{model_name}_{dataset_name}_{optimizer_name}_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
