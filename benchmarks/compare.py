"""
Compare Benchmark Results
=========================

Load multiple result JSON files and generate comparison plots.

Usage:
    # Compare all results for a given model+dataset
    python benchmarks/compare.py --dir results/ --filter cifar10_vit

    # Compare specific files
    python benchmarks/compare.py results/file1.json results/file2.json

    # Save plot to file instead of showing
    python benchmarks/compare.py --dir results/ --output comparison.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(paths: list[Path]) -> list[dict]:
    """Load result JSON files."""
    results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
            data["_file"] = p.name
            results.append(data)
    return results


def plot_comparison(results: list[dict], output: str | None = None):
    """Generate comparison plots from multiple experiment results."""
    if not results:
        print("No results to compare.")
        return

    task_type = "language_modeling" if "test_ppl" in results[0]["history"][0] else "classification"

    if task_type == "classification":
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for r in results:
            label = f"{r['optimizer']} (lr={r['lr']})"
            epochs = [h["epoch"] for h in r["history"]]
            train_loss = [h["train_loss"] for h in r["history"]]
            test_acc = [h["test_acc"] for h in r["history"]]
            test_loss = [h["test_loss"] for h in r["history"]]

            axes[0].plot(epochs, train_loss, label=label)
            axes[1].plot(epochs, test_acc, label=label)
            axes[2].plot(epochs, test_loss, label=label)

        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Test Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_title("Test Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for r in results:
            label = f"{r['optimizer']} (lr={r['lr']})"
            epochs = [h["epoch"] for h in r["history"]]
            train_loss = [h["train_loss"] for h in r["history"]]
            test_ppl = [h["test_ppl"] for h in r["history"]]

            axes[0].plot(epochs, train_loss, label=label)
            axes[1].plot(epochs, test_ppl, label=label)

        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Test Perplexity")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Perplexity")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    model_name = results[0].get("model", "unknown")
    dataset_name = results[0].get("dataset", "unknown")
    fig.suptitle(f"Optimizer Comparison: {model_name} on {dataset_name}", fontsize=14)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Optimizer':<20} {'LR':<10} {'Final Train Loss':<18} ", end="")
    if task_type == "classification":
        print(f"{'Best Test Acc':<16} {'Total Time':<12}")
    else:
        print(f"{'Best Test PPL':<16} {'Total Time':<12}")
    print("=" * 80)

    for r in results:
        history = r["history"]
        opt_name = f"{r['optimizer']} (lr={r['lr']})"
        final_train_loss = history[-1]["train_loss"]
        total_time = f"{r.get('total_time', 0):.1f}s"

        if task_type == "classification":
            best_acc = max(h["test_acc"] for h in history)
            print(f"{opt_name:<20} {r['lr']:<10} {final_train_loss:<18.4f} {best_acc:<16.4f} {total_time:<12}")
        else:
            best_ppl = min(h["test_ppl"] for h in history)
            print(f"{opt_name:<20} {r['lr']:<10} {final_train_loss:<18.4f} {best_ppl:<16.2f} {total_time:<12}")


def main():
    parser = argparse.ArgumentParser(description="Compare optimizer benchmark results")
    parser.add_argument("files", nargs="*", help="Result JSON files to compare.")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing result files.")
    parser.add_argument("--filter", type=str, default=None, help="Filter filenames (substring match).")
    parser.add_argument("--output", type=str, default=None, help="Save plot to file.")
    args = parser.parse_args()

    paths = []
    if args.files:
        paths = [Path(f) for f in args.files]
    elif args.dir:
        result_dir = Path(args.dir)
        paths = sorted(result_dir.glob("*.json"))
        if args.filter:
            paths = [p for p in paths if args.filter in p.name]
    else:
        parser.print_help()
        sys.exit(1)

    if not paths:
        print("No result files found.")
        sys.exit(1)

    print(f"Loading {len(paths)} result files...")
    results = load_results(paths)
    plot_comparison(results, args.output)


if __name__ == "__main__":
    main()
