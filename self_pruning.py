"""
Self-Pruning Neural Network on CIFAR-10
========================================
A feed-forward network that learns to prune its own weights during training.
Each weight gets a learnable "gate" — an L1 penalty on these gates drives
unimportant connections to zero, producing a sparse network automatically.

Usage:
    python self_pruning_network.py
    python self_pruning_network.py --epochs 50 --lambdas 0 1e-5 1e-4 1e-3
    python self_pruning_network.py --device cpu
"""

import math
import time
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt


# Config & Constants

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

GATE_INIT_VALUE = 2.0   # sigmoid(2) ≈ 0.88, gates start open — only λ pressure drives them to 0
PRUNE_THRESHOLD = 1e-2  # gates below this are considered "pruned"


# Custom Prunable Layer

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a learnable gate on every weight.

    Each weight w_ij is multiplied by sigmoid(g_ij) where g_ij is a trainable
    gate score. When the gate approaches 0, the weight is effectively removed.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # standard weight & bias, same init as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # the gate scores — one per weight, same shape
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        # kaiming uniform, matches PyTorch's nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # start gates at 2.0 (sigmoid(2) ≈ 0.88), gates begin "open" so the
        # network starts fully connected; only sparsity pressure (λ>0) prunes them
        nn.init.constant_(self.gate_scores, GATE_INIT_VALUE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns current gate values (detached, for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


# Network Architecture

class SelfPruningNet(nn.Module):
    """
    4-layer feed-forward network for CIFAR-10 using PrunableLinear layers.
    BatchNorm is included for better training stability and accuracy.
    """

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # 3072 = 3 * 32 * 32 (flattened CIFAR image)
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        return self.layers(x)


# Sparsity Utilities

def compute_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    L1 penalty on all gate values, normalized by total gate count.
    Using mean instead of sum so that lambda is independent of network size.
    """
    gate_sum = torch.tensor(0.0, device=next(model.parameters()).device)
    gate_count = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gate_sum = gate_sum + torch.sigmoid(module.gate_scores).sum()
            gate_count += module.gate_scores.numel()
    return gate_sum / gate_count if gate_count > 0 else gate_sum


def calculate_sparsity(model: nn.Module, threshold: float = PRUNE_THRESHOLD) -> float:
    """Percentage of gates below the pruning threshold (globally)."""
    total, pruned = 0, 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return 100.0 * pruned / total if total > 0 else 0.0


def per_layer_sparsity(model: nn.Module, threshold: float = PRUNE_THRESHOLD) -> List[Dict]:
    """Sparsity breakdown for each PrunableLinear layer."""
    results = []
    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            gates = module.get_gates()
            n_total = gates.numel()
            n_pruned = (gates < threshold).sum().item()
            results.append({
                'name': name,
                'shape': f'{module.in_features}→{module.out_features}',
                'total': n_total,
                'pruned': n_pruned,
                'sparsity': 100.0 * n_pruned / n_total,
            })
    return results


def collect_all_gate_values(model: nn.Module) -> np.ndarray:
    """Flatten all gate values into a single numpy array for plotting."""
    gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates.append(module.get_gates().cpu().numpy().ravel())
    return np.concatenate(gates)


# Gradient Verification

def verify_gradient_flow(model: nn.Module, device: torch.device):
    """
    Quick sanity check that gradients actually reach gate_scores.
    Run this once before training to catch bugs early.
    """
    model.train()
    dummy = torch.randn(4, 3, 32, 32, device=device)
    out = model(dummy)
    out.sum().backward()

    print("\nGradient Flow Check")
    all_ok = True
    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            if module.gate_scores.grad is None or module.gate_scores.grad.abs().sum() == 0:
                print(f"  ✗ {name}: NO gradient on gate_scores!")
                all_ok = False
            else:
                grad_norm = module.gate_scores.grad.norm().item()
                print(f"  ✓ {name}: grad norm = {grad_norm:.4f}")

    if all_ok:
        print("  All layers OK — gates will be updated during training.\n")
    else:
        raise RuntimeError("Gradient flow broken! Check PrunableLinear implementation.")

    model.zero_grad()


# Data Loading

def get_data_loaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    # num_workers=0 avoids multiprocessing issues on MPS
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    return train_loader, test_loader


# Training & Evaluation

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_sparse: float,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy %)."""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        cls_loss = F.cross_entropy(outputs, targets)

        # sparsity regularization — skip if lambda is 0 (baseline run)
        if lambda_sparse > 0:
            sp_loss = compute_sparsity_loss(model)
            loss = cls_loss + lambda_sparse * sp_loss
        else:
            loss = cls_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Returns test accuracy %."""
    model.eval()
    correct, total = 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def hard_prune_and_eval(model: nn.Module, loader: DataLoader, device: torch.device,
                        threshold: float = PRUNE_THRESHOLD) -> float:
    """
    Zero out weights whose gates are below threshold, then re-evaluate.
    Simulates what you'd actually deploy — no gates, just a smaller weight matrix.
    """
    model_copy = type(model)().to(device)
    model_copy.load_state_dict(model.state_dict())

    with torch.no_grad():
        for module in model_copy.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                mask = (gates >= threshold).float()
                module.weight.data *= mask
                # set pruned gate scores to a very negative value
                module.gate_scores.data[gates < threshold] = -20.0

    return evaluate(model_copy, loader, device)


def train_model(
    lambda_sparse: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> Dict:
    """
    Full training pipeline for one lambda value.
    Returns a dict with the model, metrics history, and timing.
    """
    model = SelfPruningNet().to(device)
    # separate param groups: gates get a MUCH higher learning rate so they
    # respond meaningfully to the sparsity loss within a reasonable epoch count.
    # sigmoid'(0)=0.25 dampens gradients, so gates need aggressive LR.
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gate_params.append(param)
        else:
            other_params.append(param)

    gate_lr = lr * 50  # 50x LR for gates — high enough to prune, low enough to avoid baseline drift
    optimizer = optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': gate_params, 'lr': gate_lr},
    ], weight_decay=0)
    # Only schedule the weight params LR — keep gate LR constant so
    # pruning pressure stays strong throughout training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Lock gate LR: after scheduler init, we'll manually reset gate LR each step

    # verify gradient flow on the very first run
    verify_gradient_flow(model, device)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'sparsity': []}
    label = f"λ={lambda_sparse}" if lambda_sparse > 0 else "λ=0 (baseline)"
    print(f"\n{'='*60}")
    print(f" Training with {label}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, lambda_sparse)
        test_acc = evaluate(model, test_loader, device)
        sparsity = calculate_sparsity(model)
        scheduler.step()
        # restore gate LR — scheduler would decay it, but we want it constant
        optimizer.param_groups[1]['lr'] = gate_lr

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['sparsity'].append(sparsity)

        # print every 5 epochs + first and last
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"Loss: {train_loss:.4f}  "
                  f"Train: {train_acc:.1f}%  "
                  f"Test: {test_acc:.1f}%  "
                  f"Sparsity: {sparsity:.1f}%")

    elapsed = time.time() - start_time

    # hard prune and re-evaluate
    hard_pruned_acc = hard_prune_and_eval(model, test_loader, device)

    # per-layer breakdown
    layer_sparsity = per_layer_sparsity(model)

    print(f"\n  Done in {elapsed:.0f}s — "
          f"Final acc: {test_acc:.1f}% | "
          f"After hard prune: {hard_pruned_acc:.1f}% | "
          f"Sparsity: {sparsity:.1f}%")

    # show per-layer breakdown
    print(f"\n  {'Layer':<20} {'Shape':<15} {'Sparsity':>10}")
    print(f"  {'─'*47}")
    for info in layer_sparsity:
        print(f"  {info['name']:<20} {info['shape']:<15} {info['sparsity']:>9.1f}%")

    return {
        'lambda': lambda_sparse,
        'model': model,
        'history': history,
        'final_test_acc': test_acc,
        'hard_pruned_acc': hard_pruned_acc,
        'sparsity': sparsity,
        'layer_sparsity': layer_sparsity,
        'elapsed': elapsed,
    }


# Visualization

def plot_gate_distribution(gate_values: np.ndarray, save_path: str,
                           lambda_val: float, sparsity: float):
    """Histogram of gate values — should show bimodal pattern if pruning worked."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(gate_values, bins=100, color='#1976D2', alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.set_yscale('log')

    title = "Gate Value Distribution"
    if lambda_val > 0:
        title += f" (λ = {lambda_val})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Gate Value  [sigmoid(gate_score)]", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_xlim(-0.05, 1.05)

    ax.annotate(f"Sparsity: {sparsity:.1f}%\n(gates < {PRUNE_THRESHOLD})",
                xy=(0.72, 0.92), xycoords='axes fraction', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_lambda_comparison(results: List[Dict], save_path: str):
    """Side-by-side gate histograms for each lambda."""
    # skip baseline (lambda=0) — its gate distribution is uninteresting
    pruned_results = [r for r in results if r['lambda'] > 0]
    n = len(pruned_results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, pruned_results):
        gates = collect_all_gate_values(res['model'])
        ax.hist(gates, bins=100, color='#1976D2', alpha=0.85,
                edgecolor='white', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_title(f"λ = {res['lambda']}\n"
                     f"Acc: {res['final_test_acc']:.1f}%  |  "
                     f"Sparsity: {res['sparsity']:.1f}%",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Gate Value")
        ax.set_xlim(-0.05, 1.05)

    axes[0].set_ylabel("Count (log scale)")
    fig.suptitle("Gate Distributions Across λ Values",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(results: List[Dict], save_path: str):
    """Loss and accuracy over epochs for all lambda values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#424242', '#1976D2', '#F57C00', '#D32F2F', '#388E3C']

    for i, res in enumerate(results):
        lbl = f"λ={res['lambda']}" if res['lambda'] > 0 else "baseline (λ=0)"
        c = colors[i % len(colors)]
        epochs = range(1, len(res['history']['train_loss']) + 1)

        ax1.plot(epochs, res['history']['train_loss'], color=c, linewidth=1.5, label=lbl)
        ax2.plot(epochs, res['history']['test_acc'], color=c, linewidth=1.5, label=lbl)

    ax1.set_title("Training Loss", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Test Accuracy", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_vs_sparsity(results: List[Dict], save_path: str):
    """The money plot — shows the accuracy/sparsity trade-off curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sparsities = [r['sparsity'] for r in results]
    accuracies = [r['final_test_acc'] for r in results]
    labels = [f"λ={r['lambda']}" if r['lambda'] > 0 else "baseline" for r in results]

    ax.plot(sparsities, accuracies, 'o-', color='#1976D2', linewidth=2,
            markersize=10, markerfacecolor='white', markeredgewidth=2)

    for s, a, lbl in zip(sparsities, accuracies, labels):
        ax.annotate(lbl, (s, a), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

    ax.set_title("Accuracy vs Sparsity Trade-off", fontsize=14, fontweight='bold')
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# Results Summary

def print_results_table(results: List[Dict]):
    """Print a clean results summary to stdout."""
    baseline = next((r for r in results if r['lambda'] == 0), None)

    print(f"\n{'='*75}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*75}")
    print(f" {'Lambda':<12} {'Test Acc':>10} {'Hard Prune':>12} "
          f"{'Sparsity':>10} {'vs Base':>10} {'Time':>8}")
    print(f" {'─'*70}")

    for r in results:
        lam = "0 (base)" if r['lambda'] == 0 else f"{r['lambda']}"
        delta = ""
        if baseline and r['lambda'] != 0:
            diff = r['final_test_acc'] - baseline['final_test_acc']
            delta = f"{diff:+.1f}%"

        print(f" {lam:<12} {r['final_test_acc']:>9.1f}% {r['hard_pruned_acc']:>11.1f}% "
              f"{r['sparsity']:>9.1f}% {delta:>10} {r['elapsed']:>6.0f}s")

    print()


# Main

def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network — CIFAR-10")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambdas', type=float, nargs='+',
                        default=[0, 1.0, 5.0, 20.0],
                        help='Lambda values to test (0 = baseline)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Lambda values: {args.lambdas}")

    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)

    # run experiments for each lambda
    all_results = []
    for lam in args.lambdas:
        set_seed(args.seed)  # same init for fair comparison
        result = train_model(
            lambda_sparse=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
        all_results.append(result)

    # results table
    print_results_table(all_results)

    # find the "best" pruned model — pick the one with the best sparsity/accuracy balance
    pruned = [r for r in all_results if r['lambda'] > 0 and r['sparsity'] > 20]
    if not pruned:
        # fallback: just use whatever has the highest lambda
        pruned = [r for r in all_results if r['lambda'] > 0] or all_results
    best = max(pruned, key=lambda r: r['sparsity'] - (100 - r['final_test_acc']))

    # generate all plots
    print("\nGenerating plots...")

    gate_vals = collect_all_gate_values(best['model'])
    plot_gate_distribution(
        gate_vals, os.path.join(args.output_dir, 'gate_distribution.png'),
        best['lambda'], best['sparsity'])

    plot_lambda_comparison(
        all_results, os.path.join(args.output_dir, 'lambda_comparison.png'))

    plot_training_curves(
        all_results, os.path.join(args.output_dir, 'training_curves.png'))

    plot_accuracy_vs_sparsity(
        all_results, os.path.join(args.output_dir, 'accuracy_vs_sparsity.png'))

    print(f"\nAll done. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
