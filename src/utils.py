"""
utils.py — Shared utilities: early stopping, training curve plotting, seeding, focal loss.
"""
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy/confident examples and focuses training on
    hard, misclassified ones. Especially useful when classes are imbalanced.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma  : focusing parameter. 0 = standard cross-entropy.
                 Typical values: 1.0 (mild focus) to 2.0 (strong focus).
        weight : per-class weights tensor (class_weights). Optional.
        label_smoothing: same semantics as nn.CrossEntropyLoss.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE loss per sample (no reduction)
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t = probability assigned to the true class
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Args:
        patience : number of epochs to wait before stopping
        delta    : minimum change to count as improvement
        verbose  : print messages
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epoch(s)")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

# ── History I/O ───────────────────────────────────────────────────────────────
def save_history(history: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[utils] Training history saved → {path}")


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def plot_training_history(history_path: str, output_dir: str = "outputs") -> None:
    """
    Reads training history JSON and saves loss/accuracy plots.

    Args:
        history_path : path to the JSON file written by train.py
        output_dir   : directory to save the PNG plots
    """
    if not Path(history_path).exists():
        print(f"[plot] History file not found: {history_path}")
        return

    history = load_history(history_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="o", markersize=3)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   marker="o", markersize=3)
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"[plot] Training curves saved → {out_path}")
