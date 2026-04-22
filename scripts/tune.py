"""
tune.py — Phased hyperparameter grid search for ConvNeXt V2 ischemia classifier.

Runs full 50-epoch training (with early stopping) for each trial in a phase.
All scores, training curves and per-trial checkpoints are saved.

Usage:
  python scripts/tune.py --phase 1          # Learning Rate sweep
  python scripts/tune.py --phase 2          # Batch Size + Weight Decay
  python scripts/tune.py --phase 3          # Dropout + Label Smoothing
  python scripts/tune.py --phase 4          # Model Variant + Resolution
  python scripts/tune.py --phase 1 --dry-run  # Print trial configs, no training

Outputs per trial:
  tuning_results/phase_N/trial_X/best_model.pth
  tuning_results/phase_N/trial_X/training_history.json
  tuning_results/phase_N/trial_X/training_curves.png
  tuning_results/phase_N/trial_X/val_metrics.json

Summary per phase:
  tuning_results/phase_N/phase_summary.json
  tuning_results/phase_N/phase_comparison.png
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import Config
from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import EarlyStopping, set_seed

# ── Phase Grids ───────────────────────────────────────────────────────────────
def build_phase_grid(phase: int, best: dict) -> list[dict]:
    """
    Returns the list of hyperparameter overrides to try for a given phase.
    best: dict of best values found in previous phases.
    """
    lr               = best.get("lr", 1e-4)
    batch_size       = best.get("batch_size", 16)
    weight_decay     = best.get("weight_decay", 1e-2)
    dropout          = best.get("dropout", 0.3)
    label_smoothing  = best.get("label_smoothing", 0.1)
    warmup_epochs    = best.get("warmup_epochs", 0)
    eta_min          = best.get("eta_min", 1e-6)
    use_focal_loss   = best.get("use_focal_loss", False)
    focal_loss_gamma = best.get("focal_loss_gamma", 2.0)
    use_class_weights = best.get("use_class_weights", False)
    backbone_lr_multiplier = best.get("backbone_lr_multiplier", 1.0)
    gradient_clip    = best.get("gradient_clip", 1.0)
    aug_probability  = best.get("aug_probability", 0.7)

    base = dict(
        lr=lr, batch_size=batch_size, weight_decay=weight_decay,
        dropout=dropout, label_smoothing=label_smoothing,
        warmup_epochs=warmup_epochs, eta_min=eta_min,
        use_focal_loss=use_focal_loss, focal_loss_gamma=focal_loss_gamma,
        use_class_weights=use_class_weights,
        backbone_lr_multiplier=backbone_lr_multiplier,
        gradient_clip=gradient_clip, aug_probability=aug_probability,
    )

    if phase == 1:
        # Learning Rate
        return [
            {**base, "lr": 1e-5},
            {**base, "lr": 5e-5},
            {**base, "lr": 1e-4},   # baseline
            {**base, "lr": 3e-4},
            {**base, "lr": 5e-4},
        ]
    elif phase == 2:
        # Batch Size + Weight Decay
        return [
            {**base, "batch_size": 8,  "weight_decay": 1e-2},
            {**base, "batch_size": 8,  "weight_decay": 5e-2},
            {**base, "batch_size": 16, "weight_decay": 1e-2},  # baseline
            {**base, "batch_size": 16, "weight_decay": 5e-2},
            {**base, "batch_size": 32, "weight_decay": 1e-2},
            {**base, "batch_size": 32, "weight_decay": 5e-2},
        ]
    elif phase == 3:
        # Dropout + Label Smoothing
        return [
            {**base, "dropout": 0.2, "label_smoothing": 0.0},
            {**base, "dropout": 0.2, "label_smoothing": 0.1},
            {**base, "dropout": 0.3, "label_smoothing": 0.1},  # baseline
            {**base, "dropout": 0.5, "label_smoothing": 0.1},
            {**base, "dropout": 0.5, "label_smoothing": 0.2},
        ]
    elif phase == 4:
        # Model Variant + Resolution (ConvNeXt V2 variants)
        return [
            {**base, "model_name": "convnextv2_tiny",  "image_size": 224, "batch_size": 32},
            {**base, "model_name": "convnextv2_small", "image_size": 384, "batch_size": 16},
            {**base, "model_name": "convnextv2_base",  "image_size": 384, "batch_size": 16},  # baseline
        ]
    elif phase == 5:
        # LR Scheduler: warmup_epochs + eta_min
        return [
            {**base, "warmup_epochs": 0, "eta_min": 1e-6},   # baseline
            {**base, "warmup_epochs": 3, "eta_min": 1e-6},
            {**base, "warmup_epochs": 5, "eta_min": 1e-6},
            {**base, "warmup_epochs": 5, "eta_min": 1e-7},
        ]
    elif phase == 6:
        # Loss Function: Focal Loss + class weights
        return [
            {**base, "use_focal_loss": False, "focal_loss_gamma": 2.0, "use_class_weights": False},  # baseline
            {**base, "use_focal_loss": True,  "focal_loss_gamma": 1.0, "use_class_weights": False},
            {**base, "use_focal_loss": True,  "focal_loss_gamma": 2.0, "use_class_weights": False},
            {**base, "use_focal_loss": True,  "focal_loss_gamma": 2.0, "use_class_weights": True},
            {**base, "use_focal_loss": False, "focal_loss_gamma": 2.0, "use_class_weights": True},
        ]
    elif phase == 7:
        # Differential LR: backbone_lr_multiplier
        return [
            {**base, "backbone_lr_multiplier": 0.1},
            {**base, "backbone_lr_multiplier": 0.3},
            {**base, "backbone_lr_multiplier": 1.0},    # baseline
        ]
    elif phase == 8:
        # Augmentation probability
        return [
            {**base, "aug_probability": 0.3},
            {**base, "aug_probability": 0.5},
            {**base, "aug_probability": 0.7},  # baseline
            {**base, "aug_probability": 1.0},
        ]
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1–8.")


def load_best_from_phase(phase_dir: Path) -> dict:
    """Reads phase_summary.json and returns the best hyperparameter values."""
    summary_path = phase_dir / "phase_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        summary = json.load(f)
    return summary.get("best_config", {})

# ── Training Utilities ────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, is_train):
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def compute_val_metrics(model, loader, device, use_amp, num_classes):
    """Return val accuracy, macro F1, macro precision, macro recall."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    f1         = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    precision  = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall     = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    return accuracy, f1, precision, recall

# ── Per-trial saving ──────────────────────────────────────────────────────────
def save_trial_curves(history: dict, trial_dir: Path):
    """Save loss + accuracy training curves for a single trial."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="o", markersize=3)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   marker="o", markersize=3)
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(trial_dir / "training_curves.png"), dpi=150)
    plt.close()

def save_phase_comparison_chart(results: list[dict], phase: int, phase_dir: Path):
    """Bar chart comparing all trials in a phase by val accuracy, F1, precision, recall."""
    labels = [r["trial_name"] for r in results]
    val_acc   = [r["best_val_acc"]       for r in results]
    val_f1    = [r["best_val_f1"]        for r in results]
    val_prec  = [r["best_val_precision"] for r in results]
    val_rec   = [r["best_val_recall"]    for r in results]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    ax.bar(x - 1.5*width, val_acc,  width, label="Val Accuracy (%)", color="#4C72B0")
    ax.bar(x - 0.5*width, val_f1,   width, label="Val F1 Macro (%)", color="#DD8452")
    ax.bar(x + 0.5*width, val_prec, width, label="Val Precision (%)", color="#55A868")
    ax.bar(x + 1.5*width, val_rec,  width, label="Val Recall (%)",    color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Phase {phase} — Trial Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(str(phase_dir / "phase_comparison.png"), dpi=150)
    plt.close()
    print(f"[tune] Phase comparison chart saved → {phase_dir / 'phase_comparison.png'}")

def run_trial(
    trial_idx: int,
    overrides: dict,
    phase: int,
    phase_dir: Path,
) -> dict:
    """
    Trains one full trial (50 epochs, early stopping).
    Returns a dict with the trial's final metrics.
    """
    trial_name = f"trial_{trial_idx+1}"
    trial_dir  = phase_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Build config for this trial
    cfg = Config()
    cfg.epochs = 50
    cfg.early_stop_patience = 10
    cfg.output_dir = str(trial_dir)
    cfg.checkpoint_name = "best_model.pth"
    for key, val in overrides.items():
        setattr(cfg, key, val)
    trial_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    config_str = "  ".join(f"{k}={v}" for k, v in overrides.items())
    print(f"\n{'═'*70}")
    print(f"[tune] Phase {phase} | {trial_name} | {config_str}")
    print(f"{'═'*70}")

    train_loader, val_loader, _ = get_dataloaders(cfg)
    model = get_model(cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler(enabled=cfg.use_amp)
    early_stopping = EarlyStopping(patience=cfg.early_stop_patience)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg.device, cfg.use_amp, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, cfg.device, cfg.use_amp, is_train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{cfg.epochs} "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}% "
            f"| {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": overrides,
            }, cfg.checkpoint_path)
            print(f"  ✓ Best saved (val_acc={val_acc:.2f}%)")

        with open(trial_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if early_stopping(val_loss):
            print(f"  [EarlyStopping] Triggered at epoch {epoch}.")
            break

    # ── Compute final val metrics from best checkpoint ────────────────────────
    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        model._orig_mod.load_state_dict(ckpt["model_state_dict"])

    best_acc, best_f1, best_prec, best_rec = compute_val_metrics(
        model, val_loader, cfg.device, cfg.use_amp, cfg.num_classes
    )

    save_trial_curves(history, trial_dir)

    val_metrics = {
        "best_val_acc":       best_acc,
        "best_val_f1":        best_f1,
        "best_val_precision": best_prec,
        "best_val_recall":    best_rec,
        "best_epoch":         best_epoch,
        "total_epochs_run":   len(history["train_loss"]),
    }
    with open(trial_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    return {
        "trial_name":         trial_name,
        "hyperparams":        overrides,
        **val_metrics,
    }

# ── Phase runner ──────────────────────────────────────────────────────────────
def run_phase(phase: int, dry_run: bool = False):
    cfg = Config()
    base_tuning_dir = Path(cfg.tuning_dir)

    # Load best values from previous completed phases
    best = {}
    for prev_phase in range(1, phase):
        prev_dir  = base_tuning_dir / f"phase_{prev_phase}"
        prev_best = load_best_from_phase(prev_dir)
        best.update(prev_best)

    grid = build_phase_grid(phase, best)
    phase_dir = base_tuning_dir / f"phase_{phase}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# PHASE {phase}  —  {len(grid)} trials")
    print(f"{'#'*70}\n")

    if dry_run:
        print("[tune] DRY RUN — printing trial configs only:\n")
        for i, overrides in enumerate(grid):
            print(f"  trial_{i+1}: {overrides}")
        return

    results = []
    for i, overrides in enumerate(grid):
        try:
            result = run_trial(i, overrides, phase, phase_dir)
            results.append(result)
        except Exception as e:
            print(f"\n[tune] ⚠ Trial {i+1} failed: {e}")
            traceback.print_exc()
            results.append({
                "trial_name": f"trial_{i+1}",
                "hyperparams": overrides,
                "error": str(e),
                "best_val_acc": -1,
                "best_val_f1":  -1,
                "best_val_precision": -1,
                "best_val_recall":    -1,
                "best_epoch": -1,
                "total_epochs_run": -1,
            })

    # ── Rank and summarize ────────────────────────────────────────────────────
    valid_results = [r for r in results if r["best_val_acc"] >= 0]
    valid_results.sort(key=lambda r: r["best_val_acc"], reverse=True)

    print(f"\n{'─'*70}")
    print(f"PHASE {phase} RESULTS (ranked by val accuracy)")
    print(f"{'─'*70}")
    print(f"{'Rank':<5} {'Trial':<12} {'Val Acc':>9} {'F1':>8} {'Prec':>8} {'Recall':>8}  Config")
    print(f"{'─'*70}")
    for rank, r in enumerate(valid_results, 1):
        hp_str = "  ".join(f"{k}={v}" for k, v in r["hyperparams"].items())
        print(
            f"{rank:<5} {r['trial_name']:<12} "
            f"{r['best_val_acc']:>8.2f}% "
            f"{r['best_val_f1']:>7.2f}% "
            f"{r['best_val_precision']:>7.2f}% "
            f"{r['best_val_recall']:>7.2f}%  "
            f"{hp_str}"
        )

    best_config = valid_results[0]["hyperparams"] if valid_results else {}
    summary = {
        "phase": phase,
        "best_config": best_config,
        "ranked_results": valid_results,
    }
    with open(phase_dir / "phase_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[tune] Phase {phase} summary saved → {phase_dir / 'phase_summary.json'}")

    if valid_results:
        save_phase_comparison_chart(valid_results, phase, phase_dir)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phased hyperparameter tuning for ConvNeXt V2")
    parser.add_argument("--phase", type=int, required=True, choices=list(range(1, 9)),
                        help="Which phase to run (1=LR, 2=Batch+WD, 3=Dropout+LS, 4=Model+Res, 5=Sch, 6=Loss, 7=DiffLR, 8=AugProb)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print trial configs without training")
    args = parser.parse_args()

    run_phase(args.phase, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
