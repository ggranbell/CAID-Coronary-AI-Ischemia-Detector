"""
train.py — Full GPU-accelerated training loop for ConvNeXt V2 ischemia classifier.

Usage:
  python scripts/train.py                        # Single training run (seed from config)
  python scripts/train.py --seeds 42,123,456,789,1234  # 5-seed multi-run (mean ± std)
  python scripts/train.py --debug                # CPU smoke test: 1 epoch, 2 batches

Outputs saved to outputs/
  best_model.pth         — best checkpoint (by val accuracy)
  training_history.json  — epoch-by-epoch loss & accuracy
  training_curves.png    — plotted training curves
  multi_seed_results.json — per-seed + summary stats (multi-seed mode only)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from src.config import Config
from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import EarlyStopping, FocalLoss, save_history, plot_training_history, set_seed

# ── Per-epoch pass ────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scaler,
    device: str,
    use_amp: bool,
    gradient_clip: float,
    is_train: bool,
) -> tuple[float, float]:
    """Runs one epoch of training or validation. Returns (avg_loss, accuracy%)."""
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"{'Train' if is_train else 'Val'}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            # Update progress bar
            avg_loss = total_loss / total
            avg_acc  = 100.0 * correct / total
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")

    return total_loss / total, 100.0 * correct / total

def train_one_seed(cfg: Config, debug: bool = False) -> dict:
    """
    Runs one full training for the given config + seed.
    Returns a dict with val_acc, best_epoch, total_epochs_run.
    """
    set_seed(cfg.seed)
    print(f"\n{'─'*60}")
    print(f"[train] Seed={cfg.seed}  Device={cfg.device}  "
          f"Model={cfg.model_name}@{cfg.image_size}  Epochs={cfg.epochs}")
    print(f"{'─'*60}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(cfg, debug=debug)

    # Model
    model = get_model(cfg)

    # Criterion
    class_weights = None
    if cfg.use_class_weights:
        ds = train_loader.dataset
        if hasattr(ds, "dataset"):
            ds = ds.dataset
        counts = [0] * cfg.num_classes
        for _, label in ds.samples:
            counts[label] += 1
        total_samples = sum(counts)
        class_weights = torch.tensor(
            [total_samples / (cfg.num_classes * c) for c in counts],
            dtype=torch.float32,
        ).to(cfg.device)

    if cfg.use_focal_loss:
        criterion = FocalLoss(
            gamma=cfg.focal_loss_gamma,
            weight=class_weights,
            label_smoothing=cfg.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing,
            weight=class_weights,
        )

    # Optimizer (differential LR)
    underlying = getattr(model, "_orig_mod", model)
    optimizer = torch.optim.AdamW(
        underlying.get_param_groups(cfg.lr, cfg.backbone_lr_multiplier),
        weight_decay=cfg.weight_decay,
    )

    # Scheduler
    if cfg.warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=cfg.eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[cfg.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
        )

    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)
    early_stopping = EarlyStopping(patience=cfg.early_stop_patience)

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr": [],
    }
    best_val_acc = 0.0
    best_epoch   = 0
    history_path = Path(cfg.output_dir) / "training_history.json"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            cfg.device, cfg.use_amp, cfg.gradient_clip, is_train=True,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            cfg.device, cfg.use_amp, cfg.gradient_clip, is_train=False,
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:>3}/{cfg.epochs} "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}% "
            f"| LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": cfg.__dict__,
            }, cfg.checkpoint_path)
            print(f"    ✓ Best model saved (val_acc={val_acc:.2f}%)")

        save_history(history, str(history_path))

        if early_stopping(val_loss):
            print(f"\n[train] Early stopping at epoch {epoch}.")
            break

    # Final training curve plot
    plot_training_history(str(history_path), cfg.output_dir)

    return {
        "seed": cfg.seed,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs_run": len(history["train_loss"]),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt V2 ischemia classifier")
    parser.add_argument("--debug", action="store_true",
                        help="Quick CPU smoke test: 1 epoch, 2 batches")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed training, e.g. 42,123,456,789,1234")
    args = parser.parse_args()

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [Config().seed]   # single default seed

    print(f"[train] Seeds to run: {seeds}")

    all_results = []
    for seed in seeds:
        cfg = Config()
        cfg.seed = seed
        if args.debug:
            cfg.epochs = 1
            cfg.num_workers = 0
            cfg.use_amp = False
            cfg.device = "cpu"
        # Each seed saves its best checkpoint under a seed-specific name
        if len(seeds) > 1:
            cfg.checkpoint_name = f"best_model_seed{seed}.pth"

        result = train_one_seed(cfg, debug=args.debug)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"MULTI-SEED RESULTS  ({len(seeds)} seed{'s' if len(seeds)>1 else ''})")
    print(f"{'═'*60}")
    print(f"{'Seed':<10} {'Val Acc':>10} {'Best Epoch':>12} {'Epochs Run':>12}")
    print(f"{'─'*60}")
    for r in all_results:
        print(f"{r['seed']:<10} {r['best_val_acc']:>9.2f}% {r['best_epoch']:>12} {r['total_epochs_run']:>12}")

    if len(seeds) > 1:
        accs = [r["best_val_acc"] for r in all_results]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        print(f"{'─'*60}")
        print(f"{'Mean':<10}    {mean_acc:>9.2f}%  ±{std_acc:.2f}%")

        summary = {
            "seeds": seeds,
            "per_seed": all_results,
            "mean_val_acc": mean_acc,
            "std_val_acc":  std_acc,
            "report": f"{mean_acc:.2f} ± {std_acc:.2f}%",
        }
        out_path = Path(Config().output_dir) / "multi_seed_results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[train] Multi-seed summary saved → {out_path}")
        print(f"[train] Final result to report in paper: Val Acc = {summary['report']}")


if __name__ == "__main__":
    main()
