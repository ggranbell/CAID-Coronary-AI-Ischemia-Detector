"""
evaluate.py — Evaluation script for the trained ConvNeXt V2 ischemia classifier.

Loads the best checkpoint and runs inference on the test set.
Outputs:
  - Classification report (precision, recall, F1 per class) printed to console
  - Confusion matrix heatmap saved to outputs/confusion_matrix.png
  - Per-image predictions saved to outputs/predictions.csv
  - Training curves plot saved to outputs/training_curves.png (if history exists)

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --checkpoint outputs/best_model.pth
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config
from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import plot_training_history


def evaluate(cfg: Config, checkpoint_path: str) -> None:
    print(f"\n[evaluate] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)

    model = get_model(cfg)
    # torch.compile wraps the underlying module, handle both cases
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        # Compiled model stores weights under _orig_mod
        model._orig_mod.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    _, _, test_loader = get_dataloaders(cfg)
    print(f"[evaluate] Test batches: {len(test_loader)}")

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(cfg.device, non_blocking=True)
            with torch.amp.autocast(device_type=cfg.device, enabled=cfg.use_amp and cfg.device == "cuda"):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Classification report ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("CLASSIFICATION REPORT")
    print("═" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=cfg.classes,
        digits=4,
    ))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=cfg.classes,
        yticklabels=cfg.classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.tight_layout()

    cm_path = Path(cfg.output_dir) / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {cm_path}")

    # ── Predictions CSV ───────────────────────────────────────────────────────
    csv_path = Path(cfg.output_dir) / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "predicted_label",
                         "true_class", "predicted_class", "correct"])
        for i, (true, pred) in enumerate(zip(all_labels, all_preds)):
            writer.writerow([
                i,
                int(true), int(pred),
                cfg.classes[int(true)], cfg.classes[int(pred)],
                int(true) == int(pred),
            ])
    print(f"[evaluate] Predictions saved → {csv_path}")

    # ── Overall accuracy ──────────────────────────────────────────────────────
    accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\n[evaluate] Overall test accuracy: {accuracy:.2f}%")

    # ── Training curves (if available) ────────────────────────────────────────
    history_path = Path(cfg.output_dir) / "training_history.json"
    if history_path.exists():
        plot_training_history(str(history_path), cfg.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt V2 ischemia classifier")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint (default: outputs/best_model.pth)"
    )
    args = parser.parse_args()

    cfg = Config()
    checkpoint = args.checkpoint or cfg.checkpoint_path

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run `python scripts/train.py` first to produce a trained model."
        )

    evaluate(cfg, checkpoint)


if __name__ == "__main__":
    main()
