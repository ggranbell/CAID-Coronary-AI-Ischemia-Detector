"""
inference.py — Run predictions on unseen images using the trained ConvNeXt V2 model.

Supports:
  Single image  : python scripts/inference.py --image path/to/image.png
  Multiple images: python scripts/inference.py --image img1.png img2.png img3.png
  Entire folder : python scripts/inference.py --folder path/to/folder/
  Custom model  : any of the above + --checkpoint path/to/model.pth

Outputs:
  Console       : prediction table with class probabilities for every image
  Folder mode   : inference_results.csv  (all predictions)
                  inference_grid.png     (visual montage with overlaid labels)
"""
import argparse
import csv
import math
import sys
import textwrap
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.config import Config
from src.model import get_model


# ── Supported image extensions ─────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Transform (same as val/test, no augmentation) ─────────────────────────────

def build_inference_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, cfg: Config):
    print(f"[inference] Loading checkpoint: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run `python scripts/train.py` first."
        )
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)

    # If the checkpoint stored a saved config, use it for model construction
    saved_cfg_dict = ckpt.get("config", {})
    for key, val in saved_cfg_dict.items():
        if hasattr(cfg, key) and key not in ("device", "output_dir", "tuning_dir"):
            try:
                setattr(cfg, key, val)
            except Exception:
                pass

    model = get_model(cfg)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        # torch.compile wraps under _orig_mod
        model._orig_mod.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Single-image prediction ───────────────────────────────────────────────────

def predict_image(
    image_path: str | Path,
    model,
    transform: T.Compose,
    cfg: Config,
) -> dict:
    """
    Run inference on one image.

    Returns:
        {
          "filename": str,
          "predicted_class": str,
          "confidence": float   (0–100),
          "probabilities": {class_name: float, ...}
        }
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(cfg.device)  # (1, C, H, W)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=cfg.device,
            enabled=cfg.use_amp and cfg.device == "cuda",
        ):
            logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = int(torch.argmax(torch.tensor(probs)).item())
    return {
        "filename": image_path.name,
        "filepath": str(image_path),
        "predicted_class": cfg.classes[pred_idx],
        "confidence": probs[pred_idx] * 100,
        "probabilities": {cls: prob * 100 for cls, prob in zip(cfg.classes, probs)},
    }


# ── Collect image paths ───────────────────────────────────────────────────────

def collect_image_paths(images: list[str] | None, folder: str | None) -> list[Path]:
    paths: list[Path] = []
    if images:
        for p in images:
            path = Path(p)
            if not path.exists():
                print(f"[inference] ⚠ File not found, skipping: {p}")
            elif path.suffix.lower() not in IMAGE_EXTENSIONS:
                print(f"[inference] ⚠ Unsupported format, skipping: {p}")
            else:
                paths.append(path)
    if folder:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"[inference] ⚠ Folder not found: {folder}")
        else:
            # Use rglob to find images in subfolders (e.g. dataset/test/Negative/)
            found = sorted(
                p for p in folder_path.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            print(f"[inference] Found {len(found)} image(s) in {folder_path}")
            paths.extend(found)
    return paths


# ── Console table ──────────────────────────────────────────────────────────────

def print_results_table(results: list[dict], classes: list[str]):
    col_pred  = max(25, max(len(r["predicted_class"]) for r in results) + 2)

    print("\n" + "═" * (col_pred + 46))
    print("INFERENCE RESULTS")
    print("═" * (col_pred + 46))
    prob_headers = "  ".join(f"{cls[:15]:>15}" for cls in classes)
    print(f"{'#':<6} {'Prediction':<{col_pred}} {'Confidence':>10}  {prob_headers}")
    print("─" * (col_pred + 46))
    for i, r in enumerate(results, 1):
        prob_vals = "  ".join(
            f"{r['probabilities'][cls]:>14.2f}%" for cls in classes
        )
        print(
            f"{i:<6} "
            f"{r['predicted_class']:<{col_pred}} "
            f"{r['confidence']:>9.2f}%  "
            f"{prob_vals}"
        )
    print("═" * (col_pred + 46))


# ── CSV export ─────────────────────────────────────────────────────────────────

def save_csv(results: list[dict], classes: list[str], out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename", "filepath", "predicted_class", "confidence_%"] + \
                 [f"{cls}_prob_%" for cls in classes]
        writer.writerow(header)
        for r in results:
            row = [
                r["filename"],
                r["filepath"],
                r["predicted_class"],
                f"{r['confidence']:.4f}",
            ] + [f"{r['probabilities'][cls]:.4f}" for cls in classes]
            writer.writerow(row)
    print(f"[inference] CSV saved → {out_path}")


# ── Visual grid ───────────────────────────────────────────────────────────────

# Distinct colors per class for the label border
CLASS_COLORS = ["#4C72B0", "#C44E52"]

def save_grid(results: list[dict], classes: list[str], out_path: Path,
              cols: int = 4, thumb_size: int = 224):
    """
    Creates a grid image showing each input image with its prediction label
    and confidence bar overlaid.
    """
    n = len(results)
    if n == 0:
        return

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = [axes] if rows == 1 else [[ax] for ax in axes]

    class_color_map = {cls: CLASS_COLORS[i % len(CLASS_COLORS)] for i, cls in enumerate(classes)}

    for idx, result in enumerate(results):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        img = Image.open(result["filepath"]).convert("RGB")
        img.thumbnail((thumb_size, thumb_size))
        ax.imshow(img)

        pred_cls   = result["predicted_class"]
        confidence = result["confidence"]
        color      = class_color_map.get(pred_cls, "#999999")
        label_text = f"{pred_cls}\n{confidence:.1f}%"

        ax.set_title(
            textwrap.fill(label_text, width=20),
            fontsize=8, color="white",
            bbox=dict(facecolor=color, alpha=0.85, boxstyle="round,pad=0.3"),
        )

        # Thin colored border around image
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        # Confidence bar at bottom of image
        bar_width = confidence / 100
        ax.add_patch(mpatches.FancyArrowPatch(
            (0, 0), (bar_width, 0),
            arrowstyle="-",
            color=color, linewidth=5,
            transform=ax.transAxes, clip_on=False,
        ))

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=color, label=cls)
        for cls, color in class_color_map.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(classes), fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Inference Results", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[inference] Grid image saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run ConvNeXt V2 ischemia classifier on unseen images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python scripts/inference.py --image sample.png
          python scripts/inference.py --image img1.png img2.png img3.png
          python scripts/inference.py --folder my_test_images/
          python scripts/inference.py --folder my_test_images/ --checkpoint tuning_results/phase_3/trial_2/best_model.pth
        """),
    )
    parser.add_argument("--image", nargs="+", metavar="PATH",
                        help="One or more image file paths")
    parser.add_argument("--folder", metavar="DIR", default="dataset/test",
                        help="Folder of images to classify (default: dataset/test)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (default: outputs/best_model.pth)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Where to save CSV and grid image (default: outputs/)")
    parser.add_argument("--grid-cols", type=int, default=4,
                        help="Number of columns in the result grid image (default: 4)")
    args = parser.parse_args()

    cfg = Config()
    cfg.output_dir = args.output_dir
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint or cfg.checkpoint_path
    model = load_model(checkpoint, cfg)

    transform = build_inference_transform(cfg.image_size)

    # ── Collect images ─────────────────────────────────────────────────────────
    paths = collect_image_paths(args.image, args.folder)
    if not paths:
        print("[inference] No valid images found. Exiting.")
        return

    print(f"[inference] Running inference on {len(paths)} image(s)…")

    # ── Predict ────────────────────────────────────────────────────────────────
    results = []
    for path in paths:
        try:
            result = predict_image(path, model, transform, cfg)
            results.append(result)
        except Exception as e:
            print(f"[inference] ⚠ Could not process {path.name}: {e}")

    if not results:
        print("[inference] No images were successfully processed.")
        return

    # ── Console output ─────────────────────────────────────────────────────────
    print_results_table(results, cfg.classes)

    # ── Save CSV (always) ──────────────────────────────────────────────────────
    csv_path = Path(cfg.output_dir) / "inference_results.csv"
    save_csv(results, cfg.classes, csv_path)

    # ── Save visual grid ───────────────────────────────────────────────────────
    grid_path = Path(cfg.output_dir) / "inference_grid.png"
    save_grid(results, cfg.classes, grid_path, cols=args.grid_cols)


if __name__ == "__main__":
    main()
