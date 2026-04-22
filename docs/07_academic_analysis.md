# 07 — Academic Analysis

## Overview

The `scripts/analysis.py` script generates research-quality outputs suitable for academic papers and thesis documents.

```bash
python scripts/analysis.py --all                     # All analyses
python scripts/analysis.py --roc                     # ROC only
python scripts/analysis.py --tsne                    # t-SNE only
python scripts/analysis.py --cost                    # Cost only
python scripts/analysis.py --all --checkpoint path/to/model.pth  # Custom checkpoint
```

All outputs are saved to `outputs/analysis/`.

---

## ROC Curve & AUC

**Purpose**: Evaluate classification performance across all possible decision thresholds.

For binary classification, a single ROC curve is plotted using the **Positive class probability**:
- **x-axis**: False Positive Rate (1 - Specificity)
- **y-axis**: True Positive Rate (Sensitivity)
- **AUC**: Area Under the ROC Curve (1.0 = perfect classifier)

### Clinical Interpretation:
- **AUC > 0.95**: Excellent discriminative ability
- **AUC 0.90–0.95**: Very good
- **AUC 0.80–0.90**: Good
- **AUC < 0.80**: May need improvement for clinical deployment

### Outputs:
- `outputs/analysis/roc_curves.png` — ROC plot
- `outputs/analysis/roc_auc_scores.json` — Numeric AUC value

---

## t-SNE Feature Embeddings

**Purpose**: Visualize the learned feature space to assess class separability.

### Method:
1. Pass all test images through the ConvNeXt V2 backbone (before the classification head)
2. Extract the 1024-dimensional feature vectors
3. Apply t-SNE dimensionality reduction to 2D
4. Plot as colored scatter (Negative=blue, Positive=red)

### Expected Result:
- Well-separated clusters suggest the model has learned discriminative features
- Overlapping clusters suggest the model struggles to distinguish certain cases

### Parameters:
- **Perplexity**: 30 (default), adjustable via `--tsne-perplexity`
- **Iterations**: 1000
- **Initialization**: PCA

### Output:
- `outputs/analysis/tsne_features.png` — 2D scatter plot

---

## Computational Cost

**Purpose**: Report the model's resource footprint for comparison with other approaches.

### Metrics:
| Metric | Description |
|--------|-------------|
| **Total Parameters** | Number of learnable parameters |
| **Trainable Parameters** | Parameters updated during training |
| **FLOPs** | Floating-point operations per inference (requires `thop`) |
| **Inference Speed** | Images processed per second |

### Outputs:
- `outputs/analysis/cost_report.txt` — Human-readable report
- `outputs/analysis/cost_report.json` — Machine-readable data

---

## Multi-Seed Statistical Reporting

For academic rigor, train across multiple seeds (see [`03_training.md`](03_training.md)):

```bash
python scripts/train.py --seeds 42,123,456,789,1234
```

This produces `outputs/multi_seed_results.json` with:
```json
{
  "seeds": [42, 123, 456, 789, 1234],
  "mean_val_acc": 95.12,
  "std_val_acc": 0.32,
  "report": "95.12 ± 0.32%"
}
```

**Format for papers**: "Our model achieved a validation accuracy of 95.12 ± 0.32% across 5 independent training runs."

---

## Suggested Thesis Figures

| Figure | Script | Output |
|--------|--------|--------|
| Training curves (loss + accuracy) | `evaluate.py` | `training_curves.png` |
| Confusion matrix | `evaluate.py` | `confusion_matrix.png` |
| ROC curve + AUC | `analysis.py --roc` | `roc_curves.png` |
| t-SNE embeddings | `analysis.py --tsne` | `tsne_features.png` |
| Grad-CAM examples | `visualize_cam.py` | `outputs/cams/*.png` |
| Tuning comparison | `tune.py` | `phase_comparison.png` |
