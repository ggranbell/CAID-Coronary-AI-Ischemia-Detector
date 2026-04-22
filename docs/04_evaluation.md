# 04 — Evaluation

## Evaluation Script

```bash
python scripts/evaluate.py                              # Default checkpoint
python scripts/evaluate.py --checkpoint path/to/model.pth  # Custom checkpoint
```

## Metrics

### Classification Report

Per-class and overall metrics computed on the held-out **test set**:

| Metric | Description |
|--------|-------------|
| **Precision** | Of all predicted positives, how many are actually positive? |
| **Recall (Sensitivity)** | Of all actual positives, how many are correctly identified? |
| **F1-Score** | Harmonic mean of precision and recall |
| **Support** | Number of samples per class |

### Confusion Matrix

A 2×2 heatmap saved to `outputs/confusion_matrix.png`:

```
                 Predicted
                 Neg    Pos
Actual  Neg  [  TN  |  FP  ]
        Pos  [  FN  |  TP  ]
```

In clinical context:
- **False Negative (FN)** = Missed ischemia → most dangerous
- **False Positive (FP)** = Unnecessary follow-up → less harmful but costly

### Per-Image Predictions

Saved to `outputs/predictions.csv`:

| Column | Type | Description |
|--------|------|-------------|
| index | int | Image index |
| true_label | int | Ground truth (0=Negative, 1=Positive) |
| predicted_label | int | Model prediction |
| true_class | str | Ground truth class name |
| predicted_class | str | Predicted class name |
| correct | bool | Whether prediction matches ground truth |

## Outputs

| File | Description |
|------|-------------|
| `outputs/confusion_matrix.png` | 2×2 confusion matrix heatmap |
| `outputs/predictions.csv` | Per-image predictions |
| `outputs/training_curves.png` | Loss/accuracy plots (if history exists) |
