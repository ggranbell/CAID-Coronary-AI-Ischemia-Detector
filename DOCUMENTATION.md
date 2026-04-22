# CAID — Complete Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Inference](#inference)
8. [Academic Analysis](#academic-analysis)
9. [Web Application](#web-application)
10. [Configuration Reference](#configuration-reference)

---

## Overview

CAID (Coronary AI Ischemia Detector) is a binary classification system that analyzes coronary angiography images to detect myocardial ischemia. The system uses a **ConvNeXt V2** backbone pretrained on ImageNet-1K, fine-tuned with a custom classification head for binary output (Positive/Negative).

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backbone | ConvNeXt V2 (timm) | State-of-the-art CNN with Global Response Normalization; strong on medical imaging |
| Output | 2-class softmax | Consistent with multi-class framework; enables direct ROC/AUC analysis |
| Loss | CrossEntropyLoss + Label Smoothing | Reduces overconfidence; compatible with Focal Loss toggle |
| Augmentation | Constrained (shear + rotation) | Domain-appropriate for angiography — avoids destructive transforms like color jitter |
| Resolution | 384×384 | Optimal balance between detail capture and GPU memory |

---

## Dataset

### Structure
```
dataset/
├── train/
│   ├── Negative/    # No ischemia
│   └── Positive/    # Ischemia detected
├── val/
│   ├── Negative/
│   └── Positive/
└── test/
    ├── Negative/
    └── Positive/
```

### Split Ratio
- **Training**: 70% (~7,112 images)
- **Validation**: 10%
- **Testing**: 20%

### Manifest Files (`data/`)
Each `.txt` file lists relative paths: `<class>/filename.png`

---

## Model Architecture

```
ConvNeXt V2 Base (pretrained ImageNet-1K)
    → Global Average Pooling (built-in to timm backbone)
    → LayerNorm(1024)
    → Dropout(0.3)
    → Linear(1024, 2)
```

- **Total Parameters**: ~89M
- **Trainable Parameters**: ~89M (full fine-tuning by default)
- **Differential LR** supported: backbone_lr = lr × multiplier

---

## Training Pipeline

### Optimizer & Scheduler
- **Optimizer**: AdamW with weight decay 1e-2
- **Scheduler**: Cosine Annealing (optional linear warmup)
- **Mixed Precision**: AMP with GradScaler for RTX GPUs
- **Gradient Clipping**: max_norm = 1.0

### Early Stopping
- Monitors validation loss
- Patience: 10 epochs
- Minimum delta: 1e-4

### Multi-Seed Training
```bash
python scripts/train.py --seeds 42,123,456,789,1234
```
Reports: mean ± std validation accuracy across seeds.

---

## Evaluation

Outputs:
- **Classification Report**: Per-class precision, recall, F1-score
- **Confusion Matrix**: 2×2 heatmap saved as PNG
- **Predictions CSV**: Per-image ground truth vs. prediction
- **Training Curves**: Loss and accuracy plots

---

## Hyperparameter Tuning

8-phase grid search strategy:

| Phase | Parameters | Trials |
|-------|-----------|--------|
| 1 | Learning Rate | 5 |
| 2 | Batch Size + Weight Decay | 6 |
| 3 | Dropout + Label Smoothing | 5 |
| 4 | Model Variant + Resolution | 3 |
| 5 | LR Scheduler (Warmup + eta_min) | 4 |
| 6 | Loss Function (Focal + Class Weights) | 5 |
| 7 | Differential LR | 3 |
| 8 | Augmentation Probability | 4 |

Each phase uses the best values from all previous phases (cascading optimization).

---

## Inference

### CLI
```bash
python scripts/inference.py --image coronary_001.png
python scripts/inference.py --folder test_images/ --grid-cols 5
```

### Web UI
```bash
python app.py  # → http://localhost:5000
```

---

## Academic Analysis

| Analysis | Output | Purpose |
|----------|--------|---------|
| ROC/AUC | `roc_curves.png`, `roc_auc_scores.json` | Classification performance at all thresholds |
| t-SNE | `tsne_features.png` | Feature space visualization (should show 2 clusters) |
| Cost | `cost_report.txt`, `cost_report.json` | Params, FLOPs, inference speed |

---

## Web Application

Flask-based interface with:
- Drag-and-drop image upload
- Real-time classification with probability bars
- Grad-CAM heatmap overlay
- Dark theme with glassmorphism design

---

## Configuration Reference

All hyperparameters are centralized in `src/config.py`. Key fields:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `convnextv2_base` | timm model identifier |
| `image_size` | `384` | Input resolution |
| `batch_size` | `16` | Training batch size |
| `lr` | `1e-4` | Initial learning rate |
| `epochs` | `50` | Maximum training epochs |
| `dropout` | `0.3` | Dropout before classifier head |
| `label_smoothing` | `0.1` | Cross-entropy label smoothing |
| `early_stop_patience` | `10` | Epochs without improvement before stopping |
| `seed` | `42` | Random seed for reproducibility |
