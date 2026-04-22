# 00 — Project Overview

## What is CAID?

**CAID (Coronary AI Ischemia Detector)** is a deep learning binary classification system that analyzes coronary angiography images to determine the presence or absence of myocardial ischemia.

## Clinical Context

Coronary artery disease (CAD) is the leading cause of death worldwide. Coronary Computed Tomography Angiography (CCTA) and invasive coronary angiography (ICA) are primary imaging modalities for diagnosing coronary stenosis. However, visual assessment of these images is subjective, time-consuming, and prone to inter-observer variability.

CAID aims to provide an objective, automated classification of coronary angiography images using modern deep learning architectures, specifically ConvNeXt V2 — a state-of-the-art convolutional neural network with Global Response Normalization (GRN) for improved feature learning.

## Pipeline Overview

```
Dataset (CCTA images)
    ↓
split_dataset.py → train.txt / val.txt / test.txt (70:20:10 split)
    ↓
train.py → best_model.pth (ConvNeXt V2, AMP, cosine annealing)
    ↓
evaluate.py → confusion matrix, classification report, predictions CSV
    ↓
analysis.py → ROC/AUC, t-SNE, computational cost
    ↓
inference.py / app.py → CLI inference or Web UI with Grad-CAM
```

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Train the model
python scripts/train.py

# 3. Evaluate on test set
python scripts/evaluate.py

# 4. Launch web UI
python app.py
```

## File map

| File | Purpose |
|------|---------|
| `src/config.py` | All hyperparameters and paths |
| `src/model.py` | ConvNeXt V2 + custom classification head |
| `src/dataset.py` | Image loading, augmentation, dataloaders |
| `src/utils.py` | Focal loss, early stopping, seeding |
| `src/visualize_cam.py` | Grad-CAM for explainability |
| `scripts/train.py` | Training loop with multi-seed support |
| `scripts/evaluate.py` | Evaluation and confusion matrix |
| `scripts/tune.py` | Phased hyperparameter search |
| `scripts/analysis.py` | Academic analysis (ROC, t-SNE, cost) |
| `scripts/inference.py` | CLI inference (single/batch/folder) |
| `app.py` | Flask web application |
