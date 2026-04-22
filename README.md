<div align="center">
  <h1>🫀 CAID — Coronary AI Ischemia Detector</h1>
  <p>
    <b>An academic-grade computer vision system for automated detection of myocardial ischemia from coronary angiography images.</b>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/ConvNeXt_V2-timm-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="ConvNeXt V2" />
    <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
    <img src="https://img.shields.io/badge/Status-Research--Ready-2ECC71?style=for-the-badge" alt="Status" />
  </p>
</div>

---

## 📖 Overview

**CAID (Coronary AI Ischemia Detector)** is a deep learning project that leverages the **ConvNeXt V2** architecture with transfer learning to achieve high-accuracy binary classification of coronary angiography images as **Positive** (ischemia detected) or **Negative** (no ischemia). The system ships with a full scientific pipeline — from dataset preparation and model training, to evaluation, explainability (Grad-CAM), and a deployable Flask web interface — making it ready for both academic review and practical clinical screening use.

## ✨ Features

- 🏥 **Binary Ischemia Detection**: Classifies coronary angiography images as Positive or Negative for myocardial ischemia using a fine-tuned ConvNeXt V2 backbone.
- 💡 **Explainable AI (XAI)**: Integrated Grad-CAM visualization highlights the anatomical regions (coronary arteries, stenoses) the model relies on for each prediction.
- 🌐 **Interactive Web UI**: A modern, dark-themed Flask interface for drag-and-drop image upload, real-time classification, probability charts, and Grad-CAM heatmap display.
- 📊 **Academic Analysis Suite**: Built-in scripts for multi-seed statistical reporting, ROC/AUC curve generation, t-SNE feature embedding, and computational cost profiling.
- 🛡️ **Reproducible Science**: Fixed random seeds, stratified data splits, and manifest-based data loading ensure every experiment is fully reproducible.

## 🫀 Classification Target

| Class | Description |
|-------|-------------|
| **Negative** | No significant coronary ischemia detected |
| **Positive** | Coronary ischemia detected (stenosis, reduced perfusion) |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | [PyTorch](https://pytorch.org/) + [timm](https://github.com/huggingface/pytorch-image-models) (ConvNeXt V2 with ImageNet pretrained weights) |
| **Data & Evaluation** | NumPy, Pandas, scikit-learn (classification reports, confusion matrices) |
| **Visualization** | Matplotlib, Seaborn (training curves, heatmaps, ROC plots) |
| **Explainability** | OpenCV + custom Grad-CAM implementation for class activation mapping |
| **Web Application** | [Flask](https://flask.palletsprojects.com/) with Gunicorn for production serving |

## 🚀 Quick Setup

### Prerequisites

1. **Python**: v3.11 or higher.
2. **CUDA GPU**: Recommended for training (tested on RTX series).

### Installation

1. **Clone and Install**

```bash
git clone https://github.com/ggranbell/CAID-Coronary-AI-Ischemia-Detector.git
cd CAID-Coronary-AI-Ischemia-Detector

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install PyTorch with CUDA** (Recommended for GPU support)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. **Install Other Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Web UI**

```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser to access the interactive classification dashboard.

## 🧪 Scientific Workflow

### 1. Training

```bash
# Standard training run (50 epochs, early stopping)
python scripts/train.py

# Debug / smoke test (CPU, 1 epoch, 2 batches)
python scripts/train.py --debug

# Multi-seed run for statistical reporting
python scripts/train.py --seeds 42,123,456,789,1234
```

### 2. Evaluation

```bash
python scripts/evaluate.py
```

### 3. Hyperparameter Tuning

```bash
python scripts/tune.py --phase 1          # Learning Rate sweep
python scripts/tune.py --phase 2          # Batch Size + Weight Decay
python scripts/tune.py --phase 1 --dry-run  # Preview configs
```

### 4. Academic Analysis (ROC, t-SNE, Cost)

```bash
python scripts/analysis.py --all
```

All outputs are saved to the `outputs/analysis/` directory.

### 5. Inference

```bash
# Classify a single image
python scripts/inference.py --image path/to/image.png

# Classify an entire folder
python scripts/inference.py --folder path/to/folder/

# Default: runs on the test set
python scripts/inference.py
```

## 📂 Project Architecture

```text
CAID-Coronary-AI-Ischemia-Detector/
├── data/                          # Dataset manifests (train/val/test splits)
│   ├── train_set.txt              # Training image list
│   ├── val_set.txt                # Validation image list
│   └── test_set.txt               # Test image list
├── src/                           # Core Logic
│   ├── config.py                  # Hyperparameters and configuration
│   ├── dataset.py                 # PyTorch Dataset and DataLoader
│   ├── model.py                   # ConvNeXt V2 architecture and custom head
│   ├── utils.py                   # Helper functions and metrics
│   └── visualize_cam.py           # Grad-CAM implementation
├── scripts/                       # CLI Tools
│   ├── train.py                   # Training with AMP, cosine annealing, early stopping
│   ├── evaluate.py                # Test set evaluation and confusion matrix
│   ├── tune.py                    # Phased hyperparameter grid search
│   ├── analysis.py                # ROC/AUC, t-SNE, computational cost
│   └── inference.py               # Single/batch image inference
├── templates/                     # Flask Web UI (HTML/CSS/JS)
├── docs/                          # Detailed academic documentation
├── outputs/                       # Checkpoints, training curves, and plots
├── app.py                         # Flask Web Application
├── requirements.txt               # Python dependencies
├── DOCUMENTATION.md               # Complete technical reference
└── README.md                      # This file
```

## 📖 Documentation

Detailed technical documentation for each phase is available in the [`docs/`](docs/) directory:

- [**Overview**](docs/00_overview.md) — Project summary and quick start
- [**Dataset Setup**](docs/01_dataset_setup.md) — Data organization and manifest generation
- [**Model Architecture**](docs/02_model_architecture.md) — ConvNeXt V2 backbone and classification head
- [**Training**](docs/03_training.md) — Training loop, optimizer, and scheduler details
- [**Evaluation**](docs/04_evaluation.md) — Metrics, confusion matrix, and result analysis
- [**Hyperparameter Tuning**](docs/05_hyperparameter_tuning.md) — Phased grid search strategy
- [**Inference**](docs/06_inference.md) — CLI and Web UI usage
- [**Academic Analysis**](docs/07_academic_analysis.md) — ROC, t-SNE, cost, and multi-seed reporting
- [**WSL Setup**](docs/wsl_setup.md) — Running the project in Windows Subsystem for Linux

## ⚙️ Hardware Environment

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| **CPU** | AMD Ryzen 7 7700 |
| **RAM** | 32GB DDR5 |