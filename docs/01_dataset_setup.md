# 01 — Dataset Setup

## Source Data

The dataset consists of **coronary angiography images** organized into two classes:

| Class | Description | Interpretation |
|-------|-------------|----------------|
| **Negative** | Normal coronary anatomy | No hemodynamically significant stenosis |
| **Positive** | Ischemic coronary findings | Arterial narrowing suggesting reduced blood flow |

## Directory Structure

```
dataset/
├── train/
│   ├── Negative/    # Training negative images
│   └── Positive/    # Training positive images
├── val/
│   ├── Negative/    # Validation negative images
│   └── Positive/    # Validation positive images
└── test/
    ├── Negative/    # Test negative images
    └── Positive/    # Test positive images
```

## Data Split

The dataset is split **70:20:10** (train:test:val) using `split_dataset.py`.

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model weight updates |
| Test  | 20% | Final held-out evaluation |
| Val   | 10% | Epoch-level monitoring, early stopping, hyperparameter selection |

## Manifest Files

After splitting, three text files are generated and stored in `data/`:

```
data/train_set.txt   →  Negative/original_Train_Negative_img_001.png
data/val_set.txt     →  Positive/synthetic_Val_Positive_gen_042.png
data/test_set.txt    →  Negative/original_Test_Negative_img_099.png
```

Each line is a relative path: `<class>/<filename>.png`.

The `IschemiaDataset` class in `src/dataset.py` reads these manifests and:
1. Parses the class label from the first path component
2. Constructs the full path as `dataset/<split>/<class>/<filename>.png`
3. Loads the image using PIL and applies the appropriate transform pipeline

## Image Format

| Property | Value |
|----------|-------|
| Format | PNG |
| Color | RGB |
| Resolution | Variable (resized to 384×384 during preprocessing) |

## Augmentation Strategy

Only **constrained augmentations** are applied during training — operations that preserve the clinical meaning of angiography images:

| Augmentation | Categories | Probability |
|-------------|------------|-------------|
| **RandomShear** | Mild (±15°), Strong (±30°) | 70% |
| **RandomCategoricalRotation** | 90°, 180°, 270° | 70% |

No color jitter, random cropping, or flipping is used — these could alter the clinical interpretation of coronary images.
