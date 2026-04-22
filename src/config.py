"""
config.py — Centralized hyperparameters and paths for the ConvNeXt V2 classifier.
All training scripts import from here; change values here to affect everything.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_root: str = "dataset"
    train_list: str = "data/train_set.txt"
    val_list:   str = "data/val_set.txt"
    test_list:  str = "data/test_set.txt"
    classes: list = field(default_factory=lambda: [
        "Negative",
        "Positive",
    ])
    num_classes: int = 2

    # ── Model ────────────────────────────────────────────────────────────────
    # convnextv2_base for ConvNeXt V2 backbone
    model_name: str = "convnextv2_tiny"
    image_size: int = 384
    pretrained: bool = True

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 6
    pin_memory: bool = True

    # Mixed precision (AMP)
    use_amp: bool = True
    use_compile: bool = False  # torch.compile() is unstable on Windows

    # ── Regularization ───────────────────────────────────────────────────────
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # ── Optimizer ────────────────────────────────────────────────────────────
    backbone_lr_multiplier: float = 1.0
    gradient_clip: float = 1.0

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    warmup_epochs: int = 0
    eta_min: float = 1e-6

    # ── Loss Function ─────────────────────────────────────────────────────────
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    use_class_weights: bool = False

    # ── Early Stopping ───────────────────────────────────────────────────────
    early_stop_patience: int = 10

    # ── Augmentation (constrained) ───────────────────────────────────────────
    shear_mild: float = 15.0
    shear_strong: float = 30.0
    rotate_cat1: int = 90
    rotate_cat2: int = 180
    rotate_cat3: int = 270
    aug_probability: float = 0.7

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "outputs"
    checkpoint_name: str = "best_model.pth"
    tuning_dir: str = "tuning_results"

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Computed (do not edit) ───────────────────────────────────────────────
    def __post_init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, self.checkpoint_name)
