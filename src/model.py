"""
model.py — ConvNeXt V2 model builder for coronary ischemia classification.

Uses timm (PyTorch Image Models) to load a pretrained convnextv2_base and
replaces the classification head with a 2-class linear layer for binary
classification (Positive / Negative ischemia).
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required. Install with: pip install timm"
    )

from .config import Config


class ConvNeXtV2Classifier(nn.Module):
    """
    ConvNeXt V2 backbone (pretrained on ImageNet-1K) with a custom 2-class head.

    Architecture:
        ConvNeXt V2 backbone → LayerNorm → Dropout(0.3) → Linear(in_features, 2)

    The dropout before the final layer helps regularize the model on
    medical imaging datasets where data may be limited.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=0,          # Remove the original head
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (B, in_features)
        return self.head(features)     # (B, num_classes)

    def get_param_groups(self, lr: float, backbone_lr_multiplier: float) -> list[dict]:
        """
        Returns AdamW param groups with differential learning rates.
        backbone_lr = lr * backbone_lr_multiplier  (lower LR for pretrained weights)
        head_lr     = lr                           (higher LR for new classifier)
        """
        return [
            {"params": self.backbone.parameters(), "lr": lr * backbone_lr_multiplier},
            {"params": self.head.parameters(),     "lr": lr},
        ]


def get_model(config: Config) -> ConvNeXtV2Classifier:
    """
    Factory function — create and move model to the configured device.

    Args:
        config: Config dataclass

    Returns:
        ConvNeXtV2Classifier on the correct device
    """
    model = ConvNeXtV2Classifier(config)
    model = model.to(config.device)

    # Enable torch.compile for extra speed (PyTorch 2.0+)
    if config.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[model] torch.compile() enabled — best GPU throughput")
        except Exception:
            print("[model] torch.compile() not available, skipping")

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {config.model_name} | "
          f"total params: {total_params/1e6:.1f}M | "
          f"trainable: {trainable_params/1e6:.1f}M | "
          f"device: {config.device}")
    return model
