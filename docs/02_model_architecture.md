# 02 — Model Architecture

## ConvNeXt V2

CAID uses **ConvNeXt V2** — a modernized convolutional architecture that introduces **Global Response Normalization (GRN)** for improved inter-channel feature competition. The V2 family achieves competitive performance against Vision Transformers while retaining the efficiency and simplicity of pure CNNs.

### Key innovations in ConvNeXt V2:
- **Global Response Normalization (GRN)**: Enhances feature diversity by recalibrating channel responses globally
- **Fully Convolutional Masked Autoencoder (FCMAE)**: Pretraining strategy that improves representation quality
- **Depthwise convolutions** with large 7×7 kernels

## Architecture Details

```
ConvNeXtV2Classifier
├── backbone: timm.create_model("convnextv2_base", pretrained=True, num_classes=0)
│   ├── stem          # Patchify + LayerNorm
│   ├── stages[0..3]  # 4 ConvNeXt blocks with downsampling
│   └── head.global_pool  # Global Average Pooling → (B, 1024)
│
└── head: nn.Sequential
    ├── LayerNorm(1024)
    ├── Dropout(0.3)
    └── Linear(1024, 2)   # Binary output (Negative/Positive)
```

## Model Variants

| Variant | Params | timm name | Default Resolution |
|---------|--------|-----------|-------------------|
| Tiny    | ~28M   | `convnextv2_tiny` | 224×224 |
| Small   | ~50M   | `convnextv2_small` | 384×384 |
| **Base** | **~89M** | **`convnextv2_base`** | **384×384** |
| Large   | ~198M  | `convnextv2_large` | 384×384 |

The default configuration uses `convnextv2_base` at 384×384 — offering the best balance of capacity and efficiency for medical image classification.

## Transfer Learning Strategy

1. **Pretrained backbone**: All ConvNeXt V2 weights are loaded from ImageNet-1K
2. **Custom head**: The original 1000-class head is replaced with a 2-class head
3. **Full fine-tuning**: All layers are unfrozen by default
4. **Differential LR** (optional): The backbone can use a lower learning rate than the head via `backbone_lr_multiplier`

## Implementation

See: [`src/model.py`](../src/model.py)

```python
model = get_model(config)  # Creates and returns ConvNeXtV2Classifier on device
```

The `ConvNeXtV2Classifier.get_param_groups()` method returns separate parameter groups for the optimizer:
```python
param_groups = model.get_param_groups(lr=1e-4, backbone_lr_multiplier=0.1)
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
```
