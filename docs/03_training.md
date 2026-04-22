# 03 — Training

## Training Script

```bash
python scripts/train.py                       # Single seed (42)
python scripts/train.py --seeds 42,123,456    # Multi-seed
python scripts/train.py --debug               # CPU smoke test
```

## Training Loop

Each epoch:
1. **Forward pass** with AMP autocast (mixed precision)
2. **Loss computation** (CrossEntropyLoss with label smoothing)
3. **Backward pass** with GradScaler
4. **Gradient clipping** (max_norm = 1.0)
5. **Optimizer step** (AdamW)
6. **Scheduler step** (Cosine Annealing)
7. **Early stopping check** on validation loss

## Optimizer Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-2 |
| Gradient Clipping | max_norm = 1.0 |

### Differential Learning Rate

When `backbone_lr_multiplier < 1.0`:
- **Backbone** (pretrained): `lr × backbone_lr_multiplier`
- **Head** (new): `lr`

This preserves pretrained features while allowing the new classification head to learn quickly.

## LR Scheduler

**Cosine Annealing** with optional **linear warmup**:

```
Warmup phase (if warmup_epochs > 0):
  LR: 0.01 × lr → lr  (linear ramp)

Cosine phase:
  LR: lr → eta_min  (cosine decay)
```

## Loss Function

### Default: CrossEntropyLoss
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Optional: Focal Loss
When `use_focal_loss = True`:
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
Focuses training on hard, misclassified examples. Controlled by `focal_loss_gamma`.

### Optional: Class Weights
When `use_class_weights = True`, per-class weights are computed inversely proportional to class frequency in the training set.

## Early Stopping

| Parameter | Value |
|-----------|-------|
| Monitored metric | Validation loss |
| Patience | 10 epochs |
| Minimum delta | 1e-4 |

## Outputs

| File | Description |
|------|-------------|
| `outputs/best_model.pth` | Best checkpoint (by val accuracy) |
| `outputs/training_history.json` | Epoch-by-epoch metrics |
| `outputs/training_curves.png` | Loss and accuracy plots |
| `outputs/multi_seed_results.json` | Per-seed summary + mean ± std (multi-seed mode) |

## Multi-Seed Training

For academic reporting, run training across 5 seeds:

```bash
python scripts/train.py --seeds 42,123,456,789,1234
```

Output format:
```
MULTI-SEED RESULTS  (5 seeds)
═══════════════════════════════════════════════
Seed       Val Acc    Best Epoch   Epochs Run
─────────────────────────────────────────────
42          95.23%           34           45
123         94.87%           28           38
...
─────────────────────────────────────────────
Mean        95.12%  ±0.32%

Final result to report: Val Acc = 95.12 ± 0.32%
```
