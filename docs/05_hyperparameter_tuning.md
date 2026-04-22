# 05 — Hyperparameter Tuning

## Strategy: Phased Grid Search

Hyperparameters are tuned in **8 sequential phases**. Each phase sweeps a small group of related parameters while keeping all others fixed at their best known values from previous phases. This cascading approach keeps the search space manageable (35 total trials instead of combinatorial explosion).

## Usage

```bash
python scripts/tune.py --phase 1            # Run Phase 1 (Learning Rate)
python scripts/tune.py --phase 2            # Run Phase 2 (Batch + WD)
python scripts/tune.py --phase 1 --dry-run  # Preview trial configs
```

## Phase Definitions

### Phase 1 — Learning Rate
| Trial | LR |
|-------|----|
| 1 | 1e-5 |
| 2 | 5e-5 |
| 3 | **1e-4** (baseline) |
| 4 | 3e-4 |
| 5 | 5e-4 |

### Phase 2 — Batch Size + Weight Decay
| Trial | Batch Size | Weight Decay |
|-------|-----------|-------------|
| 1 | 8 | 1e-2 |
| 2 | 8 | 5e-2 |
| 3 | **16** | **1e-2** (baseline) |
| 4 | 16 | 5e-2 |
| 5 | 32 | 1e-2 |
| 6 | 32 | 5e-2 |

### Phase 3 — Dropout + Label Smoothing
| Trial | Dropout | Label Smoothing |
|-------|---------|----------------|
| 1 | 0.2 | 0.0 |
| 2 | 0.2 | 0.1 |
| 3 | **0.3** | **0.1** (baseline) |
| 4 | 0.5 | 0.1 |
| 5 | 0.5 | 0.2 |

### Phase 4 — Model Variant + Resolution
| Trial | Model | Resolution | Batch |
|-------|-------|-----------|-------|
| 1 | convnextv2_tiny | 224 | 32 |
| 2 | convnextv2_small | 384 | 16 |
| 3 | **convnextv2_base** | **384** | **16** (baseline) |

### Phase 5 — LR Scheduler
| Trial | Warmup Epochs | eta_min |
|-------|--------------|---------|
| 1 | **0** | **1e-6** (baseline) |
| 2 | 3 | 1e-6 |
| 3 | 5 | 1e-6 |
| 4 | 5 | 1e-7 |

### Phase 6 — Loss Function
| Trial | Focal Loss | Gamma | Class Weights |
|-------|-----------|-------|--------------|
| 1 | **False** | 2.0 | **False** (baseline) |
| 2 | True | 1.0 | False |
| 3 | True | 2.0 | False |
| 4 | True | 2.0 | True |
| 5 | False | 2.0 | True |

### Phase 7 — Differential LR
| Trial | backbone_lr_multiplier |
|-------|-----------------------|
| 1 | 0.1 |
| 2 | 0.3 |
| 3 | **1.0** (baseline) |

### Phase 8 — Augmentation Probability
| Trial | aug_probability |
|-------|----------------|
| 1 | 0.3 |
| 2 | 0.5 |
| 3 | **0.7** (baseline) |
| 4 | 1.0 |

## Outputs

Per trial:
- `tuning_results/phase_N/trial_X/best_model.pth`
- `tuning_results/phase_N/trial_X/training_history.json`
- `tuning_results/phase_N/trial_X/training_curves.png`
- `tuning_results/phase_N/trial_X/val_metrics.json`

Per phase:
- `tuning_results/phase_N/phase_summary.json` — ranked results + best config
- `tuning_results/phase_N/phase_comparison.png` — bar chart comparing all trials

## Reading Results

After running a phase, check the summary:
```json
{
  "phase": 1,
  "best_config": {"lr": 5e-05, ...},
  "ranked_results": [...]
}
```

The `best_config` from each phase is automatically loaded by subsequent phases.
