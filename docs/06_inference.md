# 06 — Inference

## CLI Inference

### Single Image
```bash
python scripts/inference.py --image path/to/coronary_image.png
```

### Multiple Images
```bash
python scripts/inference.py --image img1.png img2.png img3.png
```

### Entire Folder
```bash
python scripts/inference.py --folder my_test_images/
```

### Custom Checkpoint
```bash
python scripts/inference.py --image img.png --checkpoint tuning_results/phase_3/trial_2/best_model.pth
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | — | One or more image paths |
| `--folder` | — | Folder of images |
| `--checkpoint` | `outputs/best_model.pth` | Model checkpoint |
| `--output-dir` | `outputs/` | Where to save CSV and grid |
| `--grid-cols` | `4` | Columns in result grid image |

## Outputs

| File | Description |
|------|-------------|
| Console table | Prediction + probabilities for each image |
| `inference_results.csv` | All predictions with class probabilities |
| `inference_grid.png` | Visual montage with overlaid labels |

## Web Application

```bash
python app.py
```

Navigate to `http://localhost:5000`.

### Features:
- **Drag-and-drop** image upload
- **Real-time classification** with probability bars
- **Grad-CAM overlay** showing model attention
- **Dark theme** with glassmorphism design

### API Endpoint

```
POST /predict
Content-Type: multipart/form-data
Body: image=<file>

Response (JSON):
{
  "predicted_class": "Positive",
  "confidence": "95.32",
  "probabilities": {"Negative": 4.68, "Positive": 95.32},
  "original_image": "<base64>",
  "cam_overlay": "<base64>"
}
```

## Grad-CAM Visualization

The web UI and `src/visualize_cam.py` provide Grad-CAM (Gradient-weighted Class Activation Mapping) overlays:

```bash
# CLI: Generate Grad-CAM for specific images
python -m src.visualize_cam --images img1.png,img2.png --checkpoint outputs/best_model.pth

# CLI: Generate for all test images (limited to 10)
python -m src.visualize_cam --limit 10
```

The Grad-CAM targets the **last block of the last stage** of the ConvNeXt V2 backbone (`stages[-1].blocks[-1]`), producing activation maps that highlight which image regions contributed most to the classification decision.
