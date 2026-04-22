"""
visualize_cam.py — Grad-CAM implementation for ConvNeXt V2 ischemia classifier.

Generates class activation maps to visualize which regions of coronary
angiography images the model focuses on for ischemia detection.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from .config import Config
from .model import get_model
from .dataset import build_transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32).to(input_tensor.device)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class

def visualize_samples(cfg, checkpoint_path, image_paths, output_dir="outputs/cams"):
    """
    Generate Grad-CAM visualizations for a list of images.

    Args:
        cfg             : Config dataclass
        checkpoint_path : path to trained model checkpoint
        image_paths     : list of image file paths
        output_dir      : directory to save CAM visualizations
    """
    # Load model
    model = get_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        model._orig_mod.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Find target layer (last layer of last stage of ConvNeXt V2 backbone)
    target_layer = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
        target_layer = model.backbone.stages[-1].blocks[-1]
        print(f"[cam] Targeted layer: backbone.stages[-1].blocks[-1]")
    elif hasattr(model, "_orig_mod") and hasattr(model._orig_mod.backbone, "stages"):
         target_layer = model._orig_mod.backbone.stages[-1].blocks[-1]
         print(f"[cam] Targeted layer: _orig_mod.backbone.stages[-1].blocks[-1]")

    if target_layer is None:
        raise ValueError("Could not automatically find the last stage for Grad-CAM.")

    grad_cam = GradCAM(model, target_layer)
    transform = build_transforms("test", cfg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[cam] Skipping {img_path}: Not found")
            continue

        orig_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(orig_img).unsqueeze(0).to(cfg.device)

        cam, pred_class = grad_cam.generate_cam(input_tensor)
        class_name = cfg.classes[pred_class]

        # Process for visualization
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

        # Resize original image to match input size of model (for overlay)
        img_resized = np.array(orig_img.resize((cfg.image_size, cfg.image_size)))

        # Overlay
        overlay = cv2.addWeighted(img_resized, 0.6, cam_heatmap, 0.4, 0)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_resized)
        axes[0].set_title(f"Original: {img_path.name}")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM (Pred: {class_name})")
        axes[1].axis("off")

        plt.tight_layout()
        save_path = Path(output_dir) / f"cam_{img_path.stem}.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"[cam] Saved -> {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Comma-separated paths to images")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth")
    parser.add_argument("--dataset_file", type=str, default="data/test_set.txt",
                        help="File containing image names (default: data/test_set.txt)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process")
    args = parser.parse_args()

    cfg = Config()

    if args.images:
        paths = [p.strip() for p in args.images.split(",")]
    else:
        # Pick from dataset file
        if not Path(args.dataset_file).exists():
            print(f"Error: {args.dataset_file} not found.")
            exit(1)

        with open(args.dataset_file, "r") as f:
            paths = [
                str(Path(cfg.dataset_root) / "test" / line.strip())
                for line in f.readlines() if line.strip()
            ]

        if args.limit:
            paths = paths[:args.limit]
            print(f"[cam] Limiting to {args.limit} images.")
        else:
            print(f"[cam] Processing all {len(paths)} images from {args.dataset_file}.")

    visualize_samples(cfg, args.checkpoint, paths)
