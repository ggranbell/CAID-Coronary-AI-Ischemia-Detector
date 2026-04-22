"""
app.py — Flask Web Application for CAID Coronary AI Ischemia Detector.

Provides a modern web interface for:
  - Drag-and-drop image upload
  - Real-time classification with probability chart
  - Grad-CAM heatmap visualization

Usage:
  python app.py
  Navigate to http://localhost:5000
"""
import os
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from src.config import Config
from src.model import get_model
from src.dataset import build_transforms
from src.visualize_cam import GradCAM

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and configuration
cfg = Config()
model = None
grad_cam = None
transform = None

def load_resources():
    global model, grad_cam, transform, cfg

    # Load model
    print(f"[app] Loading model from {cfg.checkpoint_path}...")
    if not os.path.exists(cfg.checkpoint_path):
        print(f"[app] ERROR: Checkpoint not found at {cfg.checkpoint_path}")
        return False

    model = get_model(cfg)
    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        model._orig_mod.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Find target layer for Grad-CAM
    target_layer = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
        target_layer = model.backbone.stages[-1].blocks[-1]
    elif hasattr(model, "_orig_mod") and hasattr(model._orig_mod.backbone, "stages"):
         target_layer = model._orig_mod.backbone.stages[-1].blocks[-1]

    if target_layer:
        grad_cam = GradCAM(model, target_layer)
        print("[app] Grad-CAM initialized.")
    else:
        print("[app] WARNING: Could not initialize Grad-CAM (target layer not found).")

    transform = build_transforms("test", cfg)
    return True

@app.route('/')
def index():
    return render_template('index.html')

def image_to_base64(image_np):
    """Convert numpy image (RGB) to base64 string."""
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load and preprocess image
            orig_img = Image.open(filepath).convert("RGB")
            input_tensor = transform(orig_img).unsqueeze(0).to(cfg.device)

            # Predict and generate Grad-CAM
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx]) * 100

            # Generate CAM heatmap
            cam, _ = grad_cam.generate_cam(input_tensor, target_class=pred_idx)

            # Process for visualization
            cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

            # Resize original image to match input size
            img_resized = np.array(orig_img.resize((cfg.image_size, cfg.image_size)))

            # Overlay
            overlay = cv2.addWeighted(img_resized, 0.6, cam_heatmap, 0.4, 0)

            # Convert images to base64
            original_base64 = image_to_base64(img_resized)
            overlay_base64 = image_to_base64(overlay)

            # Cleanup uploaded file
            os.remove(filepath)

            return jsonify({
                'predicted_class': cfg.classes[pred_idx],
                'confidence': f"{confidence:.2f}",
                'probabilities': {cfg.classes[i]: float(probs[i]) * 100 for i in range(len(cfg.classes))},
                'original_image': original_base64,
                'cam_overlay': overlay_base64
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if load_resources():
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("[app] Failed to load resources. Check config/best_model.pth.")
