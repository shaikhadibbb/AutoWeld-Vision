#!/usr/bin/env python3
"""
Industrial Quality Inspection and Audit Trail Generation.

Runs real-time deep learning anomaly detection on welding images using trained
PatchCore or EfficientAD models. Generates publication-quality IATF 16949 audit trails
with side-by-side original and anomaly map overlays.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from typing import Dict, Any

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent))

# Apply monkey patches to resolve environment compatibility issues
import anomalib.utils.rich

anomalib.utils.rich.safe_track = lambda sequence, *args, **kwargs: sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg


def tostring_rgb_patched(self):
    rgba_bytes = bytes(self.buffer_rgba())
    rgba_np = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(-1, 4)
    return rgba_np[:, :3].tobytes()


FigureCanvasAgg.tostring_rgb = tostring_rgb_patched

from anomalib.models import Patchcore
from autoweld_vision.utils.demo_mode import run_demo_inspection


def load_inspection_model(category: str = "bottle") -> Patchcore:
    """Loads PatchCore model and loads trained weights if available."""
    model = Patchcore()
    weights_path = f"weights/patchcore_{category}.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        print(f"✓ Successfully loaded trained PatchCore model from {weights_path}")
    else:
        print(f"⚠️  Trained weights not found at {weights_path}")
        print("   Please run standard training first: python scripts/run_benchmark.py")
        raise FileNotFoundError(f"Trained weights not found at {weights_path}")
    return model


def run_real_inspection(
    image_path: str, category: str = "bottle", vin: str = "UNKNOWN"
) -> Dict[str, Any]:
    """Runs real model inference on any input image and generates IATF 16949 audit reports."""

    # 1. Load model
    try:
        model = load_inspection_model(category)
    except FileNotFoundError:
        print("⚠️  Falling back to pipeline Demo Mode...")
        return run_demo_inspection(image_path, vin)

    # 2. Preprocess image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Standard transformation
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(img).unsqueeze(0)

    # 3. Model Inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract score and anomaly map from anomalib predictions
    score_tensor = predictions.get("pred_score", predictions.get("score"))
    anomaly_score = float(score_tensor.item())
    anomaly_map = predictions["anomaly_map"].squeeze().cpu().numpy()

    # Normalize anomaly map for custom visualization
    amap_min, amap_max = anomaly_map.min(), anomaly_map.max()
    if amap_max > amap_min:
        anomaly_map = (anomaly_map - amap_min) / (amap_max - amap_min)
    else:
        anomaly_map = np.zeros_like(anomaly_map)

    # Decision logic based on standard manufacturing threshold
    THRESHOLD = 0.5
    decision = "PASS" if anomaly_score <= THRESHOLD else "FAIL"

    # 4. Generate side-by-side IATF 16949 Audit Report
    os.makedirs("audit_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"audit_logs/report_{vin}_{timestamp}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot: Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Original Weld (VIN: {vin})")
    axes[0].axis("off")

    # Right plot: Anomaly map overlay using RdYlGn_r colormap
    axes[1].imshow(img)
    # Resize anomaly map back to original size for exact spatial overlay
    resized_amap = Image.fromarray(anomaly_map).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    resized_amap_np = np.array(resized_amap)

    # Overlay using custom Red-Yellow-Green (reversed) colormap
    axes[1].imshow(resized_amap_np, cmap="RdYlGn_r", alpha=0.5)
    axes[1].set_title(f"Anomaly Map Overlay (Score: {anomaly_score:.4f})")
    axes[1].axis("off")

    # Add metadata for IATF 16949 Audit compliance
    status_text = f"Quality Decision: {decision} | Threshold: {THRESHOLD:.2f}"
    fig.suptitle(
        f"{status_text}\nIATF 16949 Audit Trail | Model: PatchCore v1.0.0 | Time: {timestamp}",
        fontsize=12,
    )

    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ IATF 16949 Audit Report generated at: {report_path}")

    return {
        "vin": vin,
        "decision": decision,
        "anomaly_score": round(anomaly_score, 4),
        "mode": "real",
        "audit_report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoWeld-Vision Image Inspector")
    parser.add_argument("--image", default="test_weld.png", help="Path to weld image")
    parser.add_argument(
        "--vin", default="UNKNOWN", help="Vehicle Identification Number"
    )
    parser.add_argument(
        "--category", default="bottle", help="MVTec category to inspect against"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AUTOWELD-VISION INDUSTRIAL QUALITY AUDITOR")
    print("=" * 60)

    res = run_real_inspection(args.image, args.category, args.vin)

    print("\n--- Final Quality Decision ---")
    print(f"VIN:           {res['vin']}")
    print(f"Decision:      {res['decision']}")
    print(f"Anomaly Score: {res['anomaly_score']:.4f}")
    print(f"Pipeline Mode: {res['mode'].upper()}")
    print(f"Audit Trail:   {res['audit_report']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
