"""
Demo Mode Utilities for AutoWeld-Vision.
Provides high-fidelity mock representations and visual fallbacks when real models are not trained.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from typing import Dict, Any

def run_demo_inspection(image_path: str, vin: str = "UNKNOWN") -> Dict[str, Any]:
    """Runs a simulated demo inspection and generates a mock audit report."""
    anomaly_score = 0.148
    decision = "OK"
    
    # Generate visual audit report
    os.makedirs("audit_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"audit_logs/report_{vin}_{timestamp}.png"
    
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        img = Image.new('RGB', (256, 256), color=(73, 109, 137))
    
    # Demo heatmap: subtle brightening/color shift to simulate heatmap
    heatmap = np.array(img) * 0.9 + 20
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"Original Weld (VIN: {vin})")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap)
    axes[1].set_title(f"Anomaly Heatmap (Score: {anomaly_score:.4f})")
    axes[1].axis("off")
    
    fig.suptitle(f"Quality Decision: {decision} | IATF 16949 Audit Trail (DEMO)", fontsize=14)
    
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return {
        "vin": vin,
        "decision": decision,
        "anomaly_score": anomaly_score,
        "mode": "demo",
        "audit_report": report_path
    }
