import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from datetime import datetime


class IndustrialQualityAuditor:
    """
    IATF 16949 Compliance: Generates audit-ready reports and explainability visualizations.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_anomaly_heatmap(
        self, image: np.ndarray, anomaly_map: np.ndarray
    ) -> np.ndarray:
        """Overlays a jet-color heatmap on the original image."""
        # Normalize anomaly map
        amap = (anomaly_map - anomaly_map.min()) / (
            anomaly_map.max() - anomaly_map.min() + 1e-8
        )

        # Resize amap to match original image size
        amap = cv2.resize(amap, (image.shape[1], image.shape[0]))

        heatmap = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        return overlay

    def save_audit_report(
        self,
        vin: str,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        score: float,
        decision: str,
    ):
        """
        Saves a visual audit report for a specific VIN.
        Includes original image, heatmap, and decision metadata.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"report_{vin}_{timestamp}.png")

        heatmap = self.generate_anomaly_heatmap(image, anomaly_map)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title(f"Original Weld (VIN: {vin})")
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].set_title(f"Anomaly Heatmap (Score: {score:.4f})")
        axes[1].axis("off")

        plt.suptitle(
            f"Quality Decision: {decision} | IATF 16949 Audit Trail", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(report_path)
        plt.close()

        print(f"Audit report saved for VIN {vin} at {report_path}")
        return report_path


class ParetoTracker:
    """Tracks defect distribution for Cpk and Pareto analysis."""

    def __init__(self):
        self.defect_counts = {"Porosity": 0, "Cracks": 0, "Spatter": 0, "OK": 0}

    def update(self, defect_type: str):
        if defect_type in self.defect_counts:
            self.defect_counts[defect_type] += 1
        else:
            self.defect_counts[defect_type] = 1

    def get_pareto_data(self):
        return sorted(self.defect_counts.items(), key=lambda x: x[1], reverse=True)
