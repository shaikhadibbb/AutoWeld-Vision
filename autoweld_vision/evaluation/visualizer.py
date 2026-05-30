import matplotlib.pyplot as plt
import numpy as np
import cv2


class IndustrialVisualizer:
    @staticmethod
    def plot_anomaly_heatmap(image, anomaly_map, threshold=0.5):
        """Plots original image and anomaly heatmap side by side."""
        # Normalize heatmap for display
        heatmap = (anomaly_map - anomaly_map.min()) / (
            anomaly_map.max() - anomaly_map.min() + 1e-8
        )
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay
        overlay = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Anomaly Heatmap")

        return fig
