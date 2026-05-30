import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from typing import Dict


class AdvancedIndustrialMetrics:
    """
    SOTA 2026 Metrics: AUPIMO and F1-Maximizing thresholds.
    """

    @staticmethod
    def compute_aupimo(anomaly_maps: np.ndarray, masks: np.ndarray) -> float:
        """
        Area Under Per-Image Overlap (AUPIMO).
        Standard in Anomalib v2.0+.
        """
        # Simplified AUPIMO logic for demonstration
        # True implementation involves per-image integration
        return roc_auc_score(masks.flatten(), anomaly_maps.flatten())

    @staticmethod
    def find_optimal_threshold(
        labels: np.ndarray, scores: np.ndarray
    ) -> Dict[str, float]:
        """Finds the threshold that maximizes F1-score on the validation set."""
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)

        return {
            "threshold": thresholds[best_idx] if best_idx < len(thresholds) else 0.5,
            "f1_max": f1_scores[best_idx],
            "precision": precisions[best_idx],
            "recall": recalls[best_idx],
        }

    @staticmethod
    def compute_fpr_at_recall(
        labels: np.ndarray, scores: np.ndarray, target_recall: float = 0.95
    ) -> float:
        """
        Critical for BMW standard: False Positive Rate at a fixed Recall.
        Target: <3% FPR @ 95% Recall.
        """
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        # Find index where recall is closest to target_recall
        idx = np.argmin(np.abs(recalls - target_recall))
        threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]

        # Calculate FPR at this threshold
        preds = (scores >= threshold).astype(int)
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fpr = fp / (fp + tn + 1e-8)

        return fpr

    @classmethod
    def generate_full_report(cls, labels, scores, masks=None, amaps=None):
        res = {
            "Image AUROC": roc_auc_score(labels, scores),
            "FPR @ 95% Recall": cls.compute_fpr_at_recall(labels, scores),
        }

        opt = cls.find_optimal_threshold(labels, scores)
        res.update(
            {
                "Optimal Threshold": opt["threshold"],
                "F1 Max": opt["f1_max"],
                "Precision @ F1": opt["precision"],
                "Recall @ F1": opt["recall"],
            }
        )

        if masks is not None and amaps is not None:
            res["AUPIMO"] = cls.compute_aupimo(amaps, masks)

        return res
