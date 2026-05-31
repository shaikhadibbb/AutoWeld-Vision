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

    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculates the SHA-256 hash of a file on disk to ensure physical audit integrity."""
        import hashlib
        if not os.path.exists(filepath):
            return ""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    def _sign_metadata(self, metadata: dict, secret_key: str = "autoweld_secure_secret_2026") -> str:
        """Generates a secure HMAC-SHA256 signature for the given metadata block."""
        import hashlib
        import hmac
        import json
        serialized = json.dumps(metadata, sort_keys=True)
        signature = hmac.new(
            secret_key.encode("utf-8"),
            serialized.encode("utf-8"),
            hashlib.sha256
        )
        return signature.hexdigest()

    def save_audit_report(
        self,
        vin: str,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        score: float,
        decision: str,
    ) -> str:
        """
        Saves a visual audit report for a specific VIN.
        Includes original image, heatmap, and decision metadata.
        Generates and seals the metadata to an append-only cryptographic ledger for IATF 16949 compliance.
        """
        import hashlib
        import json
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

        # Compute cryptographic file and raw data hashes
        raw_image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        report_file_hash = self._calculate_file_hash(report_path)

        # Structure the verification block for quality audits
        record = {
            "vin": vin,
            "timestamp": datetime.now().isoformat(),
            "anomaly_score": float(score),
            "quality_decision": decision,
            "raw_image_sha256": raw_image_hash,
            "report_file_sha256": report_file_hash,
            "compliance_schema": "IATF-16949-8.5.2.1",
        }

        # Seal with HMAC signature
        signature = self._sign_metadata(record)
        record["signature_hmac_sha256"] = signature

        # Append to the tamper-evident local visual audit ledger
        ledger_path = os.path.join(self.output_dir, "audit_ledger.jsonl")
        with open(ledger_path, "a") as ledger_file:
            ledger_file.write(json.dumps(record) + "\n")

        print(f"Audit report saved for VIN {vin} at {report_path}")
        print(f"✓ Cryptographically sealed quality record logged in append-only ledger: {ledger_path}")
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
