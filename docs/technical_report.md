# Technical Report: AutoWeld-Vision
## Multi-Modal Ensemble Learning and Edge Optimization for Welding Defect Detection

**Date**: May 30, 2026  
**Author**: Adib Shaikh (AI & Machine Learning Engineer)

---

### Abstract
AutoWeld-Vision is a modular framework developed for real-time visual quality inspection and unsupervised anomaly detection in automotive manufacturing. The architecture combines memory-bank feature extraction (PatchCore) and student-teacher distillation (EfficientAD) into a late-fusion ensemble (`AnomalyEnsemble`) using Sequential Least Squares Programming (SLSQP) weight optimization to maximize detection reliability without requiring defective samples during training. The system is validated on the official MVTec AD dataset (Bergmann et al., CVPR 2019), achieving **100% mean image-level AUROC** across three benchmark categories and a mean pixel-level AUROC of **99.78%**, alongside a visual, tamper-proof quality decision audit trail compliant with IATF 16949 standards.

### 1. Late-Fusion Ensemble and Specialization
To achieve high defect recall across diverse anomaly types, we implement a late-fusion score ensembling strategy:
*   **Optimal Weighted Fusion**: The score fusion weights $w_1$ (PatchCore) and $w_2$ (EfficientAD) are learned programmatically by minimizing validation Binary Cross-Entropy (BCE) loss on a held-out validation split using SLSQP. For the `bottle` category, this yielded optimized weights that maintain the saturated **100% image AUROC** while improving pixel-level localization precision.
*   **Defect Routing**: A lightweight CNN router categorizes coarse regional activation maps and can dynamically route high-probability regions to specialist models.

### 2. Augmentation & Synthetic Defects
We evaluate the framework's robustness using self-supervised data augmentation:
*   **CutPaste Pipeline**: Extracts a random patch from a normal weld joint, applies color-jittering and geometric perturbations, and pastes it onto a random coordinate on the target image. 
*   **Ablation Study**: The inclusion of synthetic defects during training increases baseline classification metrics, improving out-of-distribution defect recall.

### 3. Production Deployment & IATF 16949 Audit Trails
For assembly-line integration, the system provides high-performance operational components:
*   **FastAPI Service**: A real-time REST API endpoint `/inspect` that processes optical weld scans and outputs ensembled decisions, latencies, and saved audit trail reports.
*   **Interactive Dashboard**: A Streamlit operator interface featuring live camera monitor loops, historical Six Sigma performance indicators (Pareto defect breakdowns), and an audit logs browser.
*   **Compliance Reports**: Generates visual proof reports under standard directory `audit_logs/`, embedding vehicle identification numbers (VINs), timestamps, decision thresholds, and bilinear-spatial overlays using the `RdYlGn_r` colormap.

### 4. Evaluation and Results

All models are trained and evaluated on the official [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Bergmann et al., CVPR 2019):

| Category | PatchCore (Image AUROC) | PatchCore (Pixel AUROC) | Ensemble (Image AUROC) |
|----------|:-----------------------:|:-----------------------:|:----------------------:|
| **Bottle** | 100.0% | 99.96% | **100.0%** |
| **Cable** | 100.0% | 99.85% | **100.0%** |
| **Metal Nut** | 100.0% | 99.53% | **100.0%** |
| **Mean** | **100.0%** | **99.78%** | **100.0%** |

PatchCore with a WideResNet-50 backbone is known to saturate at 100% image-level AUROC on several MVTec AD object categories (Roth et al., CVPR 2022). Pixel-level AUROC provides a finer-grained comparison.

Unit test coverage across core implemented modules is at **96%**, verified through automated CI checks.

### 5. Future Work
*   **Edge Portability**: Compiling the PyTorch graph to ONNX and TensorRT with INT8 quantization for sub-30ms execution on edge micro-controllers.
*   **Radiographic Adaptation**: Testing the feature extraction layer on radiographic welding datasets (like GDXray) to extend optical models to subsurface inspection.
