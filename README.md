# AutoWeld-Vision: Unsupervised Anomaly Detection for Industrial Quality Control

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.1-red.svg)](https://pytorch.org/)
[![CI Status](https://github.com/shaikhadibbb/Industrial-Computer-Vision-Defect-Detection-/actions/workflows/ci.yml/badge.svg)](https://github.com/shaikhadibbb/Industrial-Computer-Vision-Defect-Detection-/actions)
[![Test Coverage](https://img.shields.io/badge/coverage-%3E80%25-green.svg)](https://github.com/shaikhadibbb/Industrial-Computer-Vision-Defect-Detection-)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoWeld-Vision is an open-source deep learning framework developed for real-time visual quality inspection and anomaly detection in industrial manufacturing. It combines deep feature extraction (via a PatchCore-based memory bank) and student-teacher distillation (via a customized EfficientAD model) in a late-fusion ensemble to identify micro-defects, porosity, and cracks in weld joints without requiring negative (defective) training samples.

In addition to defect detection, the framework generates IATF 16949-compliant visual audit reports, detailing vehicle tracking numbers (VINs), inspection timestamps, decision thresholds, and spatial anomaly heatmaps.

---

## Quantitative Benchmarks

The following table presents image-level AUROC scores evaluated on our validation splits for three MVTec AD categories. The late-fusion ensemble (`AnomalyEnsemble`) uses optimized validation weights obtained via SLSQP binary cross-entropy minimization to combine model predictions.

| Category | PatchCore (Image AUROC) | EfficientAD (Image AUROC) | Ensemble (Image AUROC) | Baseline (Dinomaly) |
| :--- | :---: | :---: | :---: | :---: |
| **Bottle** | 98.2% | 97.8% | **98.9%** | 99.6% |
| **Cable** | 96.5% | 95.9% | **97.4%** | 99.1% |
| **Metal Nut** | 97.1% | 96.4% | **97.8%** | 99.3% |
| **Mean** | **97.3%** | **96.7%** | **98.0%** | **99.3%** |

> [!NOTE]
> **Scientific Benchmark Disclaimer**: The quantitative results above are trained and evaluated on programmatically generated local splits for the MVTec AD categories. Due to source URL 404 errors during direct programmatic dataset downloads, these scores reflect validation performance on high-fidelity synthetic splits and are not directly comparable to published full-dataset SOTA benchmarks (such as the 99.6% Dinomaly baseline). They are provided to verify pipeline integration, late-fusion BCE ensembling convergence, and top-to-bottom codebase reproducibility.

---

## Pipeline Architecture

The image pipeline preprocesses industrial optical feeds, extracts localized deep features via a pre-trained backbone, runs parallel inference through the memory-bank and student-teacher architectures, fuses scores, and logs the visual inspection data.

```mermaid
graph TD
    A[Input Weld Image] --> B[CLAHE Preprocessing]
    B --> C[Wide-ResNet50 Backbone]
    C --> D[PatchCore Memory Bank]
    C --> E[EfficientAD Distillation]
    D --> F[PatchCore Score & Map]
    E --> G[EfficientAD Score & Map]
    F & G --> H[AnomalyEnsemble Score Fusion]
    H --> I[Decision: PASS / FAIL]
    I --> J[IATF 16949 visual audit trail PNG]
```

---

## Getting Started

### 1. Clone & Enter Directory
```bash
git clone https://github.com/shaikhadibbb/Industrial-Computer-Vision-Defect-Detection- && cd Industrial-Computer-Vision-Defect-Detection-
```

### 2. Install Dependencies
```bash
pip install -r requirements-standard.txt
```

### 3. Run Inference
Inspect an image with a specific Vehicle Identification Number (VIN) to generate a visual audit report:
```bash
python test_inspection.py --image test_weld.png --vin BMW-G60-2026
```

---

## Reproducing Benchmarks

To rebuild the synthetic datasets, run the training pipeline, optimize model weights, and dump the quantitative evaluations to `results/benchmark.json`, run:
```bash
python scripts/run_benchmark.py --categories bottle cable metal_nut --output results/
```

---

## Quality Management & IATF 16949 Context

In automotive manufacturing, quality assurance is guided by the international standard **IATF 16949:2016** (Quality Management System Requirements for Automotive Production). Specifically:
* **Section 8.5.1.1 (Control Plan)**: Demands active control plans at the system, subsystem, and part levels.
* **Section 8.5.2.1 (Identification and Traceability)**: Mandates robust recording of quality decisions and tracking coordinates.

AutoWeld-Vision automates visual inspection traceability by exporting an immutable visual audit trail in `audit_logs/`. Each report seals the vehicle tracking identification, timestamp, decision threshold, and overlay score. The inspection overlays a bilinear-interpolated anomaly map using the `RdYlGn_r` colormap directly onto the source image, giving floor operators instant spatial feedback.

---

## Current Limitations & Future Work

1. **Edge Deployment**: Compiling the PyTorch graph to ONNX/TensorRT with INT8 quantization to achieve low latency (<30ms) on NVIDIA Jetson architectures.
2. **Domain Adaptation**: Testing the feature extractor on radiographic datasets (like GDXray) to extend visual-light models to X-ray inspections.
3. **Dynamic Memory Updating**: Integrating online clustering/updates to append newly verified normal profiles to the memory bank without full offline training.
4. **Natural Language Feedback**: Grounding spatial anomaly maps with visual-language models (e.g. LLaVA) to generate written defect summaries automatically.
5. **High-Resolution Tiling**: Implementing sliding-window multiscale tiling to resolve tiny microscopic fractures in high-resolution 4K optical inspections.

---

## Contact

* **Author**: Adib Shaikh (AI & Machine Learning Engineer)
* **Email**: [adib.shaikh@tum.de](mailto:adib.shaikh@tum.de) / [shaikhadib.work@gmail.com](mailto:shaikhadib.work@gmail.com)
* **LinkedIn**: [linkedin.com/in/adib-shaikh-tum](https://linkedin.com/in/adib-shaikh-tum)
* **GitHub**: [github.com/shaikhadibbb](https://github.com/shaikhadibbb)
