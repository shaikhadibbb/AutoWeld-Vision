# Implementation Plan: AutoWeld-Vision

## 1. Project Overview
**Title**: AutoWeld-Vision: Multi-Modal Deep Learning for Real-Time Welding Defect Detection in Automotive Manufacturing.
**Objective**: Build an end-to-end CV system for detecting 6 industrial defect types (Porosity, Cracks, etc.) with >97% AUROC and real-time performance.

## 2. Technical Stack
- **Framework**: PyTorch 2.3+ (torch.compile), PyTorch Lightning.
- **Anomaly Detection**: Anomalib (Intel), custom wrappers.
- **Supervised**: Ultralytics YOLOv8 (Segmentation).
- **Optimization**: ONNX Runtime, TensorRT.
- **MLOps**: Weights & Biases, Hydra.

## 3. Implementation Phases

### Phase 1: Data Infrastructure & Augmentation
- [ ] Implement `IndustrialDataset` class to handle Tiers A, B, and C.
- [ ] Integration of MVTec AD, VisA, and NEU Surface Defect datasets.
- [ ] **Tier C**: Implement synthetic anomaly generation (Perlin noise, CutPaste, and Diffusion Inpainting).
- [ ] Preprocessing: CLAHE, multi-scale tiling for high-res images.

### Phase 2: Model Architecture Implementation
- [ ] **Memory Bank**: PatchCore & PaDiM integration.
- [ ] **Student-Teacher**: EfficientAD implementation with custom welding feature distillation.
- [ ] **Generative**: DRAEM with weld-specific synthetic anomaly generator.
- [ ] **Supervised**: YOLOv8-seg training pipeline for localized defect detection.
- [ ] **Ensemble**: Weighted fusion layer for anomaly score aggregation.

### Phase 3: Training & Benchmarking
- [ ] Hydra configuration for experiment management.
- [ ] Training loops with Mixed Precision (FP16).
- [ ] Metric implementation: Image AUROC, Pixel AUROC, PRO (Per-Region Overlap).
- [ ] Ablation studies: Backbone comparison (ResNet vs ViT) and Memory Bank size.

### Phase 4: Deployment & Edge Optimization
- [ ] ONNX/TensorRT export scripts with INT8 quantization.
- [ ] Real-time FastAPI server with RTSP stream handling.
- [ ] Industrial Dashboard: VIN-linked traceability and IATF 16949 compliance reports.

### Phase 5: Documentation & Research Report
- [ ] Comprehensive README and ADRs.
- [ ] 6-8 page Technical Report in IEEE format.

## 4. Key Success Metrics
- **AUROC**: >97% (MVTec AD).
- **Inference**: <33ms (30 FPS) on target hardware.
- **False Positive Rate**: <3% @ 95% Recall.
