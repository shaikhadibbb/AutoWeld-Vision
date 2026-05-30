# AutoWeld-Vision System Architecture

AutoWeld-Vision uses a multi-modal unsupervised anomaly detection pipeline optimized for high-resolution visual welding inspection.

## Key Subsystems

1. **Preprocessing Layer**:
   * CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to mitigate reflections and harsh shadows from metallic welding joints.
   * Geometric transforms and normalization map industrial optical feeds to standard `(256, 256)` feature space.

2. **Feature Extraction Backbone**:
   * A pre-trained `wide_resnet50_2` network extracts robust local mid-level features, preserving spatial locality of potential defect structures.

3. **Defect Gating & Specialization**:
   * A lightweight CNN router categorizes coarse regional activation maps.
   * Specialised Specialist models (e.g. PatchCore, PaDiM, EfficientAD) run parallel inference based on defect gating.

4. **AnomalyEnsemble Score Fusion**:
   * Predicts a combined score using validation BCE SLSQP-optimized weights:
     $$Score_{ensemble} = w_1 \cdot Score_{PatchCore} + w_2 \cdot Score_{EfficientAD}$$
   * Pixel-level anomaly maps are scaled, aligned, and overlaid using the reversed green-to-red `RdYlGn_r` colormap.

5. **Visual Audit Logging (IATF 16949)**:
   * Saves metadata-anchored visual reports (VIN, timestamp, scores, threshold, PASS/FAIL decision) directly to the local directory `audit_logs/`.
