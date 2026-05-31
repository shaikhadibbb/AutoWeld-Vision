# AutoWeld-Vision: System Architecture and Pipeline Flow

This document details how the data flows through the AutoWeld-Vision system, from the raw camera capture to the final pass/fail signal and compliance report. I designed this pipeline to run entirely in PyTorch to keep inference quick and modular.

## Pipeline Walkthrough

1. **Preprocessing (CLAHE and Filtering)**:
   Specularity and metallic glare are major problems when inspecting weld seams. To smooth out extreme reflections, the input image goes through Contrast Limited Adaptive Histogram Equalization (CLAHE) split into 8x8 local blocks, followed by a slight Gaussian blur. The images are then resized to 256x256 and normalized using standard ImageNet mean and standard deviation.

2. **Feature Extraction (Frozen Backbone)**:
   A standard ResNet backbone (either Wide-ResNet-50 or ResNet-18) acts as the feature extractor. We freeze all parameters (`requires_grad = False`) and put the backbone in evaluation mode. We attach forward PyTorch hooks to extract intermediate activations from `layer2` and `layer3` to capture mid-level spatial shapes.

3. **Dual Model Paths**:
   - **PatchCore (Coreset Memory)**: We pool the hooked feature maps using a 3x3 local average pooling operation, concatenate them, and project them to a coreset memory bank. During inference, we calculate the L2 nearest-neighbor distance between the test image's patches and the coreset.
   - **EfficientAD (Student-Teacher Distillation)**: A student ResNet-18 is trained to match the frozen teacher's feature maps on normal welds. Anomalies are flagged by looking at the L2 gap between the student's predictions and the teacher's features, combined with an autoencoder reconstruction path.

4. **Calibrated Score Late Fusion**:
   Because PatchCore outputs Euclidean coreset distances and EfficientAD outputs MSE reconstruction gaps, we cannot combine them directly. The raw scores are passed through individual Platt Scaling calibrators (logistic sigmoids) to yield probabilities. These probabilities are then combined using ensembling weights optimized via Sequential Least Squares Programming (SLSQP).

5. **Decision Routing and PLC Link**:
   - The coarse anomaly map is passed to a geometric router (`DefectRouter`) which checks aspect ratios and defect area size.
   - The final PASS/FAIL decision is sent as a binary signal (`0x01` for reject, `0x00` for pass) to the local PLC socket over port 5002.
   - A side-by-side PNG report and a signed JSON metadata ledger (HMAC-SHA256) are written to `audit_logs/` to satisfy IATF 16949 section 8.5.2.1 requirements.
