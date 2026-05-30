# Changelog

All notable changes to the **AutoWeld-Vision** quality control pipeline will be documented in this file.

---

## [1.0.0] - 2026-05-30
### Added
- **Interactive Operator Terminal**: Created a self-contained Streamlit application (`app.py`) for factory-floor supervisors to upload custom welds, trigger the late-fusion pipeline, and download certified visual audit reports.
- **FastAPI Backend Support**: Standardized API routing under `/inspect` with automated model status metrics and fail-safe demo modes.
- **Pre-Submission Quality Checklist**: Added a 10-item deployment and replication checklist inside `docs/submission_checklist.md`.
- **Contributing Guidelines**: Added `CONTRIBUTING.md` outlining the project's academic and research scope.

---

## [0.2.0-beta] - 2026-05-15
### Added
- **Late-Fusion AnomalyEnsemble**: Implemented programmatic fusion of PatchCore memory maps and EfficientAD student-teacher maps, with weights optimized via SLSQP validation Binary Cross Entropy loss minimization.
- **CLAHE Preprocessing**: Integrated Contrast Limited Adaptive Histogram Equalization ($8\times 8$ grid size, contrast limit of 2.0) to stabilize weld image features under fluctuating workshop lighting.
- **Unit Testing Suite**: Created standard testing files for audit overlays, CutPaste augmentation, and late-fusion layers, achieving $\ge 90\%$ test coverage.

### Fixed
- **Matplotlib Canvas Buffer Patch**: Added a custom monkey patch to handle headless frame extraction crashes on Apple Silicon platforms.
- **Reflective Lighting Reductions**: Lowered the CLAHE contrast limit from 4.0 to 2.0 to resolve high-frequency specular false positives on curved aluminum weld boundaries.

---

## [0.1.0-alpha] - 2026-04-20
### Added
- **Model Registry and Wrappers**: Created standard parent class interfaces for PyTorch anomaly detection networks.
- **Baseline Models**: Wrapped Anomalib's PatchCore memory-bank core set and EfficientAD networks.
- **Auto-downloader Scripts**: Set up automated caching for standard MVTec AD validation subsets.
