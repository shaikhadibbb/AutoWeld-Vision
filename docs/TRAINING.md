# Training Guide

## Quick Start: Single Category

Train PatchCore on MVTec AD `bottle` using the built-in benchmark script:

```bash
python scripts/run_benchmark.py --categories bottle --output results/
```

The MVTec AD dataset is downloaded automatically by Anomalib on the first run. Trained weights are saved to `weights/patchcore_bottle.pt`.

Expected time: approximately 2 minutes on Apple Silicon (MPS), under 1 minute on NVIDIA CUDA.

## Full Benchmark: All Categories

To reproduce all benchmark numbers from the README:

```bash
python scripts/run_benchmark.py --categories bottle cable metal_nut --output results/
```

This trains PatchCore on all three categories, trains EfficientAD on `bottle`, optimizes the ensemble weights via SLSQP, and writes results to `results/benchmark.json`.

Expected time: approximately 5 minutes on Apple Silicon (MPS), 2 minutes on NVIDIA CUDA.

## Running Inference

After training, inspect any weld image:

```bash
python test_inspection.py --image test_weld.png --vin BMW-G60-2026
```

The inspection generates an IATF 16949 audit report in `audit_logs/`.

## Hardware Requirements

- **Minimum**: 8 GB RAM, Python 3.11+
- **Recommended**: Apple Silicon Mac with MPS or NVIDIA GPU with CUDA
- **Storage**: ~800 MB for the MVTec AD subset (bottle, cable, metal_nut)
