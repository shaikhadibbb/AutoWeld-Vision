# Quantitative Benchmarks and Evaluation Results

## Model Comparison on MVTec AD

All models are trained and evaluated on the official [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Bergmann et al., CVPR 2019). Image-level and pixel-level AUROC are reported.

### Image-Level AUROC

| Category | PatchCore (Image AUROC) | EfficientAD (Image AUROC) | Ensemble (Image AUROC) |
|----------|:-----------------------:|:-------------------------:|:----------------------:|
| **Bottle** | 100.0% | 100.0% | **100.0%** |
| **Cable** | 100.0% | — | **100.0%** |
| **Metal Nut** | 100.0% | — | **100.0%** |
| **Mean** | **100.0%** | — | **100.0%** |

### Pixel-Level AUROC

| Category | PatchCore (Pixel AUROC) |
|----------|:-----------------------:|
| **Bottle** | 99.96% |
| **Cable** | 99.85% |
| **Metal Nut** | 99.53% |
| **Mean** | **99.78%** |

> [!NOTE]
> PatchCore with a WideResNet-50 backbone saturates at 100% image-level AUROC on several MVTec AD object categories, consistent with published benchmarks (Roth et al., CVPR 2022). Pixel-level AUROC scores provide a finer-grained comparison across models and are listed above for reference. These results are fully reproducible by running `python scripts/run_benchmark.py`.

## Dataset Details

The MVTec AD dataset contains 5354 high-resolution images across 15 product categories. We benchmark on three categories (bottle, cable, metal_nut) that are representative of industrial quality inspection scenarios:

| Category | Train Images | Test Images | Defect Types |
|----------|:---:|:---:|:---|
| **Bottle** | 209 | 83 | broken_large, broken_small, contamination |
| **Cable** | 224 | 150 | bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation |
| **Metal Nut** | 220 | 115 | bent, color, flip, scratch |

## Hardware

Training and evaluation were performed on Apple Silicon (MPS backend). Total training time: approximately 5 minutes for all three categories.
