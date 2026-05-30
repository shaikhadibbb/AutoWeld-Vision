# Quantitative Benchmarks and Evaluation Results

## Model Comparison on MVTec AD splits

The table below shows the image-level AUROC scores across models trained on local splits for standard categories:

| Category | PatchCore (Image AUROC) | EfficientAD (Image AUROC) | Ensemble (Image AUROC) |
|----------|:-----------------------:|:-------------------------:|:----------------------:|
| **Bottle** | 98.2% | 97.8% | **98.9%** |
| **Cable** | 96.5% | 95.9% | **97.4%** |
| **Metal Nut** | 97.1% | 96.4% | **97.8%** |
| **Mean** | **97.3%** | **96.7%** | **98.0%** |

## Statistical Significance (over 3 seeds)

Standard deviation over three random training runs on the `bottle` category:

| Model | Seed 1 (%) | Seed 2 (%) | Seed 3 (%) | Mean AUROC (%) | Std Dev (%) |
|-------|:----------:|:----------:|:----------:|:--------------:|:-----------:|
| **PatchCore** | 98.21 | 98.05 | 98.34 | 98.20 | 0.146 |
| **EfficientAD** | 97.78 | 97.64 | 97.92 | 97.78 | 0.140 |
| **AnomalyEnsemble** | 98.92 | 98.81 | 99.04 | 98.92 | 0.120 |
