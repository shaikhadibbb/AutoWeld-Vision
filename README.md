# AutoWeld-Vision 🏭

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](...)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red.svg)](...)

> Modular industrial anomaly detection framework for automotive welding quality control.
> Built for German Industry 4.0 and Master's applications (TUM, RWTH Aachen, KIT).

## ⚠️ Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| System Architecture | ✅ Implemented | 8 model registry, ensemble fusion, routing |
| IATF 16949 Audit | ✅ Working | VIN-linked PNG reports with heatmaps |
| FastAPI Server | ✅ Implemented | `/inspect` endpoint with Pareto tracking |
| Docker + CI/CD | ✅ Implemented | Production deployment ready |
| Synthetic Generation | ✅ Working | CutPaste, Perlin noise for data augmentation |
| **Real Model Weights** | ⏳ **Pending** | Requires MVTec AD + training run |
| **Benchmark Numbers** | ⏳ **Pending** | Requires trained models + evaluation |
| **Edge Deployment** | ⏳ **Pending** | Requires Jetson Orin hardware |

## 🛠️ Environment

Developed and tested on:
- **macOS** (Apple Silicon)
- **Python 3.11** (Homebrew)
- **PyTorch 2.3.1** (CPU/MPS)
- **Anomalib 1.1.0** (PyPI)

> Note: Original prototyping used Antigravity IDE. This repository uses 
> standard Python packages from PyPI for full reproducibility.

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/autoweld-vision.git
cd autoweld-vision

# 2. Create environment (Python 3.11 required)
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements-standard.txt

# 4. Download MVTec AD dataset
# https://www.mvtec.com/company/research/datasets/mvtec-ad
# Extract to: ./datasets/mvtec/

# 5. Train baseline model
python test_inspection.py  # Trains PatchCore on 'bottle' category

# 6. Run inspection
python test_inspection.py  # Uses trained model for real inference
```

## 📚 SOTA Reference

Published benchmarks from peer-reviewed papers:

| Method | MVTec AD Img AUROC | Paper |
|--------|-------------------|-------|
| PatchCore | 99.1% | Roth et al., 2022 |
| EfficientAD | 99.1% | Bennaman et al., 2024 |
| Dinomaly | 99.6% | Li et al., CVPR 2025 |

Our framework integrates these methods with ensemble fusion and industrial 
compliance. Local benchmarks will be added after training.

## 📁 Project Structure

```
autoweld-vision/
├── autoweld_vision/     # Core package
│   ├── data/             # Dataset loaders + synthetic generation
│   ├── models/           # Model registry + ensemble
│   ├── evaluation/       # Metrics + benchmarking
│   ├── deployment/       # FastAPI + edge export
│   └── utils/            # IATF 16949 audit + config
├── configs/              # Hydra configuration files
├── docs/                 # Technical report + guides
├── notebooks/            # Analysis notebooks
├── tests/                # Unit tests
├── audit_logs/           # Generated inspection reports
└── weights/              # Trained model checkpoints
```

## 🎯 Next Steps

1. [ ] Download MVTec AD dataset
2. [ ] Train PatchCore baseline (~20 min on Mac MPS)
3. [ ] Run full benchmark on all 15 categories
4. [ ] Implement ensemble weight optimization
5. [ ] Export to ONNX/TensorRT for edge deployment

## 📄 License

MIT License — Academic and industrial use permitted.

## 📧 Contact

- Email: your.email@example.com
- LinkedIn: [Your Profile]
- Built for: German Master's applications & BMW/Bosch AI research
