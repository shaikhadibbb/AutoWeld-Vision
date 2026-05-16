# Training Roadmap for AutoWeld-Vision

## Quick Start (One Model, One Category)

Train PatchCore on MVTec AD "bottle" as baseline:

```bash
# Install anomalib CLI
pip install anomalib

# Download MVTec AD manually from https://www.mvtec.com/company/research/datasets/mvtec-ad
# Extract to ./datasets/mvtec/

# Train
anomalib train --model Patchcore \
  --data anomalib.data.datamodules.image.mvtecad.MVTecAD \
  --data.root ./datasets/mvtec \
  --data.category bottle
```

Expected time: ~20 minutes on Apple Silicon (M1/M2), ~2 minutes on NVIDIA GPU.

## Full Benchmark (All Models, All Categories)

```bash
# Run all models on all MVTec AD categories
python scripts/full_benchmark.py --config configs/experiment/benchmark_all.yaml
```

Expected time: ~8 hours on NVIDIA RTX 4090.

## Edge Optimization

After training, export to TensorRT:

```bash
python scripts/export_tensorrt.py --model_path results/patchcore/model.ckpt
```
