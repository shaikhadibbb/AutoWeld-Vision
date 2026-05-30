# Dataset Guide

## MVTec AD (Primary Benchmark)

AutoWeld-Vision is trained and evaluated on the **MVTec Anomaly Detection Dataset** (Bergmann et al., CVPR 2019).

- **Official**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Size**: ~4.9 GB (full), ~800 MB (bottle + cable + metal_nut subset)
- **Categories**: 15 texture and object categories
- **License**: Free for academic and non-commercial use
- **Auto-download**: Anomalib handles downloading automatically on first run

The benchmark script (`scripts/run_benchmark.py`) uses three categories representative of industrial quality inspection:

| Category | Train | Test | Defect Types |
|----------|:-----:|:----:|:-------------|
| **Bottle** | 209 | 83 | broken_large, broken_small, contamination |
| **Cable** | 224 | 150 | bent_wire, cable_swap, cut_inner, cut_outer, missing_cable, missing_wire, poke_insulation |
| **Metal Nut** | 220 | 115 | bent, color, flip, scratch |

## Citation

If you use MVTec AD data, please cite:

```bibtex
@inproceedings{bergmann2019mvtec,
  title     = {MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author    = {Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```

## Additional Datasets (Future Work)

- **MVTec AD 2**: https://www.mvtec.com/company/research/datasets/mvtec-ad-2
- **VisA**: https://github.com/amazon-science/visa-anomaly-detection
- **GDXray**: https://github.com/computervision-xray-testing (weld radiographs)
