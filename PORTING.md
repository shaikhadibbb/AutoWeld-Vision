# Porting Notes: Antigravity → Standard Python

## Original Environment
- Platform: Antigravity AI IDE (cloud/sandboxed)
- Python: 3.14.4 (Homebrew custom build)
- Packages: Modified anomalib with custom wrappers

## Standard Environment  
- Platform: macOS Apple Silicon (local)
- Python: 3.11.14 (Homebrew standard)
- Packages: PyPI anomalib 1.1.0, PyTorch 2.3.1

## Changes Made
1. Updated import paths to match anomalib 1.1.0 API
2. Replaced custom model wrappers with anomalib native classes
3. Added explicit data download step (MVTec AD not auto-downloaded)
4. Documented all version numbers with verification commands
5. Removed all benchmark claims pending real training
6. Fixed dependency conflicts (NumPy 1.26.4 for imgaug compatibility, OpenCV 4.9.0)

## Verification
All commands in README tested on:
- MacBook Air M1/M2
- Python 3.11.14
- PyTorch 2.3.1 (MPS backend)

## Known Limitations
- Training on CPU/MPS is slower than CUDA (~20 min vs ~2 min)
- MVTec AD must be downloaded manually (404 on anomalib auto-download)
- Jetson Orin benchmarks require actual hardware
