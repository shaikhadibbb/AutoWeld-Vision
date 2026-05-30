# Submission Quality Assurance Checklist

This checklist details the 20+ critical checks to run before submitting the repository link for job applications or university admissions. It ensures that your code is reproducible, highly professional, and grounded in standard industrial software engineering practices.

---

### 1. Repository and Reproduction
- [ ] **Fresh Clone Verification**: Run `git clone <repo-url>` into a completely new, empty directory to ensure all tracked assets and configuration files are present in the remote repository.
- [ ] **Clean Environment Setup**: Ensure that running `python3.11 -m venv venv` and `source venv/bin/activate` succeeds without shell exceptions.
- [ ] **Requirement Installation**: Run `pip install -r requirements-standard.txt` to verify that there are no package conflicts or unresolvable wheel compilation errors.
- [ ] **Pinned Version Verification**: Double-check `requirements-standard.txt` to ensure key libraries (especially `torch`, `anomalib`, `matplotlib`, and `scipy`) are pinned to standard production versions.
- [ ] **No Hardcoded Paths**: Check that all scripts use relative paths based on the project root or configure absolute paths via environment variables.

---

### 2. Code Execution and Testing
- [ ] **CLI Inspection Execution**: Verify that `python test_inspection.py --image test_weld.png --vin VERIFY-WELD` runs and completes successfully.
- [ ] **Demo Mode Fallback**: Rename the `weights/` directory temporarily and run `test_inspection.py` to confirm that the pipeline falls back gracefully to "Demo Mode" instead of crashing.
- [ ] **Report Generation**: Verify that running `test_inspection.py` outputs a side-by-side PNG in `audit_logs/` containing the original image, anomaly heatmap overlay, and formatted headers.
- [ ] **Audit Trail Integrity**: Verify that the generated audit trail image uses the reversed green-to-red (`RdYlGn_r`) colormap and accurately binds the custom input VIN.
- [ ] **Benchmark Execution**: Run `python scripts/run_benchmark.py --categories bottle --output results/` to confirm that the training, coreset extraction, and weight optimization pipeline execute successfully.
- [ ] **Unit Tests Suite**: Run `pytest tests/` (or `python -m pytest`) to ensure that all 3 unit test files pass with a test coverage of $\ge 90\%$.

---

### 3. Documentation and Verification
- [ ] **Zero AI Buzzwords**: Review the entire repository (especially `README.md` and new docs) to confirm that high-risk buzzwords (*leverage, cutting-edge, robust, seamless, delve, landscape, tapestry*) are completely removed.
- [ ] **Broken Link Check**: Click every single relative file link in the root `README.md` and documentation files to ensure they successfully navigate to the target files in the project.
- [ ] **Mermaid Diagram Rendering**: View the `README.md` in a markdown renderer to verify that the Mermaid flowchart is correctly drawn with proper, clean labels.
- [ ] **LaTeX Math Rendering**: Open `docs/technical_report.md` to ensure all mathematical formulas and matrices render correctly in standard LaTeX layout without compilation syntax errors.
- [ ] **No Inflation or Hype**: Ensure that benchmark numbers are presented honestly, including an explicit explanation that 100% Image AUROC on MVTec is a behavior of standard pre-trained Wide-ResNet50 backbones.
- [ ] **Weld Validation Honesty**: Ensure that you clearly state weld-specific dataset adaptation (GDXray, KolektorSDD) is an ongoing phase, maintaining scientific honesty.

---

### 4. Code Quality and Styling
- [ ] **Clean Import Structures**: Ensure all import statements are clean, logical, and that there are no circular dependencies in `autoweld_vision/`.
- [ ] **Consistent Type Signatures**: Verify that all core Python functions in `autoweld_vision/models/` and `autoweld_vision/utils/` utilize explicit type hinting (PEP 484).
- [ ] **Docstring Formatting**: Check that all classes and methods have explicit docstrings outlining arguments, returns, and raised exceptions.
- [ ] **Ruff / Lint Validation**: Run a python linter (e.g. `ruff check .` or `flake8 .`) to confirm there are no syntax warnings, unused imports, or style violations.
- [ ] **Clean Git History**: Run `git status` to make sure no build files, cache folders (`__pycache__`, `.pytest_cache`, `.ruff_cache`), or massive raw weight files are accidentally staged for commit. Make sure `.gitignore` successfully excludes these.

---

### 5. Final Submissions check
- [ ] **Contact Details**: Double-check that your email, LinkedIn, and GitHub links in the `README.md` are up to date and correct.
- [ ] **No Inflated Job Titles**: Verify that you are listed as "AI/ML Student" or "Computer Vision Researcher" rather than "Senior Lead AI Architect".
- [ ] **No Citation Block for Student Project**: Confirm that the `@software` bibtex block has been removed as requested.
- [ ] **Checklist Compliance**: Review each item in this file, ticking them off only after verifying the behavior on a clean terminal window.
