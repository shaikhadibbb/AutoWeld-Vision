# Submission Quality Assurance Checklist: 10 Core Verifications

Perform these 10 essential verifications before submitting this repository for academic review or job recruitment.

---

- [ ] **1. Fresh Clone Execution**: Verify the repo clones on a clean environment:
  ```bash
  git clone https://github.com/shaikhadibbb/AutoWeld-Vision.git
  cd AutoWeld-Vision
  ```
- [ ] **2. Dependency Stability**: Confirm dependencies install cleanly on a fresh virtual environment without compilation conflicts:
  ```bash
  python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements-standard.txt
  ```
- [ ] **3. PyTest Passing State**: Run the validation tests and check that all 5 test cases pass successfully:
  ```bash
  PYTHONPATH=. venv/bin/pytest tests/
  ```
- [ ] **4. CLI Inspection Execution**: Run the inspection script in real mode and check for the generated audit trail:
  ```bash
  venv/bin/python3 test_inspection.py --image test_weld.png --vin BMW-G60-2026
  ```
- [ ] **5. Streamlit App Local Start**: Run the visual app locally to ensure the streamlit server initializes correctly:
  ```bash
  venv/bin/streamlit run app.py
  ```
- [ ] **6. Audit Trail Rendering**: Open the output file `audit_logs/report_BMW-G60-2026_*.png` and confirm the bilinear-interpolated heatmap overlay uses the reversed green-to-red (`RdYlGn_r`) colormap.
- [ ] **7. Zero AI Buzzwords**: Search the repository files to confirm that high-risk buzzwords (*leverage, cutting-edge, robust, seamless, delve, landscape, tapestry*) are completely absent.
- [ ] **8. No Citation Bloat**: Ensure `CITATION.bib` has been deleted from the project root and all citations are limited to authentic data references inside `docs/DATASETS.md`.
- [ ] **9. Humble & Clear Job Titles**: Ensure you are listed exclusively as "AI/ML Student" or "Computer Vision Researcher" rather than inflated senior-engineer roles.
- [ ] **10. Working Documentation Links**: Click every relative markdown link in the root `README.md` to verify it maps correctly to the target files in the project.
