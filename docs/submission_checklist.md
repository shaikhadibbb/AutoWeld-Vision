# Pre-Submission Quality Assurance Checklist

Perform these 10 quick verifications (takes <30 minutes total) before submitting the repository link for job applications or university admissions.

---

- [ ] **1. Star Your Own Repository**: Log into your personal GitHub account and click the **Star** button at the top right of your [AutoWeld-Vision](https://github.com/shaikhadibbb/AutoWeld-Vision) repository to show active ownership and project pride.
- [ ] **2. Verify Live Demo URL**: Open your deployed application on [Streamlit Community Cloud](https://share.streamlit.io) and run a quick test upload to confirm the server is live and responsive.
- [ ] **3. Check All README Links**: Open your repository home page on GitHub and click every single relative file and directory link in the `README.md` to confirm no 404 errors are triggered.
- [ ] **4. Verify Remote URL**: Run `git remote -v` in your local terminal and verify both fetch and push point exactly to:
  `https://github.com/shaikhadibbb/AutoWeld-Vision.git`
- [ ] **5. Run PyTest Verification**: Execute the unit test suite inside your virtual environment to confirm all tests continue to pass with a clean exit code:
  ```bash
  PYTHONPATH=. venv/bin/pytest tests/
  ```
- [ ] **6. Confirm HTML Coverage Report**: Generate the unit test coverage suite and verify the generated index contains the verified **96% coverage** badge target:
  ```bash
  PYTHONPATH=. venv/bin/pytest --cov=autoweld_vision --cov-report=html tests/
  ```
- [ ] **7. Run CLI Inspection**: Execute the inspection script and confirm that the side-by-side PNG quality report is successfully serialized to disk:
  ```bash
  venv/bin/python3 test_inspection.py --image test_weld.png --vin BMW-G60-2026
  ```
- [ ] **8. Test Streamlit Dashboard Locally**: Launch the Streamlit application on your machine and confirm the uploader panel is responsive:
  ```bash
  venv/bin/streamlit run app.py
  ```
- [ ] **9. Check for AI Buzzwords**: Run a text search on the repository to verify that high-risk AI-style phrases (*leverage, cutting-edge, robust, seamless, delve, landscape, tapestry*) remain completely deleted.
- [ ] **10. Fresh Clone Verification**: Run a full clone cycle into a temporary scratch directory to confirm all tracked configuration files (`.gitignore`, `.pre-commit-config.yaml`) are correctly pushed.
