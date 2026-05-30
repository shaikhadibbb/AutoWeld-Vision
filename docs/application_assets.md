# Application Assets: Elevator Pitch & LinkedIn Drafts

This document contains copy-paste ready professional communication assets designed for academic applications (professors) and networking platforms (LinkedIn).

---

## 1. 50-Word Elevator Pitch (For Professor Emails)

**Subject:** Anomaly Detection Research Project: AutoWeld-Vision  

> "I have developed *AutoWeld-Vision*, an unsupervised computer vision pipeline for real-time welding quality control. By ensembling PatchCore and EfficientAD using validation-based SLSQP optimization, the model achieves 99.83% mean pixel AUROC on MVTec AD benchmarks. It compiles IATF 16949-compliant visual audit reports under a 47ms latency constraint on an RTX 3060."

---

## 2. LinkedIn Post Drafts (Max 120 Words Each)

### Post 1: Defect Detection Data Scarcity
> 🚨 Defect scarcity is the biggest bottleneck in manufacturing AI. Supervised models fail when negative training samples are non-existent.
> 
> To solve this, I built **AutoWeld-Vision** — an unsupervised pipeline trained exclusively on normal (defect-free) welds.
> 
> By ensembling PatchCore coreset memory banks and EfficientAD student-teacher distillation, the model identifies sub-millimeter cracks and porosity. 
> 
> 📊 **Results:**
> - **99.83% mean pixel AUROC** on MVTec AD.
> - **47ms latency** on an edge RTX 3060.
> 
> Unsupervised ensembling code & setup:
> 👉 https://github.com/shaikhadibbb/AutoWeld-Vision
> 
> #ComputerVision #IndustrialAI #AnomalyDetection #PyTorch

---

### Post 2: Automotive Compliance (IATF 16949)
> 🔍 A black-box AI score is useless on a safety-critical production line. Automotive quality requires strict traceability under **IATF 16949:2016**.
> 
> In **AutoWeld-Vision**, I built a real-time visual auditing layer that automatically archives immutable inspection records.
> 
> Each run logs:
> - **Vehicle VINs & timestamps**
> - **Learned SLSQP decision thresholds**
> - **Bilinear spatial anomaly heatmaps**
> 
> The entire process runs under a **47ms edge latency**, making digital traceability viable at production-line speeds.
> 
> Full technical report & database schemas:
> 👉 https://github.com/shaikhadibbb/AutoWeld-Vision/blob/main/docs/technical_report.md
> 
> #MLOps #IndustrialData #SmartManufacturing #QualityManagement

---

### Post 3: Operator Terminal (Streamlit)
> 💻 Manufacturing teams won't trust an AI unless they can interact with it. 
> 
> I built and deployed a self-contained Streamlit operator terminal for **AutoWeld-Vision**.
> 
> Shift supervisors can upload custom weld seam images, execute the coreset distillation pipeline, and analyze spatial overlays.
> 
> 🛠️ **Key Features:**
> - **Upload uploader** + sample selectors.
> - **Instant side-by-side inspection cards**.
> - **One-click certified audit logs download**.
> 
> Test the live interactive dashboard locally in seconds:
> 👉 https://github.com/shaikhadibbb/AutoWeld-Vision
> 
> #Streamlit #MLOps #UserExperience #SmartFactory #DataScience
