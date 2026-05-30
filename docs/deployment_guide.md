# Deployment Guide: Production Deployment and API Services

AutoWeld-Vision includes an interactive dashboard and a backend FastAPI service for assembly-line integration.

## 1. Running the FastAPI Server
The backend FastAPI server handles incoming high-speed optical inspection requests. To run the API server locally:
```bash
python autoweld_vision/deployment/server.py
```
Endpoint: `http://localhost:8000/inspect` (POST)  
Expects: `file` (image) and `vin` (query string).

## 2. Launching the Streamlit Dashboard
The operator terminal is built on Streamlit, presenting real-time inspections, Six Sigma quality KPIs, Pareto distribution charts, and historical audit logs.
```bash
streamlit run autoweld_vision/deployment/dashboard.py
```
* **Live Monitor**: Visualizes dynamic sensor feeds and outputs model ensembled decisions.
* **Inference Lab**: Allows manual testing of custom weld joints with immediate visual feedback of the official generated IATF 16949 audit trail PNG.
* **Audit Explorer**: Explores, filters, and downloads stored audit reports.
