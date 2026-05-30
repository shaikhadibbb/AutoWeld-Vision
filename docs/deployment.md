# Factory-Floor Deployment Guide: Edge Integration, Containers, and System Orchestration

This document details the practical roadmap for deploying AutoWeld-Vision in a high-speed automotive assembly environment. It covers hardware specs, container configurations, industrial camera protocols, database schemas, MQTT broker integrations, and long-term model maintenance.

---

## 1. Hardware Architecture & Requirements

### 1.1 Edge Processing Nodes
To sustain real-time inspection latencies below 50 milliseconds, edge compute modules must be positioned close to the physical welding stations.

| Component | Minimum Specification | Recommended Specification (Production) |
| :--- | :--- | :--- |
| **Compute Node** | Industrial PC / Intel i5 CPU | **NVIDIA Jetson Orin NX (20GB)** or **Jetson Orin Nano (8GB)** |
| **RAM** | 8 GB DDR4 | **16 GB DDR5** (Shared memory for GPU tensors) |
| **Storage** | 64 GB SATA SSD | **256 GB NVMe M.2 SSD** (High-speed write for audit logs) |
| **GPU** | CPU fallback enabled | **Ampere GPU with 1024 CUDA Cores & Tensor Cores** |
| **Cooling** | Active fan cooling | **Fanless Rugged Aluminum Chassis** (IP50 rating to withstand metallic shop-floor dust) |

### 1.2 Industrial Camera Specifications
The optical sensor must capture sharp, high-contrast images under rapid mechanical movement:
* **Camera Model**: Basler ace 2 Basic (a2A3840-45gcBAS) or equivalent GigE Vision industrial camera.
* **Resolution**: 4K UHD ($3840 \times 2160$ pixels) to capture micro-pores (<0.5mm).
* **Lens**: 16mm Fixed Focal Length lens with manual locking focus rings.
* **Lighting**: Dual coaxial diffuse LED panel lights (prevents direct glare reflections from curved aluminum or steel seams).
* **Connection**: Gigabit Ethernet (GigE) with Power-over-Ethernet (PoE) to isolate signal noise.

---

## 2. Production Containerization

To avoid dependency conflicts and ensure identical environments from development to edge devices, the software must be containerized.

### 2.1 Complete Production Dockerfile
Create a `Dockerfile` at the root of the project to build the optimized edge container:

```dockerfile
# Dockerfile
# Optimized for NVIDIA JetPack (L4T) or CUDA-enabled Linux hosts
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set environment parameters
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and industrial camera protocols
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency definition
COPY requirements-standard.txt .

# Install dependencies (utilizes cache if requirements are unchanged)
RUN pip install --no-cache-dir -r requirements-standard.txt

# Copy application source code
COPY autoweld_vision/ ./autoweld_vision/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY test_inspection.py .
COPY test_weld.png .

# Create persistent directories for audit reports and model checkpoints
RUN mkdir -p audit_logs weights results

# Expose ports for FastAPI (8000) and Streamlit Dashboard (8501)
EXPOSE 8000
EXPOSE 8501

# Default runtime: Launch the FastAPI REST service
CMD ["python", "autoweld_vision/deployment/server.py"]
```

### 2.2 Docker Compose Configuration
Orchestrate the API service, Streamlit dashboard, and localized database using `docker-compose.yml`:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # FastAPI REST backend for high-speed inspection
  api-service:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: autoweld_api
    restart: always
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - DB_HOST=db-service
      - DB_USER=weld_admin
      - DB_PASSWORD=weld_secure_pass
      - DB_NAME=autoweld_db
    ports:
      - "8000:8000"
    volumes:
      - ./audit_logs:/app/audit_logs
      - ./weights:/app/weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - db-service

  # Streamlit UI for operator terminal and quality logs monitoring
  operator-dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: autoweld_dashboard
    restart: always
    command: ["streamlit", "run", "autoweld_vision/deployment/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ports:
      - "8501:8501"
    volumes:
      - ./audit_logs:/app/audit_logs
    depends_on:
      - api-service

  # Archive DB for long-term IATF 16949 audit compliance tracking
  db-service:
    image: postgres:15-alpine
    container_name: autoweld_db_postgres
    restart: always
    environment:
      - POSTGRES_USER=weld_admin
      - POSTGRES_PASSWORD=weld_secure_pass
      - POSTGRES_DB=autoweld_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## 3. Installation & Setup on Edge Device (NVIDIA Jetson)

### 3.1 JetPack Setup
1. Flash the edge hardware using the latest NVIDIA JetPack 6.0 (which includes Ubuntu 22.04 LTS, CUDA 12.x, TensorRT 8.6, and cuDNN).
2. Install the NVIDIA Container Toolkit to pass GPU controls through to Docker containers:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### 3.2 Compiling PyTorch Graphs to TensorRT (Engine Compilation)
To achieve sub-30ms latencies on NVIDIA edge hardware, convert the PyTorch model graph into a static TensorRT engine:

1. **ONNX Export**:
   Export PyTorch graph to ONNX representation:
   ```python
   import torch
   # Load trained model weights
   model = load_inspection_model("bottle")
   dummy_input = torch.randn(1, 3, 256, 256)
   torch.onnx.export(
       model,
       dummy_input,
       "weights/patchcore_bottle.onnx",
       export_params=True,
       opset_version=17,
       do_constant_folding=True,
       input_names=["input"],
       output_names=["anomaly_map", "score"],
       dynamic_axes={"input": {0: "batch_size"}}
   )
   ```
2. **TensorRT Compilation**:
   Compile the ONNX file into an optimized `.engine` format using the command line:
   ```bash
   trtexec --onnx=weights/patchcore_bottle.onnx \
           --saveEngine=weights/patchcore_bottle.engine \
           --fp16 \
           --workspace=4096
   ```
   *Note on INT8 Quantization (Future Work):* For maximum throughput, INT8 calibration requires a collection of ~200 normal weld representative images to compile an entropy calibration cache, mapping floating-point tensors down to 8-bit integers without losing pixel AUROC localization precision.

---

## 4. Integration with Factory Systems

### 4.1 REST API Specification (FastAPI)
The API service receives inspection requests from a Programmable Logic Controller (PLC) or factory PC.

* **Endpoint**: `POST http://localhost:8000/inspect`
* **Headers**: `Content-Type: multipart/form-data`
* **Body Parameters**:
  - `file`: Optical image binary (JPEG or PNG).
  - `vin`: String coordinate representing vehicle identification coordinate (e.g. `VIN-BMW-G60-12345`).

**Sample REST Request (via curl):**
```bash
curl -X POST "http://localhost:8000/inspect?vin=BMW-G60-2026" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_weld.png"
```

**JSON Response Format:**
```json
{
  "vin": "BMW-G60-2026",
  "decision": "PASS",
  "anomaly_score": 0.3241,
  "threshold": 0.50,
  "inference_time_ms": 47.2,
  "audit_report_path": "audit_logs/report_BMW-G60-2026_20260530_234747.png",
  "timestamp": "2026-05-30T23:47:47"
}
```

### 4.2 MQTT Broker Integration (SCADA / PLC Triggering)
In typical PLC environments, we hook the inspection container into an MQTT network (e.g., Eclipse Mosquitto) for event-driven automation:
* **Trigger Topic**: `factory/station_12/trigger_scan`
  - *Payload*: `{"vin": "BMW-G60-1234", "camera_id": "cam_left"}`
* **Quality Decisions Output Topic**: `factory/station_12/inspection_decision`
  - *Payload*: `{"vin": "BMW-G60-1234", "decision": "FAIL", "score": 0.7420, "timestamp": "2026-05-30T23:47:47"}`
  - *PLC Response:* Upon receiving `{"decision": "FAIL"}` on this topic, the PLC trips a pneumatic relay to push the chassis off the main line into the quarantine repair area.

### 4.3 PostgreSQL Relational Database Schema
Store inspection metadata to satisfy automotive regulatory audits. Run this SQL script to initialize the PostgreSQL tracking tables:

```sql
-- Database Schema for IATF 16949 Auditing Trackability
CREATE TABLE IF NOT EXISTS weld_inspections (
    id SERIAL PRIMARY KEY,
    vin VARCHAR(50) NOT NULL UNIQUE,
    decision VARCHAR(10) NOT NULL CHECK (decision IN ('PASS', 'FAIL')),
    anomaly_score REAL NOT NULL,
    decision_threshold REAL DEFAULT 0.50,
    inference_time_ms REAL NOT NULL,
    audit_report_path VARCHAR(255) NOT NULL,
    model_version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc')
);

CREATE INDEX idx_weld_inspections_vin ON weld_inspections(vin);
CREATE INDEX idx_weld_inspections_decision ON weld_inspections(decision);
CREATE INDEX idx_weld_inspections_created_at ON weld_inspections(created_at);
```

---

## 5. Monitoring & Maintenance

### 5.1 Model Drift Detection
During manufacturing, physical factors drift:
* Camera lenses accumulate light dust, blurring the image.
* Factory lighting degrades, altering the baseline illumination.
* The raw steel coil batches change, shifting standard weld coloration.

This is **model drift**. It manifests as a gradual rise in average anomaly scores for completely normal weld seams. To monitor drift, we track a sliding window ($N = 1000$ consecutive passes) of PASS scores. If the rolling mean score moves from $\mu=0.18$ to $\mu \ge 0.35$ while physical review confirms no change in quality:
1. Trigger a warning in the Streamlit Dashboard.
2. Log an operational warning.
3. Queue images for a recalibration retraining loop.

### 5.2 Scheduled Retraining & Recalibration
Rather than training a massive network from scratch, PatchCore makes retraining straightforward:
1. Clear old drift-prone images from the database.
2. Gather 100 newly confirmed "normal" weld images from the production run.
3. Append these feature patches to the coreset memory bank using the greedy search script:
   ```bash
   python scripts/run_benchmark.py --categories bottle --output results/
   ```
4. Verify the updated model against a standard validation set before overriding the edge container `.engine` weights.

### 5.3 Automated Failure Alert Protocols
If a core camera connection drops or the REST API latency exceeds 200ms:
- The backend FastAPI triggers an immediate warning code.
- A critical payload is published to the MQTT topic `factory/station_12/system_status`:
  ```json
  {"status": "ERROR", "code": "CAMERA_CONNECTION_LOST", "action_required": "HALT_LINE_MANUAL_OVERRIDE"}
  ```
- This safety mechanism ensures that if the AI monitoring goes offline, the line stops, preventing uninspected safety-critical joints from advancing.
