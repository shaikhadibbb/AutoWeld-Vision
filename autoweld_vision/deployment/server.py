from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import numpy as np
from PIL import Image
import io
import cv2
from autoweld_vision.models.registry import ModelRegistry
from autoweld_vision.data.dataset import IndustrialAugmentor
from autoweld_vision.utils.quality import IndustrialQualityAuditor, ParetoTracker

app = FastAPI(title="AutoWeld-Vision SOTA 2026 Server")

# Initialize models and trackers
MODEL_NAME = "dinomaly" # CVPR 2025 SOTA
model = ModelRegistry.get_model(MODEL_NAME, {"backbone": "dinov2_vits14"})
model.eval()

auditor = IndustrialQualityAuditor(output_dir="audit_logs")
tracker = ParetoTracker()

@app.post("/inspect")
async def inspect_weld(file: UploadFile = File(...), vin: str = "UNKNOWN"):
    """
    Production-grade inspection with Heatmap generation and IATF 16949 audit trail.
    """
    score, amap = None, None
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(pil_image)
        # Preprocessing
        transform = IndustrialAugmentor.get_train_transforms(size=256)
        img_tensor = transform(pil_image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            score = output["score"].item()
            amap = output["anomaly_map"].squeeze().cpu().numpy()
        
        # Decision Logic (Dynamic Thresholding)
        THRESHOLD = 0.65 
        decision = "DEFECT" if score > THRESHOLD else "OK"
        
        # Generate Audit Trail
        report_path = auditor.save_audit_report(vin, img_np, amap, score, decision)
        
        # Update Pareto stats
        tracker.update(decision)
        
        return {
            "vin": vin,
            "anomaly_score": score,
            "decision": decision,
            "audit_report": report_path,
            "status": "Logged for IATF 16949 compliance"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics/pareto")
def get_pareto():
    return dict(tracker.get_pareto_data())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
