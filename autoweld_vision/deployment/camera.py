import cv2
import torch
import time
from collections import deque
from typing import Any
from autoweld_vision.data.dataset import IndustrialAugmentor


class CameraInspector:
    """
    Priority 2.3: Real-Time Camera Integration.
    Handles RTSP streams and PLC signaling for pass/fail decisions.
    """

    def __init__(self, model: Any, camera_url: str = "0", threshold: float = 0.65):
        self.model = model
        self.camera_url = camera_url
        self.threshold = threshold
        self.preprocessor = IndustrialAugmentor.get_train_transforms(size=256)
        self.buffer = deque(maxlen=30)  # 1-second buffer at 30 FPS

    def start_inspection_loop(self):
        """Main loop for real-time inspection."""
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print(f"Error: Could not open camera at {self.camera_url}")
            return

        print(f"Starting inspection loop on {self.camera_url}...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()

                # 1. Preprocess (Convert to PIL then Tensor)
                from PIL import Image

                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = self.preprocessor(pil_img).unsqueeze(0)

                # 2. Inference
                with torch.no_grad():
                    output = self.model(tensor)
                    score = output["score"].item()

                # 3. Decision & Signaling
                is_defective = score > self.threshold
                self._send_plc_signal(is_defective)

                # 4. Latency Tracking
                latency_ms = (time.time() - start_time) * 1000

                # Visual Overlay
                color = (0, 0, 255) if is_defective else (0, 255, 0)
                label = f"DEFECT ({score:.2f})" if is_defective else f"OK ({score:.2f})"
                cv2.putText(
                    frame,
                    f"{label} | {latency_ms:.1f}ms",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

                cv2.imshow("AutoWeld-Vision Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _send_plc_signal(self, is_defective: bool):
        """Simulates sending a signal to a welding robot PLC."""
        if is_defective:
            # print("[PLC] REJECT SIGNAL SENT")
            pass
        else:
            # print("[PLC] PASS SIGNAL SENT")
            pass
