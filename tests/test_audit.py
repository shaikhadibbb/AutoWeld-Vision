"""
Unit tests for the IATF 16949 visual quality audit report generation.
"""

import os
from PIL import Image
from autoweld_vision.utils.demo_mode import run_demo_inspection


def test_audit_report_generation() -> None:
    """Verifies that the audit report PNG is generated, matches naming conventions, and is valid."""
    vin = "WBA-TEST-999-XYZ"
    dummy_img_path = "test_weld_dummy.png"

    # 1. Create a temporary dummy input image
    img = Image.new("RGB", (128, 128), color=(200, 200, 200))
    img.save(dummy_img_path)

    try:
        # 2. Run the inspection report generation
        result = run_demo_inspection(dummy_img_path, vin=vin)

        # 3. Assert results
        assert result["vin"] == vin
        assert result["mode"] == "demo"

        report_path = result["audit_report"]
        assert os.path.exists(report_path)
        assert vin in report_path

        # 4. Open the generated report and assert it's a valid PIL image
        report_img = Image.open(report_path)
        w, h = report_img.size
        assert w > 300
        assert h > 300

    finally:
        # Cleanup
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)
        if "result" in locals() and os.path.exists(result["audit_report"]):
            os.remove(result["audit_report"])
