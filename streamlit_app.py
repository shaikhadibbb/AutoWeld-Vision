import os
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
from PIL import Image

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent))

from test_inspection import run_real_inspection

# Page Config: Premium aesthetics with dark mode themes and custom titles
st.set_page_config(
    page_title="AutoWeld-Vision Operator Terminal",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("🔧 AutoWeld-Vision: Visual Audit Terminal")
st.markdown(
    """
    **Unsupervised Late-Fusion Anomaly Detection for Industrial Quality Control**  
    *This interactive interface executes the AutoWeld-Vision dual-model pipeline to detect surface welding defects (porosity, cracks, and voids) in real-time, archiving an immutable, IATF 16949-compliant quality record.*
    """
)
st.write("---")

# Sidebar Configuration
st.sidebar.header("Inspection Settings")
vin_input = st.sidebar.text_input("Vehicle Identification Number (VIN)", value="BMW-G60-2026")
model_category = st.sidebar.selectbox(
    "Reference Part Profile",
    options=["bottle", "cable", "metal_nut"],
    format_func=lambda x: f"MVTec AD Profile: {x.replace('_', ' ').title()}"
)

# Sidebar System Info
st.sidebar.write("---")
st.sidebar.markdown(
    """
    **System Status:**  
    🟢 Core Engine: Active  
    🟢 Backend: FastAPI (8000)  
    🟢 IATF Logging: Active (`audit_logs/`)  
    
    *Note: If local trained model weights are missing from the `weights/` directory, the pipeline gracefully runs in simulated Demo Mode to showcase performance without environment overhead.*
    """
)

# Image Selection Logic
col_control, col_display = st.columns([1, 2])

with col_control:
    st.subheader("Select Inspection Source")
    source_type = st.radio("Input Source:", ["Use Standard Sample Weld", "Upload Custom Weld Image"])
    
    selected_image_path = None
    
    if source_type == "Use Standard Sample Weld":
        st.write("Using default weld sample: `test_weld.png`")
        selected_image_path = "test_weld.png"
        if not os.path.exists(selected_image_path):
            st.error("Default `test_weld.png` not found in root directory!")
    else:
        uploaded_file = st.file_uploader("Upload Weld Seam Scan:", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Save uploaded image to temporary scratch file
            temp_dir = Path("audit_logs/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_image_path = str(temp_path)
            st.success(f"Successfully uploaded: `{uploaded_file.name}`")
        else:
            st.info("Please upload a weld seam image to trigger the inspection.")

    st.write("---")
    
    # Run Inspection Button
    run_btn = st.button("🚀 Execute Visual Audit", type="primary", use_container_width=True)

# Process Inference
if run_btn and selected_image_path:
    with st.spinner("Processing optical scan and compiling late-fusion coreset memory bank..."):
        try:
            # Run the inspection pipeline
            results = run_real_inspection(
                image_path=selected_image_path,
                category=model_category,
                vin=vin_input
            )
            
            # Display results container
            st.success("Quality inspection complete!")
            
            # Display metrics card row
            col_vin, col_decision, col_score, col_mode = st.columns(4)
            with col_vin:
                st.metric("Logged VIN", results["vin"])
            with col_decision:
                decision_color = "🔴" if results["decision"] == "FAIL" else "🟢"
                st.metric("Quality Decision", f"{decision_color} {results['decision']}")
            with col_score:
                st.metric("Ensembled Anomaly Score", f"{results['anomaly_score']:.4f}")
            with col_mode:
                st.metric("Pipeline Mode", results["mode"].upper())
            
            st.write("---")
            
            # Display Side-by-Side Images (Original and Heatmap Audit Trail)
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.subheader("Original Input Weld Seam")
                st.image(selected_image_path, use_column_width=True)
                
            with col_img2:
                st.subheader("IATF 16949 Visual Audit Report")
                report_file_path = results["audit_report"]
                if os.path.exists(report_file_path):
                    st.image(report_file_path, use_column_width=True)
                    
                    # Read report bytes for download
                    with open(report_file_path, "rb") as file:
                        report_bytes = file.read()
                        
                    st.download_button(
                        label="📥 Download Certified Quality Report",
                        data=report_bytes,
                        file_name=os.path.basename(report_file_path),
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    st.error("Audit report image file could not be retrieved from storage!")
                    
        except Exception as e:
            st.error(f"Inspection failed during pipeline execution: {str(e)}")
            st.info("Ensure the virtual environment is active and all dependencies in requirements-standard.txt are compiled.")
else:
    if source_type == "Upload Custom Weld Image" and selected_image_path is None:
        st.warning("Awaiting uploader input to begin automated inspection.")
