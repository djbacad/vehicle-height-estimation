import os
import cv2
import subprocess
import sys
import streamlit as st
import numpy as np
import torch
from src.track_and_segment2 import process_video
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
torch.classes.__path__ = []


def run_script(script_path, video_suffix):
    """Calls estimate_depth.py or calculate.py with the given suffix."""
    try:
        args = [sys.executable, script_path, video_suffix]
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        st.text(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(e.stderr)
        raise RuntimeError(f"Error running {script_path}: {e.stderr}")

st.set_page_config(page_title="Vehicle Height Estimator", layout="wide")

# We'll store the suffix in session_state
if "video_suffix" not in st.session_state:
    st.session_state["video_suffix"] = None


st.title("🚗 Vehicle Height Estimator – Inference")

# Input for the YouTube link
yt_link = st.text_input("Enter Video Link Here:", value="")

if st.button("Run Pipeline"):
    # Extract suffix from link
    if "watch?v=" in yt_link:
        suffix = yt_link.split("watch?v=")[1].split("&")[0]
    else:
        suffix = yt_link

    st.session_state["video_suffix"] = suffix

    # 1) TRACKING & SEGMENTATION with real-time frames in the browser
    st.write("**Stage 1: Tracking & Segmentation**")
    frame_placeholder = st.empty()
    with st.spinner("Processing frames..."):
        for frame_bytes in process_video(suffix, line_x=500, frame_tolerance=20, stream=True):
            frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame_placeholder.image(frame_np, channels="BGR", use_container_width=True)

    st.success("Tracking & Segmentation complete.")

    # 2) DEPTH ESTIMATION
    st.write("**Stage 2: Depth Estimation**")
    run_script("src/estimate_depth.py", suffix)
    st.success("Depth Estimation complete.")

    # 3) VEHICLE HEIGHT CALCULATION
    st.write("**Stage 3: Vehicle Height Calculation**")
    run_script("src/calculate.py", suffix)
    st.success("Vehicle Height Calculation complete.")

    st.info("Done! Go to 'Results' page to see combined images and logs.")
