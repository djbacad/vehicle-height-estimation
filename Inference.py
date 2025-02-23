import os
import cv2
import subprocess
import sys
import argparse
import streamlit as st
import numpy as np
from src.track_and_segment2 import process_video as track_and_segment

# Paths
videos_dir = "data/videos"
# processed_video_dir = "outputs/processed_videos"  # Directory for processed videos

# os.makedirs(processed_video_dir, exist_ok=True)

def run_script(script_path, video_filename, enable_visualization=False):
    """Runs a script using subprocess and passes the video filename as an argument."""
    try:
        args = [sys.executable, script_path, video_filename]
        if enable_visualization:
            args.append("--enable_visualization")
        result = subprocess.run(
            args,  # Use the current Python executable
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise RuntimeError(f"Error running {script_path}: {e.stderr}")

def process_video_pipeline(video_filename, enable_visualization=False, status_placeholder=None, frame_placeholder=None):
    """
    Complete pipeline to process a video: track, segment, estimate depth, calculate height,
    and handle additional processing steps.
    Includes streaming frames for visualization if enabled.
    """
    try:
        # Step 1: Track and segment
        if status_placeholder:
            status_placeholder.write("Step 1: Tracking and segmenting vessels from frames...")
        for frame in track_and_segment(
            argparse.Namespace(video_filename=video_filename, enable_visualization=enable_visualization), stream=True
        ):
            if enable_visualization and frame_placeholder:
                frame_placeholder.image(frame, channels="BGR", width=800)

        # Step 2: Depth estimation
        if status_placeholder:
            status_placeholder.write("Step 2: Estimating depth of objects...")
        run_script("src/estimate_depth.py", video_filename)

        # Step 3: Calculate vessel height
        if status_placeholder:
            status_placeholder.write("Step 3: Calculating vessel height...")
        run_script("src/calculate.py", video_filename)

        # Step 4: Inserting OCR results
        if status_placeholder:
            status_placeholder.write("Step 4: Inserting OCR results...")
        run_script("src/recognize_text.py", video_filename)

        # Step 5: Deduplicating
        if status_placeholder:
            status_placeholder.write("Step 5: Deduplicating results...")
        run_script("src/deduplicate.py", video_filename)

        # Prepare results for display
        base_name = os.path.splitext(video_filename)[0]
        combined_output_dir = os.path.join("outputs", base_name, "combined")
        logs_output_dir = os.path.join("outputs", base_name, "_logs")

        combined_images = [
            os.path.join(combined_output_dir, img)
            for img in os.listdir(combined_output_dir)
            if img.endswith(".png")
        ]
        log_file = os.path.join(logs_output_dir, f"{base_name}_logs.csv")
        processed_video_path = os.path.join(processed_video_dir, f"{base_name}_processed.mp4")

        if not os.path.exists(processed_video_path):
            raise FileNotFoundError(f"Processed video not found: {processed_video_path}")

        return combined_images, log_file, processed_video_path

    except Exception as e:
        return f"An error occurred: {str(e)}", None, None

# Streamlit app
st.set_page_config(page_title="Vessel Height Estimator", layout="wide")
st.sidebar.title("🚀 Inference") 
st.title("⛴️🚢⛵🚤 Vessel Height Estimator")

st.markdown("""Use the sidebar to navigate between processing and results pages.""")

# Dropdown for selecting videos
processed_videos = set(os.listdir("outputs"))  # Get all processed video folders from the outputs directory
video_options = [
    file for file in os.listdir(videos_dir) if file.lower().endswith(".mp4") and os.path.splitext(file)[0] not in processed_videos
]

if video_options:
    video_filename = st.selectbox("Select a video", video_options)
else:
    st.write("All videos have already been processed.")

# Checkbox for enabling visualization
enable_visualization = st.checkbox("Enable Visualization", value=False)

# Button to process
if st.button("Run Inference"):
    status_placeholder = st.empty()  # Placeholder for dynamic status updates
    frame_placeholder = None

    status_placeholder.write("Initializing video pipeline...")

    if enable_visualization:
        frame_placeholder = st.empty()
        status_placeholder.write("Visualization enabled. Preparing to stream and save frames...")

    try:
        # Define the processed video path dynamically
        base_name = os.path.splitext(video_filename)[0]  # Extract video name without extension
        processed_video_dir = os.path.join("outputs", base_name, "playback_tracking")
        os.makedirs(processed_video_dir, exist_ok=True)  # Ensure the directory exists

        processed_video_path = os.path.join(processed_video_dir, f"{base_name}_processed.mp4")

        # Initialize VideoWriter for saving the processed video
        fourcc = cv2.VideoWriter_fourcc(*"H264")  # Codec for MP4 format
        video_writer = None

        # Run the pipeline
        status_placeholder.write("Step 1: Tracking and segmenting frames...")
        for frame_bytes in track_and_segment(
            argparse.Namespace(video_filename=video_filename, enable_visualization=enable_visualization), stream=True
        ):
            # Decode the bytes to a NumPy array
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Display frames in the Streamlit web app
            if enable_visualization and frame_placeholder:
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            # Save the frame to the processed video
            if video_writer is None:
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(processed_video_path, fourcc, 30, (width, height))
            video_writer.write(frame)

        # Release the video writer after all frames are processed
        if video_writer is not None:
            video_writer.release()

        # Continue with other pipeline steps
        combined_images, log_file, _ = process_video_pipeline(video_filename, enable_visualization, status_placeholder)

        # Display the saved processed video
        if os.path.exists(processed_video_path):
            st.video(processed_video_path)

        # Display results
        status_placeholder.success("Pipeline completed successfully!")
        for img_path in combined_images:
            st.image(img_path, caption=os.path.basename(img_path))

    except Exception as e:
        status_placeholder.error(f"An error occurred: {e}")
    finally:
        # Ensure the video writer is released
        if video_writer is not None:
            video_writer.release()


