import os
import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO

# Paths
outputs_dir = "outputs"

# Page title
st.title("📊 Processed Results")
st.sidebar.title("🔍 Results")

# List already processed videos
processed_videos = [folder for folder in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, folder))]

if processed_videos:
    video_selection = st.selectbox("Select a video to view results:", processed_videos)
    if video_selection:
        # Paths for results
        base_output_dir = os.path.join(outputs_dir, video_selection)
        combined_output_dir = os.path.join(base_output_dir, "combined")
        logs_output_dir = os.path.join(base_output_dir, "_logs")
        playback_dir = os.path.join(base_output_dir, "playback_tracking")
        tracked_vessels_dir = os.path.join(base_output_dir, "tracked_vessels")

        # Display tracked vessel images
        st.header("Still Photos of Detected Vessels:")
        tracked_images = [
            os.path.join(tracked_vessels_dir, img)
            for img in os.listdir(tracked_vessels_dir)
            if img.endswith(".png")
        ]
        for img_path in tracked_images:
            st.image(img_path, caption=os.path.basename(img_path))

        # Create a ZIP file for tracked vessels
        if tracked_images:
            # Create an in-memory ZIP file
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for img_path in tracked_images:
                    zip_file.write(img_path, arcname=os.path.basename(img_path))
            zip_buffer.seek(0)

            # Display download button for ZIP file
            st.download_button(
                label="Download All Tracked Vessel Images as ZIP",
                data=zip_buffer,
                file_name=f"{video_selection}_tracked_vessels.zip",
                mime="application/zip"
            )

        # Display combined images
        combined_images = [
            os.path.join(combined_output_dir, img)
            for img in os.listdir(combined_output_dir)
            if img.endswith(".png")
        ]
        st.header("Depth Heatmap & Masks Visualization:")
        for img_path in combined_images:
            st.image(img_path, caption=os.path.basename(img_path))

        # Display log file as table
        log_files = [file for file in os.listdir(logs_output_dir) if file.endswith(".csv")]
        if log_files:
            log_file_path = os.path.join(logs_output_dir, log_files[0])
            # Load CSV into a Pandas DataFrame
            df = pd.read_csv(log_file_path)

            st.header("Logs:")
            st.dataframe(df, use_container_width=True)  # Display DataFrame as a table in Streamlit

        # Provide download option for the processed video
        st.header("Playback:")
        processed_videos = [
            os.path.join(playback_dir, vid)
            for vid in os.listdir(playback_dir)
            if vid.endswith(".mp4")
        ]
        if processed_videos:
            processed_video_path = processed_videos[0]
            # Load the video file into memory
            with open(processed_video_path, "rb") as video_file:
                video_data = video_file.read()

            # Display download button for the video
            st.download_button(
                label="Download Processed Playback Video",
                data=video_data,
                file_name=os.path.basename(processed_video_path),
                mime="video/mp4"
            )
else:
    st.info("No processed videos available yet. Please process a video first!")
