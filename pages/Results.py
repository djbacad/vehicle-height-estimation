import os
import pandas as pd
import streamlit as st

st.title("🚗 Vehicle Height Estimator – Results")

# Check if the video suffix is available in session_state
if "video_suffix" not in st.session_state or st.session_state["video_suffix"] is None:
    st.warning("No video has been processed yet. Please run the pipeline from the main page first.")
else:
    suffix = st.session_state["video_suffix"]
    base_out = os.path.join("outputs", suffix)
    combined_dir = os.path.join(base_out, "combined")
    logs_dir = os.path.join(base_out, "_logs")
    logs_file = os.path.join(logs_dir, f"{suffix}_logs.csv")

    st.subheader("Combined Visualization Images")
    if os.path.exists(combined_dir):
        combined_imgs = [
            os.path.join(combined_dir, f)
            for f in os.listdir(combined_dir)
            if f.lower().endswith(".png")
        ]
        if combined_imgs:
            for img_path in combined_imgs:
                st.image(img_path, caption=os.path.basename(img_path))
        else:
            st.info("No combined images found yet.")
    else:
        st.info(f"No 'combined' folder found in outputs/{suffix}.")

    st.subheader("Logs Table")
    if os.path.exists(logs_file):
        df = pd.read_csv(logs_file)
        st.dataframe(df)
    else:
        st.info(f"No logs file found at {logs_file}.")
