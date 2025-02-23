import os
import argparse
import pandas as pd
import numpy as np

def deduplicate_logs(video_filename):
    # Define the log file path
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    logs_output_dir = os.path.join('outputs', base_name, '_logs')
    logs_file = os.path.join(logs_output_dir, f"{base_name}_logs.csv")

    if not os.path.exists(logs_file):
        print(f"Logs file not found: {logs_file}")
        return

    # Load the logs CSV
    df = pd.read_csv(logs_file)

    # Ensure the relevant columns are in the data
    required_columns = ['timestamp', 'vessel_height_actual', 'vessel_center_depth',
                        'vessel_depth_average', 'vessel_bbox_height_px', 'vessel_mask_height_px', 'image_filename']
    if not all(col in df.columns for col in required_columns):
        print("The logs file is missing required columns.")
        return

    # Convert timestamp to seconds for comparison
    df['timestamp_seconds'] = pd.to_timedelta(df['timestamp']).dt.total_seconds()

    # Group rows with close timestamps (assume ~1 second apart is the same boat)
    df['timestamp_group'] = (df['timestamp_seconds'] // 4).astype(int)

    # Initialize list to store deduplicated rows
    deduplicated_rows = []

    # Process each timestamp group
    for _, group in df.groupby('timestamp_group'):
        if len(group) > 1:
            # Rule 2a: Remove rows with vessel_height_actual values that deviate significantly
            median = group['vessel_height_actual'].median()
            mad = np.median(np.abs(group['vessel_height_actual'] - median))
            group = group[np.abs(group['vessel_height_actual'] - median) <= 3 * mad]

            # Rule 2b: Remove rows where vessel_center_depth and vessel_depth_average differ by > 20
            group = group[np.abs(group['vessel_center_depth'] - group['vessel_depth_average']) <= 20]

            # Rule 2c: Retain the row where the bbox and mask height difference is minimized
            if not group.empty:
                group['height_diff'] = np.abs(group['vessel_bbox_height_px'] - group['vessel_mask_height_px'])
                best_row = group.loc[group['height_diff'].idxmin()]
                deduplicated_rows.append(best_row)
        else:
            # If only 2 rows in the group, directly apply Rule 2c
            group['height_diff'] = np.abs(group['vessel_bbox_height_px'] - group['vessel_mask_height_px'])
            best_row = group.loc[group['height_diff'].idxmin()]
            deduplicated_rows.append(best_row)

    # Create a deduplicated DataFrame
    deduplicated_df = pd.DataFrame(deduplicated_rows)

    # Drop temporary columns
    deduplicated_df = deduplicated_df.drop(columns=['timestamp_seconds', 'timestamp_group', 'height_diff'], errors='ignore')

    # Overwrite the original logs file
    deduplicated_df.to_csv(logs_file, index=False)
    print(f"Deduplicated logs saved to: {logs_file}")

    # File Cleanup: Delete unnecessary files
    cleanup_files(video_filename, deduplicated_df)

def cleanup_files(video_filename, deduplicated_df):
    # Define directories
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    base_output_dir = os.path.join('outputs', base_name)

    # Get IDs and frames to retain
    retained_files = set(deduplicated_df['image_filename'])

    # Extract unique IDs and frames from filenames
    retained_patterns = {f"id_{f.split('_id_')[1].split('_frame_')[0]}_frame_{f.split('_frame_')[1].split('.')[0]}" for f in retained_files}
    print(retained_patterns)
    # Define subdirectories for cleanup
    subdirs = [
        "combined",
        "depth_estimation/depth_arrays",
        "depth_estimation/depth_heatmaps",
        "object_detection/cropped_vessel_imgs",
        "object_detection/mask_arrays",
        "object_detection/mask_viz",
        "tracked_vessels"
    ]

    # Traverse each subdirectory and delete files not matching retained patterns
    for subdir in subdirs:
        full_path = os.path.join(base_output_dir, subdir)
        if os.path.exists(full_path):
            for file in os.listdir(full_path):
                file_path = os.path.join(full_path, file)
                if not any(pattern in file for pattern in retained_patterns):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Deduplicate logs for detected vessels and clean up unnecessary files.")
    parser.add_argument("video_filename", type=str, help="Path to the video file used to generate the logs.")
    args = parser.parse_args()

    # Run the deduplication and cleanup process
    deduplicate_logs(args.video_filename)
