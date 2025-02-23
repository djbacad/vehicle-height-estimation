import os
import numpy as np
import cv2
import csv
from scipy.ndimage import center_of_mass

# Constants for the orange area
ORANGE_AREA_DEPTH = 230
ORANGE_AREA_HEIGHT_PX = 72
ORANGE_AREA_HEIGHT_ACTUAL = 0.36

# FPS of the video
FPS = 30

def calculate_timestamp(frame_number, fps):
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:05.2f}"

def process_vessel(video_filename):
    # Paths
    base_name = os.path.splitext(video_filename)[0]
    tracked_vessels_dir = os.path.join('outputs', base_name, 'tracked_vessels')
    mask_arrays_dir = os.path.join('outputs', base_name, 'object_detection', 'mask_arrays')
    cropped_vessel_imgs_dir = os.path.join('outputs', base_name, 'object_detection', 'cropped_vessel_imgs')

    depth_arrays_dir = os.path.join('outputs', base_name, 'depth_estimation', 'depth_arrays')
    combined_output_dir = os.path.join('outputs', base_name, 'combined')
    logs_output_dir = os.path.join('outputs', base_name, '_logs')

    os.makedirs(combined_output_dir, exist_ok=True)
    os.makedirs(logs_output_dir, exist_ok=True)

    # CSV setup
    csv_file = os.path.join(logs_output_dir, f'{base_name}_logs.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'video_filename', 'image_filename', 'timestamp', 'vessel_center',
            'vessel_bbox_height_px', 'vessel_bbox_width_px', 'vessel_mask_height_px',
            'vessel_mask_width_px', 'vessel_center_depth', 'vessel_depth_average', 'vessel_height_actual'
        ])

        # Process each image in tracked_vessels
        for image_file in os.listdir(tracked_vessels_dir):
            if not image_file.endswith('.png'):
                continue

            try:
                # Extract ID and frame from the image filename
                image_name_parts = os.path.splitext(image_file)[0].split('_')
                id_part = image_name_parts[-3]
                frame_part = image_name_parts[-1]

                # Convert frame_part to an integer for timestamp calculation
                frame_number = int(frame_part)

                # Calculate the timestamp
                timestamp = calculate_timestamp(frame_number, FPS)

                # Paths to mask, depth arrays, and cropped vessel image
                mask_file = os.path.join(mask_arrays_dir, f"{base_name}_mask_array_id_{id_part}_frame_{frame_part}.npy")
                depth_file = os.path.join(depth_arrays_dir, f"{base_name}_depth_array_id_{id_part}_frame_{frame_part}.npy")
                cropped_image_file = os.path.join(cropped_vessel_imgs_dir, f"{base_name}_cropped_vessel_id_{id_part}_frame_{frame_part}.png")

                if not os.path.exists(mask_file) or not os.path.exists(depth_file) or not os.path.exists(cropped_image_file):
                    print(f"Missing mask, depth array, or cropped image for {image_file}. Skipping.")
                    continue

                # Load arrays and image
                vessel_mask = np.load(mask_file)
                depth = np.load(depth_file)
                img = cv2.imread(os.path.join(tracked_vessels_dir, image_file))
                cropped_img = cv2.imread(cropped_image_file)

                if vessel_mask.size == 0 or depth.size == 0:
                    print(f"Empty mask or depth array for {image_file}. Skipping.")
                    continue

                # Calculate vessel height in pixels
                rows, _ = np.where(vessel_mask > 0)
                if rows.size == 0:
                    print(f"No valid mask found in {image_file}. Skipping.")
                    continue
                vessel_mask_height_px = np.max(rows) - np.min(rows)


                # Calculate the vessel width in pixels
                _, cols = np.where(vessel_mask > 0)
                if cols.size == 0:  
                    print(f"No valid mask found in {image_file}. Skipping.")
                    continue
                vessel_mask_width_px = np.max(cols) - np.min(cols)

                # Calculate bounding box dimensions
                vessel_bbox_height_px = cropped_img.shape[0]  # Height of the cropped vessel image
                vessel_bbox_width_px = cropped_img.shape[1]   # Width of the cropped vessel image

                # Find center of the boat
                center_y, center_x = center_of_mass(vessel_mask)

                # Normalize depth map
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # Create overlays
                mask_overlay = cv2.addWeighted(depth_colored, 0.4, cv2.cvtColor((vessel_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.6, 0)
                image_overlay = cv2.addWeighted(img, 0.6, depth_colored, 0.4, 0)

                # Combine the heatmap, mask overlay, and image overlay
                combined_image = cv2.hconcat([depth_colored, mask_overlay, image_overlay])

                # Save combined heatmap
                combined_filename = f"{base_name}_combined_viz_id_{id_part}_frame_{frame_part}.png"
                combined_filepath = os.path.join(combined_output_dir, combined_filename)
                cv2.imwrite(combined_filepath, combined_image)

                # Get depth values within the mask
                vessel_mask_boolean = vessel_mask > 0
                rows_with_mask = np.any(vessel_mask_boolean, axis=1)
                vessel_depth_values = depth_normalized[rows_with_mask, :]
                vessel_depth_average = vessel_depth_values[vessel_mask_boolean[rows_with_mask, :]].mean()

                # Get depth at vessel center
                vessel_center_depth = depth_normalized[int(center_y), int(center_x)]

                # Calculate actual vessel height
                vessel_height_actual = vessel_mask_height_px * (ORANGE_AREA_HEIGHT_ACTUAL / ORANGE_AREA_HEIGHT_PX) * (ORANGE_AREA_DEPTH / vessel_center_depth)

                # Log results
                writer.writerow([
                    video_filename, image_file, timestamp, (center_x, center_y), vessel_bbox_height_px, 
                    vessel_bbox_width_px, vessel_mask_height_px, vessel_mask_width_px, vessel_center_depth, vessel_depth_average, vessel_height_actual
                ])

                print(f"Processed {image_file}: Combined heatmap saved at {combined_filepath}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}. Skipping.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process vessel height from tracked images and save outputs.")
    parser.add_argument("video_filename", type=str, help="Name of the video file located in 'data/videos'.")
    args = parser.parse_args()

    process_vessel(args.video_filename)
