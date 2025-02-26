import os
import numpy as np
import cv2
import csv
from scipy.ndimage import center_of_mass

# Constants for the reference object
REFERENCE_OBJ_DEPTH = 46
REFERENCE_OBJ_HEIGHT_PX = 60
REFERENCE_OBJ_HEIGHT_ACTUAL = 0.65

# FPS of the video
FPS = 30

def calculate_timestamp(frame_number, fps):
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:05.2f}"

def process_vehicle(video_filename):
    base_name = os.path.splitext(video_filename)[0]
    tracked_vehicles_dir = os.path.join('outputs', base_name, 'tracked_vehicles')
    mask_arrays_dir = os.path.join('outputs', base_name, 'mask_arrays')
    cropped_vehicle_imgs_dir = os.path.join('outputs', base_name, 'cropped_vehicles')
    depth_arrays_dir = os.path.join('outputs', base_name, 'depth_estimation', 'depth_arrays')
    combined_output_dir = os.path.join('outputs', base_name, 'combined')
    logs_output_dir = os.path.join('outputs', base_name, '_logs')

    os.makedirs(combined_output_dir, exist_ok=True)
    os.makedirs(logs_output_dir, exist_ok=True)

    csv_file = os.path.join(logs_output_dir, f'{base_name}_logs.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'video_filename', 'image_filename', 'timestamp', 'vehicle_center',
            'vehicle_bbox_height_px', 'vehicle_bbox_width_px', 'vehicle_mask_height_px',
            'vehicle_mask_width_px', 'vehicle_center_depth', 'vehicle_depth_average', 'vehicle_height_actual'
        ])

        for image_file in os.listdir(tracked_vehicles_dir):
            if not image_file.lower().endswith(('.jpg', '.png')):
                continue

            try:
                image_name_parts = os.path.splitext(image_file)[0].split('_')
                if len(image_name_parts) < 4:
                    print(f"Unexpected filename format: {image_file}. Skipping.")
                    continue
                id_part = image_name_parts[1]
                frame_part = image_name_parts[3]
                frame_num = int(frame_part) 

                timestamp = calculate_timestamp(frame_num, FPS)

                mask_file = os.path.join(mask_arrays_dir, f"vehicle_{id_part}_frame_{frame_part}_mask.npy")
                depth_file = os.path.join(depth_arrays_dir, f"{base_name}_depth_array_id_{id_part}_frame_{frame_part}.npy")
                cropped_image_file = os.path.join(cropped_vehicle_imgs_dir, f"vehicle_{id_part}_frame_{frame_part}_cropped.png")

                if not (os.path.exists(mask_file) and os.path.exists(depth_file) and os.path.exists(cropped_image_file)):
                    print(f"Missing mask, depth array, or cropped image for {image_file}. Skipping.")
                    continue

                # Load arrays and images
                vehicle_mask = np.load(mask_file)
                depth = np.load(depth_file)
                img_path = os.path.join(tracked_vehicles_dir, image_file)
                img = cv2.imread(img_path)
                cropped_img = cv2.imread(cropped_image_file)

                if vehicle_mask.size == 0 or depth.size == 0:
                    print(f"Empty mask or depth array for {image_file}. Skipping.")
                    continue

                rows = np.where(vehicle_mask > 0)[0]
                if rows.size == 0:
                    print(f"No valid mask found in {image_file}. Skipping.")
                    continue
                vehicle_mask_height_px = np.max(rows) - np.min(rows)

                cols = np.where(vehicle_mask > 0)[0]
                if cols.size == 0:
                    print(f"No valid mask found in {image_file}. Skipping.")
                    continue
                vehicle_mask_width_px = np.max(cols) - np.min(cols)

                vehicle_bbox_height_px = cropped_img.shape[0]
                vehicle_bbox_width_px = cropped_img.shape[1]

                # Center of mass for the mask
                center_y, center_x = center_of_mass(vehicle_mask)

                # Normalize depth map
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                depth_normalized = depth_normalized.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                # Middle image overlay (depth + mask)
                mask_overlay = cv2.addWeighted(
                    depth_colored, 0.4,
                    cv2.cvtColor((vehicle_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                    0.6, 0
                )

                # ===== Overlay the height in pixels on the middle image =====
                height_text_px = f"Height(px): {vehicle_mask_height_px}"

                text_x_mid = int(center_x)
                text_y_mid = int(center_y)
                text_y_mid = max(20, min(text_y_mid, mask_overlay.shape[0] - 10))
                text_x_mid = max(0, min(text_x_mid, mask_overlay.shape[1] - 200))

                cv2.putText(
                    mask_overlay,
                    height_text_px,
                    (text_x_mid, text_y_mid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,               
                    (255, 255, 255),    
                    2                  
                )

                image_overlay = cv2.addWeighted(img, 0.6, depth_colored, 0.4, 0)

                vehicle_depth_values = depth_normalized[(vehicle_mask > 0)]
                vehicle_center_depth = depth_normalized[int(center_y), int(center_x)]
                vehicle_height_actual = (
                    vehicle_mask_height_px *
                    (REFERENCE_OBJ_HEIGHT_ACTUAL / REFERENCE_OBJ_HEIGHT_PX) *
                    (REFERENCE_OBJ_DEPTH / vehicle_center_depth)
                )

                height_text_m = f"Height(m): {vehicle_height_actual:.2f}"

                text_x_right = int(center_x)
                text_y_right = int(center_y)
                text_y_right = max(20, min(text_y_right, image_overlay.shape[0] - 10))
                text_x_right = max(0, min(text_x_right, image_overlay.shape[1] - 300))

                cv2.putText(
                    image_overlay,
                    height_text_m,
                    (text_x_right, text_y_right),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,                  
                    (255, 255, 255),      
                    2                     
                )

                # Combine all three horizontally
                combined_image = cv2.hconcat([depth_colored, mask_overlay, image_overlay])
                combined_filename = f"{base_name}_combined_viz_id_{id_part}_frame_{frame_part}.png"
                combined_filepath = os.path.join(combined_output_dir, combined_filename)
                cv2.imwrite(combined_filepath, combined_image)

                # Compute average depth within the mask for logging
                rows_with_mask = np.any(vehicle_mask > 0, axis=1)
                vehicle_depth_average = vehicle_depth_values.mean()

                # Log results
                writer.writerow([
                    video_filename, image_file, timestamp, (center_x, center_y),
                    vehicle_bbox_height_px, vehicle_bbox_width_px, vehicle_mask_height_px,
                    vehicle_mask_width_px, vehicle_center_depth, vehicle_depth_average, vehicle_height_actual
                ])

                print(f"Processed {image_file}: Combined heatmap saved at {combined_filepath}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}. Skipping.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process vehicle height from tracked images and save outputs.")
    parser.add_argument("video_filename", type=str, help="Identifier of the video (e.g., K6xsEng2PhU).")
    args = parser.parse_args()

    process_vehicle(args.video_filename)
