import argparse
import os
import torch
import numpy as np
import cv2
import matplotlib
from third_party.depth_anything_v2.dpt import DepthAnythingV2

def main(video_filename):
    # Determine device
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available() 
        else 'cpu'
    )
    print(f"Using device: {DEVICE}")

    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Choose an encoder (change if needed)
    encoder = 'vitl'

    # Load the DepthAnythingV2 model and its weights
    model = DepthAnythingV2(**model_configs[encoder])
    model_path = f'models_dav2/depth_anything_v2_{encoder}.pth'
    # model_path = os.path.join('dav2_models', f'depth_anything_v2_{encoder}.pth')

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # Determine the absolute base output directory based on the script's location.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_output_dir = os.path.join(script_dir, "..", "outputs", video_filename)

    # Locate the tracked vehicles folder
    tracked_vehicles_dir = os.path.join(base_output_dir, 'tracked_vehicles')
    if not os.path.exists(tracked_vehicles_dir):
        raise FileNotFoundError(
            f"The directory '{tracked_vehicles_dir}' does not exist. Make sure the video has been processed."
        )

    # Iterate over all tracked vehicle images (accept .jpg and .png)
    for image_file in os.listdir(tracked_vehicles_dir):
        if not image_file.lower().endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(tracked_vehicles_dir, image_file)
        if not os.path.exists(image_path):
            print(f"File does not exist: {image_path}")
            continue

        raw_img = cv2.imread(image_path)
        if raw_img is None:
            print(f"Skipping file: {image_file} (could not be read as an image)")
            continue

        print(f"Processing file: {image_path}")

        # Perform depth inference
        with torch.no_grad():
            depth = model.infer_image(raw_img)  # Depth inferred as a NumPy array

        # Parse filename assuming format: vehicle_{vehicle_id}_frame_{frame_number}.jpg
        image_file_parts = image_file.split("_")
        # image_file_parts = ["vehicle", "{vehicle_id}", "frame", "{frame_number}.jpg"]
        vehicle_id = image_file_parts[1]
        frame_number = image_file_parts[3].split('.')[0]  # Remove file extension

        # Save the depth array as .npy
        depth_array_file_name = f"{video_filename}_depth_array_id_{vehicle_id}_frame_{frame_number}.npy"
        depth_array_output_path = os.path.join(
            base_output_dir, 'depth_estimation', 'depth_arrays', depth_array_file_name
        )
        os.makedirs(os.path.dirname(depth_array_output_path), exist_ok=True)
        np.save(depth_array_output_path, depth)
        print(f"Depth array saved to: {depth_array_output_path}")

        # Prepare the directory for heatmap visualizations
        depth_heatmap_output_dir = os.path.join(
            base_output_dir, 'depth_estimation', 'depth_heatmaps'
        )
        os.makedirs(depth_heatmap_output_dir, exist_ok=True)

        # Normalize the depth map to the range 0-255
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)

        # Apply a colormap for visualization
        use_grayscale = False  # Set to True for grayscale output
        if use_grayscale:
            depth_colored = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
        else:
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)

        combine_images = True
        if combine_images:
            split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255 
            combined_result = cv2.hconcat([raw_img, split_region, depth_colored])
            output_image = combined_result
        else:
            output_image = depth_colored

        # Save the heatmap visualization image
        heatmap_file_name = f"{video_filename}_heatmap_viz_id_{vehicle_id}_frame_{frame_number}.png"
        heatmap_output_path = os.path.join(depth_heatmap_output_dir, heatmap_file_name)
        cv2.imwrite(heatmap_output_path, output_image)
        print(f"Heatmap visualization saved to: {heatmap_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth estimation using DepthAnythingV2 for tracked vehicles."
    )
    parser.add_argument(
        "video_filename",
        type=str,
        help="Identifier of the video (used as the output folder name)."
    )
    args = parser.parse_args()
    main(args.video_filename)
