import argparse
import os
import torch
import numpy as np
import cv2
import matplotlib
from third_party.depth_anything_v2.dpt import DepthAnythingV2

def main(video_filename):
    # Determine device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Encoder choice
    encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

    # Load the model
    model = DepthAnythingV2(**model_configs[encoder])
    #model_path = f'src/third_party/checkpoints/depth_anything_v2_{encoder}.pth'
    model_path = f'../dav2_weights/depth_anything_v2_{encoder}.pth'

    # Load model weights to the correct device
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # Locate tracked vessels folder
    base_name = os.path.splitext(video_filename)[0]
    tracked_vessels_dir = os.path.join('outputs', base_name, 'tracked_vessels')

    if not os.path.exists(tracked_vessels_dir):
        raise FileNotFoundError(f"The directory '{tracked_vessels_dir}' does not exist. Make sure the video has been processed.")

    # Iterate over all images in the tracked vessels directory
    for image_file in os.listdir(tracked_vessels_dir):
        if not image_file.endswith(".png"):
            continue

        image_path = os.path.join(tracked_vessels_dir, image_file)
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
            depth = model.infer_image(raw_img)  # Pass raw image (NumPy array)

        # Parse the image file name for correct naming convention
        base_name_parts = base_name.split("_")
        image_file_parts = image_file.split("_")
        vessel_id = image_file_parts[-3]  # Extracting id_x
        frame_number = image_file_parts[-1].replace(".png", "")

        # Save the depth array as .npy
        depth_array_file_name = f"{base_name}_depth_array_id_{vessel_id}_frame_{frame_number}.npy"
        depth_array_output_path = os.path.join('outputs', base_name, 'depth_estimation', 'depth_arrays', depth_array_file_name)
        os.makedirs(os.path.dirname(depth_array_output_path), exist_ok=True)
        np.save(depth_array_output_path, depth)
        print(f"Depth array saved to: {depth_array_output_path}")

        # Additional processing for heatmap visualization
        depth_heatmap_output_dir = os.path.join('outputs', base_name, 'depth_estimation', 'depth_heatmaps')
        os.makedirs(depth_heatmap_output_dir, exist_ok=True)

        # Normalize the depth map to 0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Apply colormap (change to 'grayscale' if needed)
        use_grayscale = False  # Set to True for grayscale output
        if use_grayscale:
            depth_colored = np.repeat(depth[..., np.newaxis], 3, axis=-1)  # Grayscale
        else:
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')  # Use 'Spectral_r' colormap
            depth_colored = (cmap(depth / 255.0)[:, :, :3] * 255).astype(np.uint8)  # RGB to BGR for OpenCV

        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_PLASMA)

        # Load the raw image (for combining)
        raw_image = cv2.imread(image_path)

        # Combine raw image with the depth map (optional)
        combine_images = True  # Set to False if you only want the heatmap
        if combine_images:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255  # White space
            combined_result = cv2.hconcat([raw_image, split_region, depth_colored])
            output_image = combined_result
        else:
            output_image = depth_colored

        # Save the result
        heatmap_file_name = f"{base_name}_heatmap_viz_id_{vessel_id}_frame_{frame_number}.png"
        heatmap_output_path = os.path.join(depth_heatmap_output_dir, heatmap_file_name)
        cv2.imwrite(heatmap_output_path, output_image)
        print(f"Heatmap visualization saved to: {heatmap_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth estimation script using DepthAnythingV2.")
    parser.add_argument("video_filename", type=str, help="Name of the video file located in the 'data/videos' folder.")
    args = parser.parse_args()

    main(args.video_filename)
