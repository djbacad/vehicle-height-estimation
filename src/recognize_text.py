import os
import csv
import cv2
import keras_ocr
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

def recognize_text_from_images(video_filename):
    # Initialize the OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Define paths
    base_name = os.path.splitext(video_filename)[0]
    cropped_vessel_imgs_dir = os.path.join('outputs', base_name, 'object_detection', 'cropped_vessel_imgs')
    logs_dir = os.path.join('outputs', base_name, '_logs')
    csv_file_path = os.path.join(logs_dir, f'{base_name}_logs.csv')

    # Check if directories exist
    if not os.path.exists(cropped_vessel_imgs_dir):
        raise FileNotFoundError(f"Cropped vessel images folder not found: {cropped_vessel_imgs_dir}")

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV log file not found: {csv_file_path}")

    # Perform OCR for each image in the cropped_vessel_imgs folder
    ocr_results = {}
    for image_file in os.listdir(cropped_vessel_imgs_dir):
        image_path = os.path.join(cropped_vessel_imgs_dir, image_file)
        if not image_file.endswith('.png'):
            continue

        # Read and process the image
        image = cv2.imread(image_path)
        print(f"Performing OCR for image: {image_path}")
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Check and adjust image dimensions
        height, width, _ = image.shape
        if height % 6 != 0 or width % 6 != 0:
            new_height = (height // 6) * 6
            new_width = (width // 6) * 6
            print(f"Resizing image from ({height}, {width}) to ({new_height}, {new_width})")
            image = cv2.resize(image, (new_width, new_height))

        try:
            # Perform OCR
            detected_chars = []
            results = pipeline.recognize([image])
            for text, box in results[0]:
                detected_chars.append(text)
            print(f"Detected characters: {detected_chars}")

            # Normalize the filename for matching
            normalized_filename = image_file.replace('cropped_vessel_', 'vessel_detected_shot_')
            ocr_results[normalized_filename] = detected_chars
            print(f"Normalized filename: {normalized_filename}, OCR Results: {ocr_results[normalized_filename]}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}. Skipping this image.")

    # Update the CSV file with OCR results
    updated_rows = []
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames

        # Add 'ocr_results' column if it doesn't exist
        if 'ocr_results' not in fieldnames:
            fieldnames.append('ocr_results')

        for row in reader:
            image_filename = row['image_filename'].strip()  # Ensure no leading/trailing spaces
            print(f"CSV image_filename: {image_filename}")  # Debugging: Print the image filename from CSV

            if image_filename in ocr_results:
                row['ocr_results'] = ', '.join(ocr_results[image_filename])  # Append OCR results
                print(f"Match found! Adding OCR results: {ocr_results[image_filename]}")  # Debugging
            else:
                row['ocr_results'] = ''  # Empty if no results
                print(f"No match found for {image_filename} in OCR results.")  # Debugging

            updated_rows.append(row)

    # Write updated CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Updated CSV file saved at: {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform OCR on cropped vessel images and update the log CSV file.")
    parser.add_argument('video_filename', type=str, help="Name of the video file (e.g., 2024_1126_150000F_trimmed.MP4).")
    args = parser.parse_args()

    recognize_text_from_images(args.video_filename)
