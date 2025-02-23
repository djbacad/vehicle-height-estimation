import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO models
detection_model = YOLO("./yolo_weights/yolo11x.pt")
segmentation_model = YOLO("./yolo_weights/yolo11x-seg.pt")

# Frame tolerance for duplicate detection
FRAME_TOLERANCE = 12  # Adjust this as needed

# Dictionary to track last saved frame per vessel ID
last_saved_frame = {}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Track and segment vessels in a video.")
    parser.add_argument('video_filename', help="Name of the video file located in the input directory.")
    parser.add_argument('--enable_visualization', action='store_true', help="If set, display the video with annotations.")
    return parser.parse_args()

def setup_output_directories(video_filename):
    base_name = os.path.splitext(video_filename)[0]
    base_output_dir = os.path.join('outputs', base_name)
    tracked_vessels_dir = os.path.join(base_output_dir, 'tracked_vessels')
    object_detection_dir = os.path.join(base_output_dir, 'object_detection')
    mask_viz_dir = os.path.join(object_detection_dir, 'mask_viz')
    mask_arrays_dir = os.path.join(object_detection_dir, 'mask_arrays')
    cropped_vessel_img_dir = os.path.join(object_detection_dir, 'cropped_vessel_imgs')

    os.makedirs(tracked_vessels_dir, exist_ok=True)
    os.makedirs(mask_viz_dir, exist_ok=True)
    os.makedirs(mask_arrays_dir, exist_ok=True)
    os.makedirs(cropped_vessel_img_dir, exist_ok=True)

    return {
        "base_output_dir": base_output_dir,
        "tracked_vessels_dir": tracked_vessels_dir,
        "mask_viz_dir": mask_viz_dir,
        "mask_arrays_dir": mask_arrays_dir,
        "cropped_vessel_img_dir": cropped_vessel_img_dir
    }

def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Calculate intersection
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    # Calculate union
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou

def save_segmented_vessel(frame, box, vessel_id, args, frame_count, output_dirs):
    """Segments the specific vessel ID and overlays it on the full image."""
    # Generate a unique filename for the segmented mask
    base_name = os.path.splitext(args.video_filename)[0]
    output_file = os.path.join(output_dirs['mask_viz_dir'], f"{base_name}_mask_viz_id_{vessel_id}_frame_{frame_count}.png")
    mask_array_output_file = os.path.join(output_dirs['mask_arrays_dir'], f"{base_name}_mask_array_id_{vessel_id}_frame_{frame_count}.npy")

    # Check if the file already exists, if yes, skip processing
    if os.path.exists(output_file):
        print(f"Segmented mask already exists: {output_file}")
        return True  # Signal to break out of the loop

    # Run segmentation on the full image
    results = segmentation_model(frame, imgsz=1920, conf=0.2, retina_masks=True) # 0.2 is enough since we just wanna find the overlap of the segmentation model and detection model

    # Create a blank mask for the full frame
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Create a copy of the frame for the mask overlay
    mask_frame = frame.copy()

    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, box)
    print(f"Detection box: {x1, y1, x2, y2}")  # Debugging detection box

    # Iterate through segmentation results to find masks matching vessel ID
    for r in results:
        for mask, seg_box in zip(r.masks.xy, r.boxes.xyxy):
            # Convert segmentation box coordinates
            sx1, sy1, sx2, sy2 = map(int, seg_box)
            print(f"Segmentation box: {sx1, sy1, sx2, sy2}")  # Debugging segmentation box
            
            # Check overlap using IoU
            computed_iou = compute_iou((x1, y1, x2, y2), (sx1, sy1, sx2, sy2))
            print(computed_iou)
            if computed_iou > 0.30:  # IoU > 30%
                # Draw the mask (scaled to full image)
                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(full_mask, [mask], 1)  # Fill binary mask with 1 for detected region
                cv2.polylines(mask_frame, [mask], isClosed=True, color=(0, 255, 0), thickness=2)  # Green outline
                cv2.fillPoly(mask_frame, [mask], color=(0, 255, 0, 50))  # Semi-transparent fill

    # Save the full-size mask array
    np.save(mask_array_output_file, full_mask)
    print(f"Full-size mask array saved: {mask_array_output_file}")

    # Save the segmented image with the mask applied (high quality, no compression)
    cv2.imwrite(output_file, mask_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Segmented mask saved: {output_file}")
    return False

def save_cropped_vessel(frame, box, vessel_id, frame_count, output_dirs, args):
    """Saves the cropped region of the detected vessel."""
    base_name = os.path.splitext(args.video_filename)[0]
    cropped_vessel_img_dir = os.path.join(output_dirs['cropped_vessel_img_dir'])
    os.makedirs(cropped_vessel_img_dir, exist_ok=True)

    # Generate the output file path
    output_file = os.path.join(
        cropped_vessel_img_dir,
        f"{base_name}_cropped_vessel_id_{vessel_id}_frame_{frame_count}.png"
    )

    # Extract bounding box coordinates and crop
    x1, y1, x2, y2 = map(int, box)
    cropped_vessel = frame[y1:y2, x1:x2]

    # Save the cropped image
    cv2.imwrite(output_file, cropped_vessel, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Cropped vessel image saved: {output_file}")

def main():
    args = parse_arguments()

    # Setup output directories
    output_dirs = setup_output_directories(args.video_filename)

    # Paths
    input_video_path = os.path.join('data/videos/', args.video_filename)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Line positions
    black_line_x = 1090
    blue_line_x = black_line_x + 220
    tolerance = 5

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Perform object detection and tracking
        results = detection_model.track(frame, classes=[8], persist=True, conf=0.35)  # Detect boats (class 8)
        annotated_frame = results[0].plot()

        # Iterate over detected objects
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract class ID and confidence
                class_id = int(box.cls)
                confidence = box.conf

                if class_id == 8 and confidence > 0.5:  # Boat class detected
                    # Get bounding box and vessel ID
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = map(int, xyxy)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Draw the center dot
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow dot

                    vessel_id = int(box.id.cpu().numpy()) if box.id is not None else 0

                    # Check if the boat crosses the blue line
                    if blue_line_x - tolerance <= center_x <= blue_line_x + tolerance:
                        if vessel_id in last_saved_frame and abs(frame_count - last_saved_frame[vessel_id]) <= FRAME_TOLERANCE:
                            print(f"Skipping duplicate frame for vessel ID {vessel_id} at frame {frame_count}")
                            continue

                        base_name = os.path.splitext(args.video_filename)[0]
                        frame_filename = os.path.join(output_dirs['tracked_vessels_dir'], f"{base_name}_vessel_detected_shot_id_{vessel_id}_frame_{frame_count}.png")
                        cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        last_saved_frame[vessel_id] = frame_count
                        print(f"Frame saved: {frame_filename}")

                        # Perform segmentation for this specific vessel ID
                        if save_segmented_vessel(frame, xyxy, vessel_id, args, frame_count, output_dirs):
                            break  # Exit loop if file already exists

                        # Save cropped vessel image
                        save_cropped_vessel(frame, xyxy, vessel_id, frame_count, output_dirs, args)

        # Draw vertical lines
        cv2.line(annotated_frame, (black_line_x, 0), (black_line_x, frame.shape[0]), (0, 0, 0), 10)
        cv2.line(annotated_frame, (blue_line_x, 0), (blue_line_x, frame.shape[0]), (255, 0, 0), 10)

        # Display visualization
        if args.enable_visualization:
            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources
    cap.release()
    if args.enable_visualization:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
