import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Utility Functions

def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def save_segmented_vehicle(frame, box, vehicle_id, frame_number, base_output_dir, segmentation_model):
    """
    Segments the specific vehicle in the frame.
    Saves both the visualization (mask overlay) and the raw mask array.
    """
    seg_viz_dir = os.path.join(base_output_dir, "mask_viz")
    seg_array_dir = os.path.join(base_output_dir, "mask_arrays")
    os.makedirs(seg_viz_dir, exist_ok=True)
    os.makedirs(seg_array_dir, exist_ok=True)

    base_name = f"vehicle_{vehicle_id}_frame_{frame_number}"
    output_file = os.path.join(seg_viz_dir, f"{base_name}_mask.png")
    mask_array_output_file = os.path.join(seg_array_dir, f"{base_name}_mask.npy")

    # If segmentation already exists, skip
    if os.path.exists(output_file):
        print(f"Segmented mask already exists: {output_file}")
        return True

    # Run segmentation on the full image
    results_seg = segmentation_model(frame, imgsz=1920, conf=0.2, retina_masks=True)
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_frame = frame.copy()

    x1, y1, x2, y2 = map(int, box)
    print(f"Detection box: {x1}, {y1}, {x2}, {y2}")

    for r in results_seg:
        for mask, seg_box in zip(r.masks.xy, r.boxes.xyxy):
            sx1, sy1, sx2, sy2 = map(int, seg_box)
            iou_val = compute_iou((x1, y1, x2, y2), (sx1, sy1, sx2, sy2))
            print(f"IoU: {iou_val}")
            if iou_val > 0.30:
                mask_np = np.array(mask, dtype=np.int32)
                cv2.fillPoly(full_mask, [mask_np], 1)
                cv2.polylines(mask_frame, [mask_np], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.fillPoly(mask_frame, [mask_np], color=(0, 255, 0))

    # Save the mask array and visualization
    np.save(mask_array_output_file, full_mask)
    print(f"Full-size mask array saved: {mask_array_output_file}")
    cv2.imwrite(output_file, mask_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Segmented mask saved: {output_file}")
    return False

def save_cropped_vehicle(frame, box, vehicle_id, frame_number, base_output_dir):
    """Crops and saves the detected vehicle image."""
    cropped_dir = os.path.join(base_output_dir, "cropped_vehicles")
    os.makedirs(cropped_dir, exist_ok=True)

    base_name = f"vehicle_{vehicle_id}_frame_{frame_number}"
    output_file = os.path.join(cropped_dir, f"{base_name}_cropped.png")
    x1, y1, x2, y2 = map(int, box)
    cropped = frame[y1:y2, x1:x2]
    cv2.imwrite(output_file, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Cropped vehicle image saved: {output_file}")

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Track and segment vehicles from a YouTube video."
    )
    parser.add_argument("yt_suffix", type=str, help="YouTube video suffix (the part after 'watch?v=').")
    parser.add_argument("--line_x", type=int, default=500,
                        help="X-position of the vertical line to detect crossing (default: 500).")
    parser.add_argument("--frame_tolerance", type=int, default=20,
                        help="Frame tolerance to prevent duplicate captures (default: 20).")
    return parser.parse_args()

# Main
def main():
    args = parse_args()
    yt_suffix = args.yt_suffix
    line_x = args.line_x
    FRAME_TOLERANCE = args.frame_tolerance

    # Create the base output directory if it doesn't exist
    base_output_dir = f"outputs/{yt_suffix}"
    tracked_output_dir = os.path.join(base_output_dir, "tracked_vehicles")
    os.makedirs(tracked_output_dir, exist_ok=True)

    # Load YOLO models (detection and segmentation)
    detection_model = YOLO("models_yolo/yolo12x.pt")
    segmentation_model = YOLO("models_yolo/yolo11x-seg.pt")

    # Run inference in stream mode from YouTube
    results = detection_model.track(
        source=f"https://www.youtube.com/watch?v={yt_suffix}",
        stream_buffer=True,
        stream=True,
        persist=True,
        vid_stride=1,
        classes=[2]  # class 2: cars
    )

    # Dictionary to store the last saved frame for each vehicle (tracking ID)
    last_saved_frame = {}
    frame_number = 0  # Global frame counter

    # Process each frame from the stream
    for result in results:
        frame_number += 1

        # Get the annotated frame (for visualization only)
        annotated_frame = result.plot()
        frame_height, frame_width = annotated_frame.shape[:2]

        # Draw the vertical line for visualization
        cv2.line(annotated_frame, (line_x, 0), (line_x, frame_height), (0, 255, 0), 2)

        save_frame = False
        updated_ids = []
        detection_boxes = {}

        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            for i, box in enumerate(result.boxes.xyxy):
                # Extract bounding box coordinates
                if hasattr(box, "cpu"):
                    coords = [int(coord) for coord in box.cpu().numpy()]
                else:
                    coords = [int(coord) for coord in box]
                x1, y1, x2, y2 = coords
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw center (on annotated frame for visualization)
                cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)

                # Retrieve the tracking ID
                try:
                    track_id = int(result.boxes.id[i])
                except (TypeError, AttributeError):
                    track_id = i

                cv2.putText(annotated_frame, f"ID:{track_id}", (center_x - 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Check if the vehicle's center is within tolerance of the vertical line
                if line_x - FRAME_TOLERANCE <= center_x <= line_x + FRAME_TOLERANCE:
                    # Skip if recently captured
                    if track_id in last_saved_frame and (frame_number - last_saved_frame[track_id]) <= FRAME_TOLERANCE:
                        print(f"Skipping duplicate frame for vehicle ID {track_id} at frame {frame_number}")
                        continue
                    save_frame = True
                    updated_ids.append(track_id)
                    detection_boxes[track_id] = coords

        if save_frame:
            if hasattr(result, 'orig_img'):
                pure_frame = result.orig_img
            elif hasattr(result, 'imgs'):
                pure_frame = result.imgs[0]
            else:
                pure_frame = annotated_frame.copy()

            for tid in updated_ids:
                # Save the pure (unannotated) frame
                frame_filename = os.path.join(
                    tracked_output_dir,
                    f"vehicle_{tid}_frame_{frame_number}.jpg"
                )
                cv2.imwrite(frame_filename, pure_frame)
                print(f"Tracked frame saved: {frame_filename}")

                last_saved_frame[tid] = frame_number

                # Use the detection bounding box for segmentation and cropping
                bbox = detection_boxes[tid]
                _ = save_segmented_vehicle(
                    frame=pure_frame,
                    box=bbox,
                    vehicle_id=tid,
                    frame_number=frame_number,
                    base_output_dir=base_output_dir,
                    segmentation_model=segmentation_model
                )
                save_cropped_vehicle(
                    frame=pure_frame,
                    box=bbox,
                    vehicle_id=tid,
                    frame_number=frame_number,
                    base_output_dir=base_output_dir
                )

        # Display the annotated frame (for visualization purposes)
        cv2.imshow("Vehicle Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
