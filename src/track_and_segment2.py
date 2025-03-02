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
    return inter_area / union_area if union_area != 0 else 0

def segment_vehicle(frame, box, track_id, frame_no, base_out, seg_model):
    """
    Segments the specific vehicle in the frame.
    Saves both the visualization (mask overlay) and the raw mask array.
    """
    seg_viz_dir = os.path.join(base_out, "mask_viz")
    seg_array_dir = os.path.join(base_out, "mask_arrays")
    os.makedirs(seg_viz_dir, exist_ok=True)
    os.makedirs(seg_array_dir, exist_ok=True)

    base_name = f"vehicle_{track_id}_frame_{frame_no}"
    out_viz = os.path.join(seg_viz_dir, f"{base_name}_mask.png")
    out_mask = os.path.join(seg_array_dir, f"{base_name}_mask.npy")

    # If segmentation already exists, skip processing.
    if os.path.exists(out_viz):
        print(f"Segmented mask already exists: {out_viz}")
        return True

    # Run segmentation on the full image.
    results_seg = seg_model(frame, imgsz=1920, conf=0.2, retina_masks=True)
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_frame = frame.copy()

    x1, y1, x2, y2 = map(int, box)
    print(f"Detection box: {x1}, {y1}, {x2}, {y2}")

    # Iterate through segmentation results to find matching masks.
    for r in results_seg:
        for mask, seg_box in zip(r.masks.xy, r.boxes.xyxy):
            sx1, sy1, sx2, sy2 = map(int, seg_box)
            print(f"Segmentation box: {sx1}, {sy1}, {sx2}, {sy2}")
            iou_val = compute_iou((x1, y1, x2, y2), (sx1, sy1, sx2, sy2))
            print(f"IoU: {iou_val}")
            if iou_val > 0.30:
                mask_np = np.array(mask, dtype=np.int32)
                cv2.fillPoly(full_mask, [mask_np], 1)
                cv2.polylines(mask_frame, [mask_np], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.fillPoly(mask_frame, [mask_np], color=(0, 255, 0))
    
    # Save the mask array and visualization.
    np.save(out_mask, full_mask)
    print(f"Full-size mask array saved: {out_mask}")
    cv2.imwrite(out_viz, mask_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Segmented mask saved: {out_viz}")
    return False

def crop_vehicle(frame, box, track_id, frame_no, base_out):
    """
    Crops and saves the detected vehicle image.
    """
    cropped_dir = os.path.join(base_out, "cropped_vehicles")
    os.makedirs(cropped_dir, exist_ok=True)

    base_name = f"vehicle_{track_id}_frame_{frame_no}"
    out_crop = os.path.join(cropped_dir, f"{base_name}_cropped.png")

    x1, y1, x2, y2 = map(int, box)
    cropped = frame[y1:y2, x1:x2]
    cv2.imwrite(out_crop, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Cropped vehicle image saved: {out_crop}")

def process_video(yt_suffix, line_x=500, frame_tolerance=20, stream=False):
    """
    Streams frames from https://www.youtube.com/watch?v=<yt_suffix>.
    For each frame:
      - Runs detection on vehicles (class=2).
      - Draws a vertical line at `line_x`.
      - If a vehicle center crosses the line within `frame_tolerance`, saves the unannotated frame,
        performs segmentation and cropping.
    If stream=True, yields each annotated frame (as JPEG bytes) for real-time visualization.
    """
    base_out = os.path.join("outputs", yt_suffix)
    tracked_dir = os.path.join(base_out, "tracked_vehicles")
    os.makedirs(tracked_dir, exist_ok=True)

    detection_model = YOLO("models_yolo/yolo12x.pt")
    segmentation_model = YOLO("models_yolo/yolo11x-seg.pt")

    results = detection_model.track(
        source=f"https://www.youtube.com/watch?v={yt_suffix}",
        stream_buffer=True,
        stream=True,
        persist=True,
        vid_stride=1,
        classes=[2]  # detect cars
    )

    last_saved_frame = {}
    frame_no = 0

    while True:
        try:
            result = next(results)
        except StopIteration:
            break

        frame_no += 1
        annotated_frame = result.plot()
        cv2.line(annotated_frame, (line_x, 0), (line_x, annotated_frame.shape[0]), (0, 255, 0), 2)

        updated_ids = {}
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            for i, box in enumerate(result.boxes.xyxy):
                coords = [int(c) for c in box.cpu().numpy()] if hasattr(box, "cpu") else [int(c) for c in box]
                x1, y1, x2, y2 = coords
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw center and ID on annotated frame
                cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), -1)
                try:
                    track_id = int(result.boxes.id[i])
                except (TypeError, AttributeError):
                    track_id = i
                cv2.putText(annotated_frame, f"ID:{track_id}", (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Check if vehicle center is near vertical line
                if abs(cx - line_x) <= frame_tolerance:
                    if track_id in last_saved_frame and (frame_no - last_saved_frame[track_id]) <= frame_tolerance:
                        print(f"Skipping duplicate for ID {track_id} at frame {frame_no}")
                        continue
                    updated_ids[track_id] = coords

        if updated_ids:
            # Retrieve the unannotated (pure) frame if available.
            if hasattr(result, "orig_img"):
                pure_frame = result.orig_img
            elif hasattr(result, "imgs"):
                pure_frame = result.imgs[0]
            else:
                pure_frame = annotated_frame.copy()

            for t_id, bbox in updated_ids.items():
                last_saved_frame[t_id] = frame_no
                out_file = os.path.join(tracked_dir, f"vehicle_{t_id}_frame_{frame_no}.jpg")
                cv2.imwrite(out_file, pure_frame)
                print(f"Saved {out_file}")

                # Perform segmentation and cropping
                segment_vehicle(pure_frame, bbox, t_id, frame_no, base_out, segmentation_model)
                crop_vehicle(pure_frame, bbox, t_id, frame_no, base_out)

        if stream:
            _, encoded_frame = cv2.imencode(".jpg", annotated_frame)
            yield encoded_frame.tobytes()

def parse_args():
    parser = argparse.ArgumentParser(description="Track and segment vehicles from a YouTube video.")
    parser.add_argument("yt_suffix", help="YouTube video suffix (the part after 'watch?v=').")
    parser.add_argument("--line_x", type=int, default=500,
                        help="X-position of the vertical line to detect crossing (default: 500).")
    parser.add_argument("--frame_tolerance", type=int, default=20,
                        help="Frame tolerance to prevent duplicate captures (default: 20).")
    return parser.parse_args()

def main():
    args = parse_args()
    for frame_bytes in process_video(args.yt_suffix, args.line_x, args.frame_tolerance, stream=True):
        frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("Vehicle Detection", frame_np)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
