import os
import cv2
import numpy as np
from ultralytics import YOLO

def process_video(yt_suffix, line_x=500, frame_tolerance=20, stream=False):
    """
    Streams frames from https://www.youtube.com/watch?v=<yt_suffix>.
    For each frame:
      - Runs detection on vehicles (class=2).
      - Draws a vertical line at `line_x`.
      - If a vehicle center crosses the line within `frame_tolerance`, saves the unannotated frame,
        segments the vehicle, and saves a cropped image.
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
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                # Draw center + ID
                cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), -1)
                try:
                    track_id = int(result.boxes.id[i])
                except:
                    track_id = i
                cv2.putText(annotated_frame, f"ID:{track_id}", (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                # Check crossing
                if abs(cx - line_x) <= frame_tolerance:
                    if track_id in last_saved_frame and (frame_no - last_saved_frame[track_id]) <= frame_tolerance:
                        print(f"Skipping duplicate for ID {track_id} at frame {frame_no}")
                        continue
                    updated_ids[track_id] = coords

        # If we have new vehicles crossing the line, save frames
        if updated_ids:
            if hasattr(result, 'orig_img'):
                pure_frame = result.orig_img
            elif hasattr(result, 'imgs'):
                pure_frame = result.imgs[0]
            else:
                pure_frame = annotated_frame.copy()

            for t_id, bbox in updated_ids.items():
                last_saved_frame[t_id] = frame_no
                out_file = os.path.join(tracked_dir, f"vehicle_{t_id}_frame_{frame_no}.jpg")
                cv2.imwrite(out_file, pure_frame)
                print(f"Saved {out_file}")

                # Segment
                segment_vehicle(pure_frame, bbox, t_id, frame_no, base_out, segmentation_model)
                # Crop
                crop_vehicle(pure_frame, bbox, t_id, frame_no, base_out)

        if stream:
            # Yield annotated frame as JPEG
            _, enc = cv2.imencode(".jpg", annotated_frame)
            yield enc.tobytes()

# def segment_vehicle(frame, box, track_id, frame_no, base_out, seg_model):
#     # ... do segmentation, saving mask + mask viz
#     pass

# def crop_vehicle(frame, box, track_id, frame_no, base_out):
#     # ... do cropping
#     pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("yt_suffix")
    parser.add_argument("--line_x", type=int, default=500)
    parser.add_argument("--frame_tolerance", type=int, default=20)
    args = parser.parse_args()

    for frame_bytes in process_video(args.yt_suffix, args.line_x, args.frame_tolerance, stream=True):
        frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("Vehicle Detection", frame_np)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
