import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

VEHICLE_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

def main():
    parser = argparse.ArgumentParser(description="Detect vehicles in a video/webcam.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model checkpoint, e.g. yolov8n.pt or yolo11n.pt")
    parser.add_argument("--source", required=True, help="Video path or webcam index such as 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)

    # stream=True processes frames incrementally rather than storing all results.
    results = model.predict(
        source=source,
        stream=True,
        classes=VEHICLE_IDS,
        imgsz=args.imgsz,
        conf=args.conf,
        device="cpu",
        show=True,
        save=True,
        project="runs/detect",
        name=Path(args.model).stem,
        exist_ok=True,
        verbose=False,
    )

    try:
        for _ in results:
            # The preview is handled by Ultralytics. Press q in the preview window.
            pass
    except KeyboardInterrupt:
        print("Detection stopped by user.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
