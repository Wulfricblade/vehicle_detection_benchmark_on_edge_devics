import csv
import json
import time
from pathlib import Path
import torch
from ultralytics import YOLO

MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
}
VEHICLE_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
OUT = Path("results")
OUT.mkdir(exist_ok=True, parents=True)

def main():
    dataset_yaml = Path("datasets/coco_vehicle/coco_vehicle.yaml")
    if not dataset_yaml.exists():
        raise SystemExit("Dataset not prepared. Run: python src/prepare_dataset.py")
    rows = []
    for label, weights in MODELS.items():
        print(f"\nEvaluating {label} ({weights}) on COCO val vehicle classes...")
        model = YOLO(weights)
        start = time.perf_counter()

        metrics = model.val(
            data="datasets/coco_vehicle/coco_vehicle.yaml",
            split="val",
            imgsz=640,
            batch=1,
            device="cpu",
            workers=0,
            classes=VEHICLE_IDS,
            plots=False,
            verbose=True,
        )
        elapsed = time.perf_counter() - start

        box = metrics.box
        row = {
            "model": label,
            "weights": weights,
            "classes": "car,motorcycle,bus,truck",
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "evaluation_time_min": round(elapsed / 60, 2),
        }
        rows.append(row)
        print("Result:", row)

    csv_path = OUT / "accuracy_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (OUT / "accuracy_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved accuracy results to {csv_path.resolve()}")
    print("Note: metrics are for the selected four COCO vehicle classes.")

if __name__ == "__main__":
    main()
