
import cv2
import torch
import time
import csv
import os
from ultralytics import YOLO

# Video and trained model paths
VIDEO = "videos/traffic_test.mp4"

MODELS = {
    "YOLOv8n": "runs/detect/runs/benchmark/yolov8n/weights/best.pt",
    "YOLO11n": "runs/detect/runs/benchmark/yolo11n/weights/best.pt"
}

os.makedirs("benchmark_results", exist_ok=True)

results = []

for model_name, model_path in MODELS.items():

    print(f"\nTesting {model_name}...")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    total_inference_time = 0
    total_detections = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_start = time.perf_counter()

        prediction = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            device=0,
            verbose=False
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_end = time.perf_counter()

        total_inference_time += inference_end - inference_start
        total_detections += len(prediction[0].boxes)
        processed_frames += 1

    end_time = time.perf_counter()
    cap.release()

    total_time = end_time - start_time
    avg_latency = (total_inference_time / processed_frames) * 1000
    avg_fps = processed_frames / total_time

    gpu_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() else 0
    )

    results.append({
        "Model": model_name,
        "Frames": processed_frames,
        "Detections": total_detections,
        "Average FPS": round(avg_fps, 2),
        "Latency (ms)": round(avg_latency, 2),
        "Processing Time (s)": round(total_time, 2),
        "Peak GPU Memory (GB)": round(gpu_memory, 2)
    })

    print(f"Frames processed: {processed_frames}/{total_frames}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Peak GPU memory: {gpu_memory:.2f} GB")

# Save benchmark results
with open("benchmark_results/video_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nBenchmark completed!")
print("Results saved to benchmark_results/video_benchmark.csv")
