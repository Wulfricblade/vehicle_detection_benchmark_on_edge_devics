
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
GRAPH_DIR = Path("benchmark_results/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "YOLOv8n": Path("runs/detect/runs/benchmark/yolov8n/results.csv"),
    "YOLO11n": Path("runs/detect/runs/benchmark/yolo11n/results.csv")
}

VIDEO_FILE = Path("benchmark_results/video_benchmark.csv")

# Load training histories
histories = {
    name: pd.read_csv(path)
    for name, path in MODELS.items()
}

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300)
    plt.close()
    print("Saved:", GRAPH_DIR / filename)


# 1. Training loss comparison
for metric, title, filename in [
    ("train/box_loss", "Training Box Loss", "training_box_loss.png"),
    ("train/cls_loss", "Training Classification Loss", "training_cls_loss.png"),
    ("train/dfl_loss", "Training DFL Loss", "training_dfl_loss.png"),
    ("val/box_loss", "Validation Box Loss", "validation_box_loss.png")
]:
    plt.figure(figsize=(9, 5))

    for name, df in histories.items():
        plt.plot(df["epoch"], df[metric], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_plot(filename)


# 2. Precision and recall over epochs
for metric, title, filename in [
    ("metrics/precision(B)", "Precision Comparison", "precision_epochs.png"),
    ("metrics/recall(B)", "Recall Comparison", "recall_epochs.png"),
    ("metrics/mAP50(B)", "mAP50 Comparison", "map50_epochs.png"),
    ("metrics/mAP50-95(B)", "mAP50-95 Comparison", "map5095_epochs.png")
]:
    plt.figure(figsize=(9, 5))

    for name, df in histories.items():
        plt.plot(df["epoch"], df[metric], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_plot(filename)


# 3. Video FPS comparison
if VIDEO_FILE.exists():
    video = pd.read_csv(VIDEO_FILE)

    for metric, title, ylabel, filename in [
        ("Average FPS", "Video FPS Comparison",
         "Frames per second", "video_fps.png"),

        ("Latency (ms)", "Video Inference Latency",
         "Milliseconds", "video_latency.png"),

        ("Processing Time (s)", "Total Video Processing Time",
         "Seconds", "video_processing_time.png"),

        ("Detections", "Total Vehicle Detections",
         "Number of detections", "video_detections.png")
    ]:
        plt.figure(figsize=(8, 5))
        plt.bar(video["Model"], video[metric])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Model")

        save_plot(filename)

else:
    print("Video benchmark CSV not found.")


# 4. Final epoch validation comparison
final_metrics = {
    "YOLOv8n": {
        "Precision": 0.93666,
        "Recall": 0.75201,
        "mAP50": 0.79746,
        "mAP50-95": 0.68697
    },
    "YOLO11n": {
        "Precision": 0.94119,
        "Recall": 0.74932,
        "mAP50": 0.79678,
        "mAP50-95": 0.68943
    }
}

best_metrics = pd.DataFrame(final_metrics).T.reset_index()
best_metrics.rename(columns={"index": "Model"}, inplace=True)

for metric in ["Precision", "Recall", "mAP50", "mAP50-95"]:
    plt.figure(figsize=(8, 5))
    plt.bar(best_metrics["Model"], best_metrics[metric])
    plt.title(f"Final Epoch - {metric} Comparison")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.ylim(0, 1)
    save_plot(f"final_{metric.lower().replace('-', '_')}.png")