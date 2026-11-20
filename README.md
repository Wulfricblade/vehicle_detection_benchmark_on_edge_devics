# Comparative Benchmarking of Lightweight Object Detectors for Vehicle Detection

A beginner-friendly laptop project comparing **YOLOv8n** and **YOLO11n** for vehicle detection.

## What this project measures

- Vehicle-class mAP50 and mAP50-95 on the COCO 2017 validation set
- Precision and recall on the selected COCO vehicle classes
- Video inference latency and FPS on your laptop
- Approximate process RAM use
- Model checkpoint file size

**Important:** Results are specific to your laptop, software versions, input size, and settings. They are not measurements from a Jetson or Raspberry Pi. Describe the work as a laptop-based edge-suitability benchmark unless you later test on a real edge device.

## Models and vehicle classes

- `yolov8n.pt` — YOLOv8 Nano
- `yolo11n.pt` — YOLO11 Nano
- COCO class IDs used: car = 2, motorcycle = 3, bus = 5, truck = 7

Both models use pretrained COCO weights. No training is required for the baseline comparison.

## 1. Install Python

Use Python 3.10 or 3.11 if possible. Install VS Code if you want an editor.

Open a terminal in this project folder.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Check the installation

```bash
python src/check_setup.py
```

The first model run downloads the model weights automatically. An internet connection is required for the first run.

## 3. Get a traffic video

Place an MP4 video in the `videos` folder, for example:

```text
videos/traffic.mp4
```

Use a road/traffic video that you are permitted to use. A short 30–90 second 720p or 1080p video is enough for the demonstration. The program reads a local video; it does not download videos automatically.

You can also use your laptop webcam with `--source 0` if it is available, but a saved video is better for a repeatable comparison.

## 4. Run vehicle detection

```bash
python src/detect_video.py --model yolov8n.pt --source videos/traffic.mp4
python src/detect_video.py --model yolo11n.pt --source videos/traffic.mp4
```

Press `q` to close the preview. Annotated videos are saved in `runs/detect/`.

To use a webcam:

```bash
python src/detect_video.py --model yolo11n.pt --source 0
```

## 5. Prepare a manageable COCO vehicle subset

```bash
python src/prepare_dataset.py
```

This downloads the COCO 2017 validation annotations and a reproducible subset of up to 300 validation images containing cars, motorcycles, buses or trucks. It downloads only selected images rather than the entire COCO image archive. Expect a few hundred MB of downloads depending on the selected images and network conditions.

The script creates `datasets/coco_vehicle/coco_vehicle.yaml` and YOLO-format labels. It retains the original COCO class IDs so the pretrained model's class IDs match the labels.

## 6. Benchmark accuracy

```bash
python src/benchmark_accuracy.py
```

This evaluates the four vehicle classes (car, motorcycle, bus, truck) on the prepared COCO validation subset. The first run may take several minutes or longer on a CPU.

The script writes:
- `results/accuracy_results.csv`
- `results/accuracy_results.json`

The script reports the selected-class metrics returned by Ultralytics. If the dataset download fails, check your internet connection and rerun the same command.

## 7. Benchmark speed on your video

```bash
python src/benchmark_video.py --source videos/traffic.mp4 --max-frames 300
```

Use the exact same video, frame limit, image size, confidence threshold, and device for both models. The script runs on CPU by default to make laptop comparisons consistent.

It writes:
- `results/video_benchmark.csv`

Metrics:
- Average inference latency in milliseconds per frame
- Throughput in frames per second (FPS)
- Approximate maximum process RAM during the run
- Model weight file size in MB

Timing includes the Ultralytics prediction pipeline (preprocessing, model inference, and postprocessing), but excludes video decoding and display. It is therefore an inference-pipeline measurement, not camera-to-screen end-to-end latency.

## 8. Generate graphs

After both benchmark commands finish:

```bash
python src/plot_results.py
```

Charts are saved under `results/`:
- `fps_comparison.png`
- `latency_comparison.png`
- `map_comparison.png`
- `precision_recall_comparison.png`
- `ram_comparison.png`
- `model_size_comparison.png`

If a metric is absent because a run failed, the plotting script skips that chart.

## 9. Create your report

Use `REPORT_TEMPLATE.md` in this folder. Replace all square-bracket placeholders with your actual results and observations. Do not copy example or expected values from the internet as if they were your own experimental results.

## Recommended experimental procedure

1. Restart your laptop or close heavy background applications.
2. Connect the laptop to power and use the same power mode for both runs.
3. Use CPU for both models (`device=cpu`).
4. Use `imgsz=640`, `conf=0.25`, and batch size 1.
5. Run each benchmark twice; report the second run or the average of repeated runs, and state which you used.
6. Keep the same video and frame limit for both models.
7. Record laptop CPU, RAM, OS, Python, PyTorch and Ultralytics versions in your report.

## Limitations to discuss

- The two pretrained models were trained using COCO data; the benchmark uses COCO validation images, which are held out from the training split.
- COCO is a general-purpose dataset, not exclusively a road-scene dataset.
- A short traffic video is useful for speed testing and visual demonstration, but it does not provide ground-truth accuracy unless every evaluated frame is labeled.
- RAM is an approximate process measurement and can vary with other running applications.
- Laptop CPU performance cannot be generalized to embedded edge hardware.

## Project deliverables

- Source code in `src/`
- Annotated video(s) in `runs/detect/`
- CSV/JSON benchmark outputs in `results/`
- Comparison graphs in `results/`
- Completed project report based on `REPORT_TEMPLATE.md`
