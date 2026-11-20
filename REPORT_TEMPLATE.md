# Comparative Benchmarking of Lightweight Object Detectors for Real-Time Vehicle Detection on Edge Devices

**Student name:** [Your name]  
**USN:** [Your USN]  
**Department:** [Department]  
**Institution:** [College]  
**Guide:** [Guide name]  
**Date:** [Submission date]

## Abstract

This project compares two lightweight object detection models, YOLOv8n and YOLO11n, for vehicle detection. Both pretrained models are evaluated using the same COCO 2017 validation data and the same laptop-based video inference setup. The vehicle categories considered are car, motorcycle, bus and truck. Performance is assessed using precision, recall, mAP@0.5, mAP@0.5:0.95, inference latency, frames per second (FPS), approximate process memory use and model file size. The aim is to identify the model that offers the most suitable balance between detection quality and computational cost on the tested laptop. The results are specific to the hardware and software configuration used in this experiment.

## 1. Introduction

Advanced Driver Assistance Systems (ADAS) use sensors and computer vision to help identify surrounding road users and support safer driving. Vehicle detection is an important perception task in applications such as traffic monitoring, collision warning and autonomous driving. Deep-learning detectors can provide accurate detections, but their computational requirements can make deployment on low-power hardware challenging.

This project benchmarks two nano-sized YOLO detectors under consistent conditions. It focuses on the trade-off between detection quality and inference efficiency.

## 2. Problem statement

Large object detection models may require more computing resources than are available on low-power systems. A lightweight detector should provide useful detection accuracy while keeping inference latency, memory use and model size manageable. This project compares YOLOv8n and YOLO11n to study this trade-off on a laptop CPU.

## 3. Objectives

1. Implement vehicle detection using YOLOv8n and YOLO11n.
2. Evaluate both models on the same COCO 2017 validation split.
3. Measure precision, recall, mAP@0.5 and mAP@0.5:0.95 for car, motorcycle, bus and truck.
4. Measure video inference FPS, latency, approximate RAM use and checkpoint size on the same laptop.
5. Compare the results using tables and graphs.
6. Identify the model that best balances detection quality and computational efficiency on the tested system.

## 4. Dataset

The project uses a reproducible subset of up to 300 images from the COCO 2017 validation split. The subset is prepared by `src/prepare_dataset.py`; only images containing at least one of the selected vehicle categories are downloaded. The evaluated COCO category IDs are:
- Car: 2
- Motorcycle: 3
- Bus: 5
- Truck: 7

The validation split is used for evaluation, not model training. Both detectors use pretrained COCO weights. Because the models were pretrained on COCO, this is a held-out validation split from the same dataset family, not an independent external-domain test. The separate traffic video used for speed testing is only a repeatable inference input; it is not used to calculate accuracy unless it has ground-truth annotations.

**Dataset source:** https://cocodataset.org/

## 5. Tools and technologies

- Python [version]
- Ultralytics [version]
- PyTorch [version]
- OpenCV [version]
- Pandas [version]
- Matplotlib [version]
- Laptop operating system: [Windows/macOS/Linux]
- Processor: [CPU model]
- RAM: [installed RAM]
- GPU: [none / model]

## 6. Models

### 6.1 YOLOv8n
YOLOv8n is the nano variant of the YOLOv8 object detection model. It is selected as a compact baseline detector.

### 6.2 YOLO11n
YOLO11n is the nano variant of YOLO11. It is compared with YOLOv8n using the same dataset, image size and inference device.

Both models use pretrained weights. No custom training is performed in this baseline project.

## 7. Methodology

1. Install the Python libraries and load the pretrained model weights.
2. Evaluate each model on the COCO 2017 validation split, restricted to the four vehicle categories.
3. Run each model on the same local traffic video.
4. Use CPU inference, image size 640 and confidence threshold 0.25.
5. Warm up each model before timing.
6. Measure average inference latency and throughput over the same number of video frames.
7. Record process RAM and model checkpoint size.
8. Save the results in CSV files and generate comparison graphs.

The video timing measures the Ultralytics prediction pipeline and excludes video decoding and display. The RAM value is an approximate process-level measurement.

## 8. Evaluation metrics

- **Precision:** proportion of predicted detections that are correct.
- **Recall:** proportion of ground-truth objects that are detected.
- **mAP@0.5:** mean average precision at an intersection-over-union threshold of 0.50.
- **mAP@0.5:0.95:** average mAP across IoU thresholds from 0.50 to 0.95.
- **Latency:** average time spent in the prediction pipeline for one frame, in milliseconds.
- **FPS:** number of frames processed per second; higher is faster.
- **RAM:** approximate process resident memory during benchmarking.
- **Model size:** size of the downloaded model checkpoint file.

## 9. Results

Paste the actual rows from `results/accuracy_results.csv` and `results/video_benchmark.csv` here.

### 9.1 Accuracy results

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLOv8n | [ ] | [ ] | [ ] | [ ] |
| YOLO11n | [ ] | [ ] | [ ] | [ ] |

### 9.2 Speed and resource results

| Model | FPS | Latency (ms) | RAM (MB) | Model size (MB) |
|---|---:|---:|---:|---:|
| YOLOv8n | [ ] | [ ] | [ ] | [ ] |
| YOLO11n | [ ] | [ ] | [ ] | [ ] |

### 9.3 Graphs

Insert the generated graphs:
- FPS comparison
- Latency comparison
- mAP comparison
- Precision and recall comparison
- RAM comparison
- Model-size comparison

## 10. Discussion

Compare the measured results rather than assuming one model is better. State which model achieved higher mAP, which had lower latency, which achieved higher FPS, and which used less memory or storage. Discuss whether a small accuracy difference is worth a speed or resource improvement.

**Your interpretation:** [Write 1–2 paragraphs based on your actual results.]

## 11. Conclusion

This project implemented and compared YOLOv8n and YOLO11n for vehicle detection using a common validation dataset and a common laptop CPU video benchmark. The experiment measured detection quality, inference speed and approximate resource requirements. Based on the measured results, [model name] provided the best balance for the tested laptop because [give evidence from your results]. The conclusion applies to the tested laptop configuration and does not establish performance on dedicated edge hardware.

## 12. Limitations and future work

- Only two nano-sized models are compared.
- The speed test is performed on a laptop CPU, not a dedicated embedded device.
- COCO contains general scenes and is not exclusively a road-driving dataset.
- RAM measurements are approximate and can vary with other processes.
- Future work can test Raspberry Pi or NVIDIA Jetson hardware, export models to ONNX or TensorRT where supported, evaluate quantized models, and use a dedicated driving dataset such as BDD100K or KITTI.

## 13. References

1. COCO Dataset: https://cocodataset.org/
2. Ultralytics YOLOv8 documentation: https://docs.ultralytics.com/models/yolov8/
3. Ultralytics YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/
4. Ultralytics validation documentation: https://docs.ultralytics.com/modes/val/
5. Ultralytics prediction documentation: https://docs.ultralytics.com/modes/predict/

## Appendix: Commands used

```bash
python src/check_setup.py
python src/prepare_dataset.py
python src/detect_video.py --model yolov8n.pt --source videos/traffic.mp4
python src/detect_video.py --model yolo11n.pt --source videos/traffic.mp4
python src/benchmark_accuracy.py
python src/benchmark_video.py --source videos/traffic.mp4 --max-frames 300
python src/plot_results.py
```
