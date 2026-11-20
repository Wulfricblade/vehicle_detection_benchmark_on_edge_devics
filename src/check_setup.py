import platform
import sys
import cv2
import psutil
import torch
import ultralytics

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("CPU:", platform.processor() or "Not reported")
print("Logical CPU cores:", psutil.cpu_count(logical=True))
print("RAM (GB):", round(psutil.virtual_memory().total / (1024**3), 2))
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Benchmark device in this starter project: CPU")
