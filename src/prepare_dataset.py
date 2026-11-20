import json
import random
import urllib.request
import zipfile
from pathlib import Path

# Downloads COCO annotations, then downloads only a manageable subset of
# validation images containing at least one vehicle of interest.
ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "https://images.cocodataset.org/val2017/"
TARGET_CATEGORIES = {3, 4, 6, 8}  # COCO category IDs: car, motorcycle, bus, truck
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
# COCO category_id -> pretrained YOLO COCO class index
CATEGORY_TO_CLASS = {3: 2, 4: 3, 6: 5, 8: 7}
MAX_IMAGES = 300
ROOT = Path("datasets/coco_vehicle")
ANN_ZIP = Path("datasets/annotations_trainval2017.zip")
ANN_JSON = Path("datasets/annotations/instances_val2017.json")

def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination}")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    download(ANNOTATIONS_URL, ANN_ZIP)

    if not ANN_JSON.exists():
        print("Extracting validation annotations...")
        with zipfile.ZipFile(ANN_ZIP) as zf:
            member = "annotations/instances_val2017.json"
            zf.extract(member, path="datasets")

    with ANN_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        if ann["category_id"] in TARGET_CATEGORIES and not ann.get("iscrowd", 0):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    candidate_ids = sorted(anns_by_image.keys())
    random.Random(42).shuffle(candidate_ids)
    selected_ids = candidate_ids[:MAX_IMAGES]

    image_dir = ROOT / "images" / "val2017"
    label_dir = ROOT / "labels" / "val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for image_id in selected_ids:
        info = images[image_id]
        filename = info["file_name"]
        image_path = image_dir / filename
        if not image_path.exists():
            try:
                download(IMAGE_BASE_URL + filename, image_path)
            except Exception as exc:
                print(f"Could not download {filename}: {exc}")
                continue

        width, height = info["width"], info["height"]
        lines = []
        for ann in anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            class_id = CATEGORY_TO_CLASS[ann["category_id"]]
            xc = (x + w / 2) / width
            yc = (y + h / 2) / height
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w/width:.6f} {h/height:.6f}")

        if lines:
            (label_dir / (Path(filename).stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            downloaded += 1

    # Keep the original COCO class IDs so pretrained model class IDs align.
    yaml_lines = [
        f"path: {ROOT.resolve().as_posix()}",
        "train: images/val2017",
        "val: images/val2017",
        "names:",
    ]
    for i, name in enumerate(COCO_NAMES):
        yaml_lines.append(f"  {i}: {name!r}")
    yaml_path = ROOT / "coco_vehicle.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"\nPrepared {downloaded} labeled vehicle images.")
    print(f"Dataset YAML: {yaml_path}")
    print("Vehicle classes evaluated: car (2), motorcycle (3), bus (5), truck (7)")
    print("This is a reproducible subset of COCO val2017, not the full validation split.")

if __name__ == "__main__":
    main()
