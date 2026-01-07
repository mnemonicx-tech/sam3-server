import os
import cv2
import shutil
import random
import numpy as np

# ---------------- CONFIG ----------------
SAM_MASK_ROOT = "sam3_output"
IMAGE_ROOT = "node_downloads"
YOLO_OUT = "yolo_dataset"

TRAIN_RATIO = 0.9
MIN_CONTOUR_AREA = 500  # skip tiny junk masks
RANDOM_SEED = 42
# ----------------------------------------

random.seed(RANDOM_SEED)

IMG_TRAIN = os.path.join(YOLO_OUT, "images/train")
IMG_VAL = os.path.join(YOLO_OUT, "images/val")
LBL_TRAIN = os.path.join(YOLO_OUT, "labels/train")
LBL_VAL = os.path.join(YOLO_OUT, "labels/val")

for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(p, exist_ok=True)

# ---------------- Class Mapping ----------------
categories = sorted([
    d for d in os.listdir(SAM_MASK_ROOT)
    if os.path.isdir(os.path.join(SAM_MASK_ROOT, d))
])

class_map = {cat: idx for idx, cat in enumerate(categories)}

with open(os.path.join(YOLO_OUT, "classes.txt"), "w") as f:
    for c in categories:
        f.write(c + "\n")

print("📌 Class mapping:")
for k, v in class_map.items():
    print(f"{v}: {k}")

# ---------------- Processing ----------------
samples = []

for category in categories:
    mask_dir = os.path.join(SAM_MASK_ROOT, category)
    img_dir = os.path.join(IMAGE_ROOT, category)

    for file in os.listdir(mask_dir):
        if not file.endswith("_mask.png"):
            continue

        img_name = file.replace("_mask.png", ".jpg")
        mask_path = os.path.join(mask_dir, file)
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        samples.append((category, img_path, mask_path))

random.shuffle(samples)
split_idx = int(len(samples) * TRAIN_RATIO)

train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

def process(samples, img_out, lbl_out):
    for category, img_path, mask_path in samples:
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        h, w = mask.shape

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        polygon = cnt.reshape(-1, 2)

        if len(polygon) < 6:
            continue

        yolo_line = [str(class_map[category])]
        for x, y in polygon:
            yolo_line.append(f"{x / w:.6f}")
            yolo_line.append(f"{y / h:.6f}")

        base = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(img_out, base))

        label_path = os.path.join(
            lbl_out, base.replace(".jpg", ".txt")
        )

        with open(label_path, "w") as f:
            f.write(" ".join(yolo_line))

process(train_samples, IMG_TRAIN, LBL_TRAIN)
process(val_samples, IMG_VAL, LBL_VAL)

print("\n🎉 YOLOv8 segmentation dataset ready!")
print(f"Train images: {len(train_samples)}")
print(f"Val images: {len(val_samples)}")
