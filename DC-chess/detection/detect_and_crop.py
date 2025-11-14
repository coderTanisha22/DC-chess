# detection/detect_and_crop.py

import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/detector/exp1/weights/best.pt"
OUT_DIR = "data/detector_crops"

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)


def detect_and_crop(image_path, out_prefix):
    """Run YOLO detection and save cropped regions."""
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    crops = []

    for idx, box in enumerate(result.boxes.data.cpu().numpy()):
        x1, y1, x2, y2, score, cls_id = box[:6]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Crop image safely (avoids crashes if coords go outside)
        crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

        out_path = os.path.join(OUT_DIR, f"{out_prefix}_det{idx}.jpg")
        cv2.imwrite(out_path, crop)
        crops.append(out_path)

    return crops


if __name__ == "__main__":
    input_files = [
        "data/full_boards/IMG_1336.jpg"
    ]

    for path in input_files:
        prefix = os.path.splitext(os.path.basename(path))[0]
        detect_and_crop(path, prefix)
