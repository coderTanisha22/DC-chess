# detection/detect_and_crop.py
from ultralytics import YOLO
import cv2, os, numpy as np

MODEL = "models/detector/exp1/weights/best.pt"
OUT_DIR = "data/detector_crops"

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL)

def detect_and_crop(image_path, out_prefix):
    results = model.predict(source=image_path, conf=0.25, save=False, verbose=False)
    # results is a list; take first
    r = results[0]
    img = cv2.imread(image_path)
    crops = []
    for i,box in enumerate(r.boxes.data.cpu().numpy()):
        x1,y1,x2,y2,conf,cls = box[:6]
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        crop = img[y1:y2, x1:x2]
        fn = os.path.join(OUT_DIR, f"{out_prefix}_det{i}.jpg")
        cv2.imwrite(fn, crop)
        crops.append(fn)
    return crops

if __name__ == "__main__":
    files = ["data/full_boards/IMG_1336.jpg"]
    for f in files:
        detect_and_crop(f, os.path.splitext(os.path.basename(f))[0])
