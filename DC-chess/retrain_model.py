"""
YOLOv8 Training Script for DC-Chess (Windows + GPU safe)
"""

import os
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# ---------------------------------------------------------
#  FIX: Force disable Ultralytics dataset override
# ---------------------------------------------------------
SETTINGS["datasets_dir"] = None


# ---------------------------------------------------------
#  FIX: Ensure correct working directory
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"[INFO] Using project root: {PROJECT_ROOT}")

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "roboflow_yolov8", "data.yaml")
print(f"[INFO] Using dataset config: {DATA_PATH}")

# ---------------------------------------------------------
#  Training Function
# ---------------------------------------------------------
def train_yolo():
    print("\n================ YOLOv8 Training (GPU) ================\n")

    # Load model
    model = YOLO("yolov8n.pt")

    print("✔ GPU Available:", model.device)
    
    # Train
    results = model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,          # GPU 0
        workers=0,         # IMPORTANT for Windows
        cache=False,       # Avoid Windows caching issues
        optimizer="AdamW",
        project="runs/detect",
        name="train_50epochs",
        exist_ok=True
    )

    print("\n================ Training Completed ================\n")
    return results


# ---------------------------------------------------------
#  Windows Multiprocessing Guard
# ---------------------------------------------------------
if __name__ == "__main__":
    train_yolo()
