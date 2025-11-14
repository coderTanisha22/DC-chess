from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_PATH = "/home/tanisha_ubuntu/Desktop/dc project/dc-vs-chess-detection/DC-chess/DC-chess/data/roboflow_yolov8/train/images/IMG_1583_jpg.rf.53b84613d31f78671fc439bdae8ee11d.jpg"
DATA_YAML = "data/roboflow_yolov8/data.yaml"

# Load model once
model = YOLO(MODEL_PATH)

# -------------------------
# Run Inference
# -------------------------
results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=True,          # Saves annotated image automatically
    verbose=False
)

# Print detected labels + confidence
result = results[0]
for box in result.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = result.names[cls_id]
    print(f"{label}: {conf:.2f}")

# -------------------------
# Display the saved detection image
# -------------------------
annotated_img_path = result.path       # Path to saved annotated image
img = cv2.imread(annotated_img_path)

if img is None:
    raise ValueError(f"Failed to load annotated image: {annotated_img_path}")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Pieces")
plt.show()

# -------------------------
# Validation Metrics
# -------------------------
metrics = model.val(data=DATA_YAML)
print(metrics)
