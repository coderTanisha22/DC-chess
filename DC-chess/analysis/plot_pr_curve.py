from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Validate model
metrics = model.val(
    data="data/roboflow_yolov8/data.yaml",
    imgsz=640,
    conf=0.001,
    plots=False
)

# Extract PR curves
pr_curves = metrics.curves
print("Available curves:", [f"{x_label} vs {y_label}" for (_, _, x_label, y_label) in pr_curves])

# Unpack PR curve data (x: recall, y: precision per class)
x, y, xlabel, ylabel = pr_curves[0]

# Get class names
class_names = getattr(metrics, "names", model.names)
num_classes = y.shape[0]

# Plot PR curves
plt.figure(figsize=(8, 6))
for cls_id in range(num_classes):
    plt.plot(x, y[cls_id], label=class_names[cls_id])

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title("Precision–Recall Curves per Class")
plt.legend(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Print mAP metrics
if hasattr(metrics, "box"):
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50–95: {metrics.box.map:.3f}")
