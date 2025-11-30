from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(data="data/roboflow_yolov8/data.yaml", imgsz=640, conf=0.001, plots=False)


pr_curves = metrics.curves  # list of tuples: (x, y, xlabel, ylabel)
print("Available curves:", [c[2] + " vs " + c[3] for c in pr_curves])

x, y, xlabel, ylabel = pr_curves[0]


num_classes = y.shape[0]
class_names = metrics.names if hasattr(metrics, 'names') else model.names

plt.figure(figsize=(8,6))
for i in range(num_classes):
    plt.plot(x, y[i], label=class_names[i])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title("Precision–Recall Curves per Class")
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

if hasattr(metrics, 'box'):
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50–95: {metrics.box.map:.3f}")
