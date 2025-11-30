from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("runs/detect/train/weights/best.pt")


results = model.predict(
    "/home/tanisha_ubuntu/Desktop/dc project/dc-vs-chess-detection/DC-chess/DC-chess/data/roboflow_yolov8/train/images/IMG_1583_jpg.rf.53b84613d31f78671fc439bdae8ee11d.jpg",
    conf=0.25,
    save=True  
)


for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = results[0].names[cls_id]
    print(f"{label}: {conf:.2f}")


img_path = results[0].path  
img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Pieces")
plt.show()


from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="data/roboflow_yolov8/data.yaml")
print(metrics)
