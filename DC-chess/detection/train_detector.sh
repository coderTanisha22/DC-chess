#!/usr/bin/env bash
# Train YOLOv8 detector using ultralytics yolo CLI
# Place Roboflow export under data/roboflow_yolov8 with data.yaml
set -euo pipefail

DATA="data/roboflow_yolov8/data.yaml"
MODEL="yolov8s.pt"   # change to yolov8s.pt/yolov8m.pt for larger models
EPOCHS=50
IMGSZ=640
BATCH=16
PROJECT="models/detector"
NAME="exp1"

if [ ! -f "$DATA" ]; then
  echo "ERROR: data.yaml not found: $DATA"
  exit 1
fi

echo "Training YOLOv8 detector..."
python -m ultralytics task=detect mode=train data="$DATA" model="$MODEL" epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH project="$PROJECT" name="$NAME"
echo "Done. Check models/detector/$NAME/weights for best.pt"
