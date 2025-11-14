"""
retrain_model.py
Properly retrain YOLOv8 model for chess piece detection
"""

import os
from ultralytics import YOLO

print("\n" + "="*60)
print("RETRAINING YOLOV8 FOR CHESS PIECE DETECTION")
print("="*60)

# Check data.yaml exists
data_yaml = "data/roboflow_yolov8/data.yaml"

if not os.path.exists(data_yaml):
    print(f"\n❌ Dataset config not found: {data_yaml}")
    exit(1)

print(f"\n✓ Dataset: {data_yaml}")

# Load base model
print("\n📦 Loading base YOLOv8 model...")
model = YOLO('yolov8m.pt')  # Medium model (good balance)
print("✓ Model loaded")

# Train
print("\n🚀 Starting training...")
print("   This may take 5-15 minutes depending on GPU\n")

results = model.train(
    data=data_yaml,
    epochs=30,                     # 50 epochs (CPU training is slower)
    imgsz=416,                     # Smaller image size for CPU
    batch=4,                       # Small batch size for CPU
    patience=15,                   # Early stopping patience
    device='cpu',                  # Use CPU
    save=True,                     # Save checkpoints
    cache=False,                   # Don't cache (saves RAM on CPU)
    verbose=True,                  # Verbose output
    project='runs/detect',         # Save to runs/detect
    name='train',                  # Experiment name
    exist_ok=True,                 # Overwrite existing
    optimizer='SGD',               # SGD optimizer
    lr0=0.01,                      # Initial learning rate
    momentum=0.937,                # Momentum
    weight_decay=0.0005,           # Weight decay
    augment=True,                  # Data augmentation
    workers=2,                     # Fewer workers for CPU
)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Print results
print("\nResults saved to:")
print("  - runs/detect/train/weights/best.pt (best model)")
print("  - runs/detect/train/results.csv (metrics)")
print("  - runs/detect/train/results.png (plots)")

# Get metrics
if results:
    print("\nFinal Metrics:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        for key, value in metrics.items():
            print(f"  {key}: {value}")

print("\n✅ Next: Run simple_move_test.py to test detection!")