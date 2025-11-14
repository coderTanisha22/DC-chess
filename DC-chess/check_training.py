"""
check_training.py
Check if the model actually trained properly
"""

import os
import csv
import pandas as pd

print("\n" + "="*60)
print("CHECKING MODEL TRAINING STATUS")
print("="*60)

# Check if training directory exists
train_dir = "runs/detect/train"

if not os.path.exists(train_dir):
    print(f"\n Training directory not found: {train_dir}")
    print("   Model was never trained!")
    exit(1)

print(f"\n✓ Found training directory: {train_dir}")

# List contents
print("\nContents:")
for item in os.listdir(train_dir):
    path = os.path.join(train_dir, item)
    if os.path.isdir(path):
        print(f"  {item}/")
    else:
        size = os.path.getsize(path)
        print(f"   {item} ({size} bytes)")

# Check results.csv
results_csv = os.path.join(train_dir, "results.csv")

if not os.path.exists(results_csv):
    print(f"\n  No results.csv found")
    print("   Training might not have completed properly")
else:
    print(f"\n Checking training metrics from results.csv...\n")
    
    try:
        df = pd.read_csv(results_csv)
        
        print("Training Statistics:")
        print("-" * 60)
        
        # Get last epoch
        last_epoch = df.iloc[-1]
        
        print(f"\nLast Epoch ({len(df)} total epochs):")
        print(f"  box_loss: {last_epoch.get('box_loss', 'N/A')}")
        print(f"  cls_loss: {last_epoch.get('cls_loss', 'N/A')}")
        print(f"  metrics/precision: {last_epoch.get('metrics/precision', 'N/A')}")
        print(f"  metrics/recall: {last_epoch.get('metrics/recall', 'N/A')}")
        print(f"  metrics/mAP50: {last_epoch.get('metrics/mAP50', 'N/A')}")
        print(f"  metrics/mAP50-95: {last_epoch.get('metrics/mAP50-95', 'N/A')}")
        
        # Check if metrics are good
        mAP = last_epoch.get('metrics/mAP50', 0)
        precision = last_epoch.get('metrics/precision', 0)
        recall = last_epoch.get('metrics/recall', 0)
        
        print("\n" + "-" * 60)
        
        if mAP < 0.3:
            print("\n PROBLEM: Model has very low mAP (< 0.3)")
            print("   Model is not trained well!")
        elif precision < 0.3 or recall < 0.3:
            print("\n PROBLEM: Low precision/recall")
            print("   Model is not working!")
        else:
            print("\n Metrics look reasonable")
            print("   But model still isn't detecting pieces...")
            print("   Possible issue: Dataset mismatch")
        
    except Exception as e:
        print(f" Error reading results.csv: {e}")

# Check best.pt file size
model_path = os.path.join(train_dir, "weights/best.pt")

if os.path.exists(model_path):
    size = os.path.getsize(model_path)
    print(f"\nModel file: {model_path}")
    print(f"   Size: {size} bytes")
    
    if size < 100000:
        print("   Model file is very small (might be corrupted)")
else:
    print(f"\n Model file not found: {model_path}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

print("\nLikely problems:")
print("  1. Dataset was individual piece crops, not full boards")
print("  2. Training didn't converge (low mAP)")
print("  3. Class labels don't match (mismatch between train and test)")
print("  4. Dataset corruption")
print("\nSolutions:")
print("  ✓ Retrain model with data.yaml")
print("  ✓ Check data/roboflow_yolov8/data.yaml for class names")
print("  ✓ Verify training/test split is correct")