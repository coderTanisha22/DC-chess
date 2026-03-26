"""
test_model_raw.py
Test if the model can detect pieces on RAW (non-warped) images.
This tells us if the model is working at all.
"""

import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

model_path = os.getenv('MODEL_PATH', 'runs/detect/train/weights/best.pt')
conf_threshold = float(os.getenv('CONF_THRESHOLD', 0.25))
test_dir = os.getenv('TEST_IMAGE_DIR', 'data/roboflow_yolov8/test/images')

def test_model():
    print("\n" + "="*60)
    print("TESTING YOLOV8 MODEL ON RAW IMAGES")
    print("="*60)
    
    print(f"\n Loading model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Model not found!")
        return
    
    model = YOLO(model_path)
    print("✓ Model loaded\n")
    
    # Test on roboflow test images (raw, not warped)
    test_dir = "data/roboflow_yolov8/test/images"
    
    images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])[:3]
    
    print(f"Testing on {len(images)} images from {test_dir}\n")
    
    total_detections = 0
    
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        
        print(f"{img_name}")
        img = cv2.imread(img_path)
        
        # Run inference
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        if results[0].boxes is not None:
            count = len(results[0].boxes)
            total_detections += count
            print(f"   ✓ Found {count} pieces")
            
            for i, box in enumerate(results[0].boxes.data.cpu().numpy()):
                conf = box[4]
                cls = int(box[5])
                piece = results[0].names[cls]
                print(f"     {i+1}. {piece} (conf: {conf:.2f})")
        else:
            print(f"  No pieces detected")
        
        print()
    
    print("="*60)
    print(f"SUMMARY: {total_detections} total pieces found")
    print("="*60)
    
    if total_detections == 0:
        print("\n Ohh No! MODEL IS NOT WORKING!")
        print("\nPossible causes:")
        print("  1. Model was trained on individual piece crops, not full boards")
        print("  2. Model didn't train properly (low accuracy)")
        print("  3. Confidence threshold too high (try 0.1 instead of 0.25)")
        print("\nNext steps:")
        print("  → Check if training data was full boards or individual pieces")
        print("  → Check training loss/accuracy in runs/detect/train/results.csv")
        print("  → Try retraining model with full board images")
    else:
        print(f"\nYayy!!! MODEL IS WORKING!")
        print(f"   Detected {total_detections} pieces across {len(images)} images")
        print("\n   Next: The warping might need adjustment")

if __name__ == "__main__":
    test_model()