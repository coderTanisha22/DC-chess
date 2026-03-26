"""
debug_detection.py
Debug script to see what's happening with the warping and detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import os

def load_calibration(filename="calibration.pkl"):
    """Load calibration from file"""
    if not os.path.exists(filename):
        print(f" Error: {filename} not found!")
        return None
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    print("\n" + "="*50)
    print("DEBUG: DETECTION PIPELINE")
    print("="*50)
    
    # Load calibration
    print("\n Loading calibration...")
    calibration = load_calibration("calibration.pkl")
    if not calibration:
        return
    print("✓ Calibration loaded")
    print(f"  Board size: {calibration['board_size']}")
    print(f"  Cell size: {calibration['cell_size']}")
    
    # Load model
    print("\nLoading YOLOv8 model...")
    model_path = "runs/detect/train/weights/best.pt"
    model = YOLO(model_path)
    print("✓ Model loaded")
    
    # Test image
    test_image = "data/roboflow_yolov8/test/images/IMG_1333_jpg.rf.c69052a122fb44fbdde203fa498fee71.jpg"
    
    print(f"\nLoading test image...")
    frame = cv2.imread(test_image)
    if frame is None:
        print(f" Image not found: {test_image}")
        return
    
    print(f"✓ Image loaded: {frame.shape}")
    
    # Test 1: Run detection on ORIGINAL image
    print("\n" + "="*50)
    print("TEST 1: Detection on ORIGINAL image")
    print("="*50)
    
    results = model.predict(source=frame, conf=0.25, verbose=False)
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        print(f"✓ Found {count} pieces in original image")
        for i, box in enumerate(results[0].boxes.data.cpu().numpy()):
            conf = box[4]
            cls = int(box[5])
            piece = results[0].names[cls]
            print(f"  Piece {i+1}: {piece} (conf: {conf:.2f})")
    else:
        print(" No pieces found in original image")
    
    # Test 2: Warp the image
    print("\n" + "="*50)
    print("TEST 2: Perspective transformation")
    print("="*50)
    
    warped = cv2.warpPerspective(frame, calibration['transform_matrix'], 
                                (calibration['board_size'], calibration['board_size']))
    
    print(f"✓ Warped image shape: {warped.shape}")
    
    # Save warped image for inspection
    warped_path = "debug_warped.jpg"
    cv2.imwrite(warped_path, warped)
    print(f"✓ Saved warped image to: {warped_path}")
    
    # Test 3: Run detection on WARPED image
    print("\n" + "="*50)
    print("TEST 3: Detection on WARPED image")
    print("="*50)
    
    results = model.predict(source=warped, conf=0.25, verbose=False)
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        print(f"✓ Found {count} pieces in warped image")
        for i, box in enumerate(results[0].boxes.data.cpu().numpy()):
            x1, y1, x2, y2, conf, cls = box[:6]
            piece = results[0].names[int(cls)]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"  Piece {i+1}: {piece}")
            print(f"    Confidence: {conf:.2f}")
            print(f"    Bbox: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
            print(f"    Center: ({center_x:.0f}, {center_y:.0f})")
    else:
        print(" No pieces found in warped image")
    
    # Test 4: Try lower confidence threshold
    print("\n" + "="*50)
    print("TEST 4: Detection with LOWER confidence (0.1)")
    print("="*50)
    
    results = model.predict(source=warped, conf=0.1, verbose=False)
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        print(f"✓ Found {count} pieces with conf=0.1")
    else:
        print("Still no pieces with conf=0.1")
    
    # Test 5: Visual inspection
    print("\n" + "="*50)
    print("TEST 5: Visual Inspection")
    print("="*50)
    
    # Draw grid on warped image
    warped_grid = warped.copy()
    size = calibration['board_size']
    cell_size = calibration['cell_size']
    
    for i in range(9):
        # Vertical lines
        cv2.line(warped_grid, (i * cell_size, 0), (i * cell_size, size), (0, 255, 0), 2)
        # Horizontal lines
        cv2.line(warped_grid, (0, i * cell_size), (size, i * cell_size), (0, 255, 0), 2)
    
    grid_path = "debug_warped_grid.jpg"
    cv2.imwrite(grid_path, warped_grid)
    print(f"✓ Saved grid overlay to: {grid_path}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("\nFiles saved for inspection:")
    print(f"  - debug_warped.jpg (warped board image)")
    print(f"  - debug_warped_grid.jpg (with 8x8 grid overlay)")
    print("\nChecklist:")
    print("  [ ] Does the warped image look correct? (straight-on view)")
    print("  [ ] Are the grid lines aligned with the board squares?")
    print("  [ ] Can you see pieces in the warped image?")
    print("  [ ] Does the original image detect pieces?")
    print("\nNext steps:")
    print("  1. Open debug_warped.jpg and check if it looks correct")
    print("  2. If warping is bad, recalibrate (rerun board_calibration.py)")
    print("  3. If pieces still not detected, check model training")

if __name__ == "__main__":
    main()