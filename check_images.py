"""
check_images.py
Quick check - do these test images actually have chess pieces?
"""

import cv2
import os

test_dir = "data/roboflow_yolov8/test/images"

images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])[:5]

print("\n" + "="*60)
print("CHECKING TEST IMAGES")
print("="*60)

for img_name in images:
    path = os.path.join(test_dir, img_name)
    img = cv2.imread(path)
    
    if img is None:
        print(f"\n❌ {img_name}")
        print("   COULD NOT LOAD")
        continue
    
    height, width = img.shape[:2]
    
    # Check if image has actual content (not mostly white/empty)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    
    print(f"\n✓ {img_name}")
    print(f"   Size: {width}×{height}")
    print(f"   Mean brightness: {mean_brightness:.1f} (0=black, 255=white)")
    
    if mean_brightness > 200:
        print(f"   ⚠️  Image is very bright/empty!")
    elif mean_brightness < 50:
        print(f"   ⚠️  Image is very dark!")
    else:
        print(f"   ✓ Good brightness")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)
print("\nUse images from TRAIN set instead of TEST set.")
print("They're more likely to have pieces visible.")
print("\nOr use one of your original board photos:")
print("  - data/full_boards/IMG_1336.jpg")
print("  - data/roboflow_yolov8/train/images/...")