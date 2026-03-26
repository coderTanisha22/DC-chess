"""
manual_calibration_tool.py
Click exactly on the 4 board corners in the image
"""

import cv2
import numpy as np
import pickle
import os
from dotenv import load_dotenv
import os

load_dotenv()

image_path = os.getenv('INPUT_IMAGE_PATH')

corners = []
image_display = None
image_original = None

def mouse_callback(event, x, y, flags, param):
    global corners, image_display
    
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        # Fix: Handle None param
        if param is None:
            param = {}
        
        # Scale coordinates back to original image size if image was resized
        if 'scale' in param:
            x_orig = int(x / param['scale'])
            y_orig = int(y / param['scale'])
        else:
            x_orig = x
            y_orig = y
        
        corners.append([x_orig, y_orig])
        image_display = image_original.copy()
        
        # Draw all corners
        for i, (cx, cy) in enumerate(corners):
            # Scale for display
            if 'scale' in param:
                display_x = int(cx * param['scale'])
                display_y = int(cy * param['scale'])
            else:
                display_x = cx
                display_y = cy
            
            cv2.circle(image_display, (display_x, display_y), 15, (0, 255, 0), -1)
            cv2.circle(image_display, (display_x, display_y), 15, (0, 0, 255), 2)
            cv2.putText(image_display, str(i+1), (display_x-10, display_y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            
            if i > 0:
                if 'scale' in param:
                    px = int(corners[i-1][0] * param['scale'])
                    py = int(corners[i-1][1] * param['scale'])
                else:
                    px, py = corners[i-1]
                cv2.line(image_display, (px, py), (display_x, display_y), (0, 255, 0), 3)
        
        # Close the loop
        if len(corners) == 4:
            if 'scale' in param:
                p1x = int(corners[3][0] * param['scale'])
                p1y = int(corners[3][1] * param['scale'])
                p2x = int(corners[0][0] * param['scale'])
                p2y = int(corners[0][1] * param['scale'])
            else:
                p1x, p1y = corners[3]
                p2x, p2y = corners[0]
            cv2.line(image_display, (p1x, p1y), (p2x, p2y), (0, 255, 0), 3)
        
        cv2.imshow("Click 4 Board Corners (TL, TR, BR, BL)", image_display)
        print(f"✓ Corner {len(corners)}: Original coords ({x_orig}, {y_orig})")
        
        if len(corners) == 4:
            print("\n All 4 corners selected!")
            print("Corners (original image coords):")
            print(f"  TL: {corners[0]}")
            print(f"  TR: {corners[1]}")
            print(f"  BR: {corners[2]}")
            print(f"  BL: {corners[3]}")

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def create_calibration(corners):
    corners = np.array(corners, dtype="float32")
    rect = order_points(corners)
    
    # Calculate board size
    width = int(max(
        np.linalg.norm(rect[1] - rect[0]),
        np.linalg.norm(rect[2] - rect[3])
    ))
    height = int(max(
        np.linalg.norm(rect[3] - rect[0]),
        np.linalg.norm(rect[2] - rect[1])
    ))
    size = max(width, height)
    
    # Create perspective transform
    dst = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Create square mapping
    cell_size = size // 8
    square_map = {}
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    for row in range(8):
        for col in range(8):
            square_name = cols[col] + str(8 - row)
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            square_map[square_name] = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            }
    
    return {
        'corners': rect,
        'transform_matrix': M,
        'board_size': size,
        'square_map': square_map,
        'cell_size': cell_size
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MANUAL BOARD CALIBRATION")
    print("="*60)
    
    
    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        exit(1)
    
    print(f"\n📷 Loading image: {image_path}")
    image_original = cv2.imread(image_path)
    image_display = image_original.copy()
    
    # Resize if too large
    height, width = image_original.shape[:2]
    if height > 1200:
        scale = 1200 / height
        image_original = cv2.resize(image_original, (int(width*scale), int(height*scale)))
        image_display = image_original.copy()
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("Click the 4 corners of the WOODEN BOARD in this order:")
    print("  1. Top-Left (TL) - where top edge meets left edge")
    print("  2. Top-Right (TR) - where top edge meets right edge")
    print("  3. Bottom-Right (BR) - where bottom edge meets right edge")
    print("  4. Bottom-Left (BL) - where bottom edge meets left edge")
    print("\nClick at the VERY EDGE of the wood, not on labels or pieces")
    print("="*60 + "\n")
    
    cv2.namedWindow("Click 4 Board Corners (TL, TR, BR, BL)")
    cv2.setMouseCallback("Click 4 Board Corners (TL, TR, BR, BL)", mouse_callback, {'scale':scale})
    cv2.imshow("Click 4 Board Corners (TL, TR, BR, BL)", image_display)
    
    print("Waiting for 4 clicks...")
    
    while len(corners) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            print("Cancelled")
            exit(1)
    
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.waitKey(100)
    
    # Create calibration
    print("\n🔧 Creating calibration...")
    calibration = create_calibration(corners)
    
    # Save calibration
    with open('calibration.pkl', 'wb') as f:
        pickle.dump(calibration, f)
    
    print("\n✓ Calibration saved to calibration.pkl")
    print(f"✓ Board size: {calibration['board_size']}×{calibration['board_size']}")
    print(f"✓ Cell size: {calibration['cell_size']}×{calibration['cell_size']}")
    
    # Verify
    print("\n🔍 Verifying calibration...")
    img = cv2.imread(image_path)
    warped = cv2.warpPerspective(img, calibration['transform_matrix'],
                                (calibration['board_size'], calibration['board_size']))
    
    warped_grid = warped.copy()
    size = calibration['board_size']
    cell_size = calibration['cell_size']
    
    for i in range(9):
        cv2.line(warped_grid, (i * cell_size, 0), (i * cell_size, size), (0, 255, 0), 2)
        cv2.line(warped_grid, (0, i * cell_size), (size, i * cell_size), (0, 255, 0), 2)
    
    cv2.imwrite('calibration_result.jpg', warped_grid)
    print("✓ Saved calibration_result.jpg")
    
    print("\n Calibration complete!")
    print("Check calibration_result.jpg - grid should align perfectly with squares")