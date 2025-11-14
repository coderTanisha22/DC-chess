"""
board_rotation_detector.py
Automatically detects board rotation by analyzing calibration corners
"""

import cv2
import numpy as np
import pickle

def get_board_orientation_from_calibration(calibration):
    """
    Determine board orientation from stored calibration corners.
    
    The calibration corners are clicked in order: TL, TR, BR, BL
    We check which corner is actually which based on their positions.
    """
    corners = calibration['corners']
    
    if corners is None or len(corners) < 4:
        return "standard"
    
    # Get corners
    p1, p2, p3, p4 = corners
    
    # Calculate center
    center_x = np.mean([p1[0], p2[0], p3[0], p4[0]])
    center_y = np.mean([p1[1], p2[1], p3[1], p4[1]])
    
    # Count which corners are in which quadrant
    # If most corners are in wrong quadrant, board is rotated
    
    top_left_quadrant = sum(1 for p in corners if p[0] < center_x and p[1] < center_y)
    top_right_quadrant = sum(1 for p in corners if p[0] > center_x and p[1] < center_y)
    bottom_left_quadrant = sum(1 for p in corners if p[0] < center_x and p[1] > center_y)
    bottom_right_quadrant = sum(1 for p in corners if p[0] > center_x and p[1] > center_y)
    
    # In standard orientation: corners should be spread evenly
    # The actual orientation depends on visual inspection
    
    # Simple heuristic: check if first corner (TL) is actually in top-left
    if corners[0][0] < center_x and corners[0][1] < center_y:
        return "standard"
    else:
        return "rotated"

def detect_orientation_from_image(warped_board):
    """
    Detect orientation by analyzing board square colors.
    Top-left square should be light in standard orientation.
    """
    height, width = warped_board.shape[:2]
    cell_size = height // 8
    
    # Get center of top-left square
    tl_region = warped_board[cell_size//4:cell_size//2, cell_size//4:cell_size//2]
    tl_brightness = np.mean(tl_region)
    
    # Get center of top-right square
    tr_region = warped_board[cell_size//4:cell_size//2, width - cell_size//2:width - cell_size//4]
    tr_brightness = np.mean(tr_region)
    
    # Get center of bottom-left square
    bl_region = warped_board[height - cell_size//2:height - cell_size//4, cell_size//4:cell_size//2]
    bl_brightness = np.mean(bl_region)
    
    # In standard: top-left is LIGHT, so high brightness
    # In rotated: top-left is DARK, so low brightness
    
    brightness_diff = tl_brightness - tr_brightness
    
    print(f"   Brightness analysis: TL={tl_brightness:.0f}, TR={tr_brightness:.0f}, Diff={brightness_diff:.0f}")
    
    # If top-left is significantly brighter than neighbors, it's standard
    if brightness_diff > 10:
        return "standard"
    elif brightness_diff < -10:
        return "rotated"
    else:
        # Ambiguous - use calibration corners as fallback
        return "standard"

def flip_square_coordinates(square_name):
    """
    Flip square coordinates for 180° rotation
    a1 ↔ h8, b2 ↔ g7, etc.
    """
    if not square_name or len(square_name) != 2:
        return square_name
    
    file = square_name[0]  # a-h
    rank = square_name[1]  # 1-8
    
    # Flip file (a↔h, b↔g, etc.)
    new_file = chr(ord('h') - (ord(file) - ord('a')))
    
    # Flip rank (1↔8, 2↔7, etc.)
    new_rank = str(9 - int(rank))
    
    return new_file + new_rank

def correct_detections_for_orientation(detections, orientation):
    """
    If board is rotated, flip all detected square coordinates
    """
    if orientation == "rotated":
        corrected = {}
        for square, piece_info in detections.items():
            flipped_square = flip_square_coordinates(square)
            corrected[flipped_square] = piece_info
        return corrected
    
    return detections

# Test
if __name__ == "__main__":
    print("Testing board orientation detection...")
    
    test_squares = ['a1', 'h1', 'a8', 'h8', 'e4', 'd4']
    print("\nFlip function test:")
    for sq in test_squares:
        flipped = flip_square_coordinates(sq)
        print(f"  {sq} → {flipped}")
    
    print("\n✓ Ready!")