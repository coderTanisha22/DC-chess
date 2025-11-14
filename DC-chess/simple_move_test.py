"""
simple_move_test.py
Test piece detection and move detection with minimal setup.
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

def bbox_to_square(bbox, square_map):
    """Convert bounding box center to chess square"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    for square_name, region in square_map.items():
        if (region['x1'] <= center_x < region['x2'] and 
            region['y1'] <= center_y < region['y2']):
            return square_name
    return None

def detect_pieces(image_path, model, calibration):
    """Detect all pieces in an image"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f" Image not found: {image_path}")
        return None
    
    print(f"\n📷 Processing: {image_path}")
    
    # Apply perspective transform
    warped = cv2.warpPerspective(frame, calibration['transform_matrix'], 
                                (calibration['board_size'], calibration['board_size']))
    
    # Run YOLO
    print("🤖 Running YOLOv8 inference...")
    results = model.predict(source=warped, conf=0.25, verbose=False)
    
    detections = {}
    if results and results[0].boxes is not None:
        print(f"✓ Found {len(results[0].boxes)} pieces")
        
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box[:6]
            bbox = [x1, y1, x2, y2]
            piece_type = results[0].names[int(cls)]
            square = bbox_to_square(bbox, calibration['square_map'])
            
            if square:
                detections[square] = {
                    'piece': piece_type,
                    'confidence': float(conf)
                }
    else:
        print("No pieces detected")
    
    return detections

def print_board(detections, title="Board State"):
    """Print ASCII board"""
    board = [['.' for _ in range(8)] for _ in range(8)]
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    for square, piece_info in detections.items():
        col = cols.index(square[0])
        row = 8 - int(square[1])
        piece_name = piece_info['piece']
        
        abbrev = {
            'white_king': '♔', 'white_queen': '♕', 'white_rook': '♖',
            'white_bishop': '♗', 'white_knight': '♘', 'white_pawn': '♙',
            'black_king': '♚', 'black_queen': '♛', 'black_rook': '♜',
            'black_bishop': '♝', 'black_knight': '♞', 'black_pawn': '♟'
        }
        board[row][col] = abbrev.get(piece_name, '?')
    
    print(f"\n{'='*42}")
    print(f"{title}")
    print(f"{'='*42}")
    print("  a b c d e f g h")
    for row in range(8):
        print(f"{8-row} {' '.join(board[row])} {8-row}")
    print("  a b c d e f g h")
    print(f"{'='*42}")

def print_pieces(detections, title="Detected Pieces"):
    """Print all pieces"""
    print(f"\n{title}:")
    if not detections:
        print("  (none)")
        return
    
    for square in sorted(detections.keys()):
        piece = detections[square]['piece']
        conf = detections[square]['confidence']
        print(f"  {square}: {piece} ({conf:.2f})")

def detect_move(state_before, state_after):
    """Find what move happened"""
    moves = []
    
    # Find disappeared pieces
    disappeared = {}
    for square, piece_info in state_before.items():
        if square not in state_after:
            disappeared[square] = piece_info
    
    # Find appeared pieces
    appeared = {}
    for square, piece_info in state_after.items():
        if square not in state_before:
            appeared[square] = piece_info
    
    # Simple move: one disappeared, one appeared of same type
    if len(disappeared) == 1 and len(appeared) == 1:
        from_sq = list(disappeared.keys())[0]
        to_sq = list(appeared.keys())[0]
        piece = disappeared[from_sq]['piece']
        
        # Check if it's the same piece
        if disappeared[from_sq]['piece'] == appeared[to_sq]['piece']:
            moves.append({
                'from': from_sq,
                'to': to_sq,
                'piece': piece,
                'type': 'move'
            })
    
    # Capture: two disappeared, one appeared
    elif len(disappeared) == 2 and len(appeared) == 1:
        to_sq = list(appeared.keys())[0]
        moving_piece = appeared[to_sq]['piece']
        
        from_sq = None
        captured = None
        for sq, p in disappeared.items():
            if p['piece'] == moving_piece:
                from_sq = sq
            else:
                captured = p['piece']
        
        if from_sq and captured:
            moves.append({
                'from': from_sq,
                'to': to_sq,
                'piece': moving_piece,
                'captured': captured,
                'type': 'capture'
            })
    
    return moves

def main():
    print("\n" + "="*50)
    print("CHESS MOVE DETECTION TEST")
    print("="*50)
    
    # Load calibration
    print("\nLoading calibration...")
    calibration = load_calibration("calibration.pkl")
    if not calibration:
        return
    print("✓ Calibration loaded")
    
    # Load model
    print("\n Loading YOLOv8 model...")
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    model = YOLO(model_path)
    print("✓ Model loaded")
    
    # Get two test images
    print("\nFinding test images...")
    
    # Look for images in roboflow dataset
    test_dir = "data/roboflow_yolov8/test/images"
    if os.path.exists(test_dir):
        images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
        if len(images) >= 2:
            img1 = os.path.join(test_dir, images[0])
            img2 = os.path.join(test_dir, images[1])
        else:
            print(f" Only {len(images)} images found in {test_dir}")
            return
    else:
        print(f"Directory not found: {test_dir}")
        return
    
    print(f"✓ Found test images:")
    print(f"  1. {images[0]}")
    print(f"  2. {images[1]}")
    
    # Detect pieces in both images
    state_1 = detect_pieces(img1, model, calibration)
    if state_1 is None:
        return
    
    state_2 = detect_pieces(img2, model, calibration)
    if state_2 is None:
        return
    
    # Print results
    print_pieces(state_1, "BEFORE - Detected Pieces")
    print_board(state_1, "BEFORE - Board State")
    
    print_pieces(state_2, "AFTER - Detected Pieces")
    print_board(state_2, "AFTER - Board State")
    
    # Detect moves
    print("\n🔍 Analyzing moves...")
    moves = detect_move(state_1, state_2)
    
    if moves:
        print(f"\nMOVE(S) DETECTED:")
        for move in moves:
            print(f"\n  From: {move['from']}")
            print(f"  To: {move['to']}")
            print(f"  Piece: {move['piece']}")
            if move['type'] == 'capture':
                print(f"  Captured: {move['captured']}")
            print(f"\n  ➜ {move['from']} → {move['to']}")
    else:
        print("\n No clear move detected")
        print("\nDifferences:")
        all_squares = set(state_1.keys()) | set(state_2.keys())
        for sq in sorted(all_squares):
            b1 = state_1.get(sq, {}).get('piece', 'empty')
            b2 = state_2.get(sq, {}).get('piece', 'empty')
            if b1 != b2:
                print(f"  {sq}: {b1} → {b2}")

if __name__ == "__main__":
    main()