"""
static_move_detector.py
Takes two board images (before and after a move) and detects what move was made.
Perfect for testing and debugging the move detection logic.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from board_calibration import (
    get_perspective_transform, create_square_mapping, 
    bbox_to_square, load_calibration, detect_board_corners
)
import sys

class StaticMoveDetector:
    def __init__(self, model_path, calibration_path="calibration.pkl"):
        """Initialize detector with YOLOv8 model and calibration"""
        self.model = YOLO(model_path)
        
        # Load or create calibration
        calib = load_calibration(calibration_path)
        if calib is None:
            print(" Calibration not found. Creating new one...")
            raise FileNotFoundError(f"Calibration not found at {calibration_path}")
        
        self.calibration = calib
        self.square_map = calib['square_map']
        self.transform_matrix = calib['transform_matrix']
        self.board_size = calib['board_size']
        
    def detect_pieces_in_image(self, image_path):
        """
        Detect all pieces in a single image.
        Returns: dict {square: {'piece': piece_type, 'confidence': conf}}
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Apply perspective transform to normalize board
        warped = cv2.warpPerspective(frame, self.transform_matrix, 
                                    (self.board_size, self.board_size))
        
        # Run YOLO inference
        results = self.model.predict(source=warped, conf=0.25, verbose=False)
        
        detections = {}
        if results and results[0].boxes is not None:
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box[:6]
                bbox = [x1, y1, x2, y2]
                
                # Get piece class name
                piece_type = results[0].names[int(cls)]
                
                # Map bbox center to chess square
                square = bbox_to_square(bbox, self.square_map)
                
                if square:
                    detections[square] = {
                        'piece': piece_type,
                        'confidence': float(conf),
                        'bbox': bbox
                    }
        
        return detections, warped
    
    def detect_move(self, before_state, after_state):
        """
        Compare two board states and detect what move happened.
        
        Returns: {
            'move': 'e2 → e4',
            'from_square': 'e2',
            'to_square': 'e4',
            'piece': 'white_pawn',
            'capture': False,
            'captured_piece': None
        }
        """
        moves = []
        
        # Find all differences
        all_squares = set(before_state.keys()) | set(after_state.keys())
        
        disappeared_pieces = {}  # pieces that were there before but not now
        appeared_pieces = {}     # pieces that are there now but weren't before
        moved_pieces = {}        # pieces that changed position
        
        for square in all_squares:
            before_has = square in before_state
            after_has = square in after_state
            
            if before_has and not after_has:
                # Piece disappeared from this square
                disappeared_pieces[square] = before_state[square]
            
            elif not before_has and after_has:
                # Piece appeared on this square
                appeared_pieces[square] = after_state[square]
            
            elif before_has and after_has:
                # Square has a piece before and after
                if before_state[square]['piece'] != after_state[square]['piece']:
                    # Different piece type = capture + move
                    moved_pieces[square] = {
                        'before': before_state[square],
                        'after': after_state[square]
                    }
        
        # Case 1: Simple move (one piece disappeared, one piece appeared)
        if len(disappeared_pieces) == 1 and len(appeared_pieces) == 1 and len(moved_pieces) == 0:
            from_square = list(disappeared_pieces.keys())[0]
            to_square = list(appeared_pieces.keys())[0]
            piece_type = disappeared_pieces[from_square]['piece']
            
            move = {
                'move': f"{from_square} → {to_square}",
                'from_square': from_square,
                'to_square': to_square,
                'piece': piece_type,
                'capture': False,
                'captured_piece': None,
                'type': 'simple_move'
            }
            moves.append(move)
        
        # Case 2: Capture (piece moved and captured another)
        # One piece disappears from source, different piece disappears from destination, new piece appears at destination
        elif len(disappeared_pieces) == 2 and len(appeared_pieces) == 1:
            # Find which piece moved (the one that appears at destination is the moving piece)
            to_square = list(appeared_pieces.keys())[0]
            moving_piece = appeared_pieces[to_square]['piece']
            
            # Find source square (disappearing piece of same type)
            from_square = None
            captured_piece = None
            
            for square, piece_info in disappeared_pieces.items():
                if piece_info['piece'] == moving_piece:
                    from_square = square
                else:
                    captured_piece = piece_info['piece']
            
            if from_square and captured_piece:
                move = {
                    'move': f"{from_square} → {to_square} (captures {captured_piece})",
                    'from_square': from_square,
                    'to_square': to_square,
                    'piece': moving_piece,
                    'capture': True,
                    'captured_piece': captured_piece,
                    'type': 'capture'
                }
                moves.append(move)
        
        # Case 3: Castling (two pieces move)
        elif len(disappeared_pieces) == 2 and len(appeared_pieces) == 2 and len(moved_pieces) == 0:
            # Check if it's castling (king and rook both move)
            disappeared_list = list(disappeared_pieces.items())
            appeared_list = list(appeared_pieces.items())
            
            pieces_before = {p['piece'] for _, p in disappeared_list}
            pieces_after = {p['piece'] for _, p in appeared_list}
            
            if pieces_before == pieces_after and \
                ('white_king' in pieces_before or 'black_king' in pieces_before):
                
                # Determine castling side
                king_before = [sq for sq, p in disappeared_list if 'king' in p['piece']][0]
                king_after = [sq for sq, p in appeared_list if 'king' in p['piece']][0]
                
                move = {
                    'move': f"{king_before} → {king_after} (Castling)",
                    'from_square': king_before,
                    'to_square': king_after,
                    'piece': disappeared_pieces[king_before]['piece'],
                    'capture': False,
                    'captured_piece': None,
                    'type': 'castling'
                }
                moves.append(move)
        
        # Case 4: Pawn promotion (pawn disappears, new piece appears)
        elif len(disappeared_pieces) == 1 and len(appeared_pieces) == 1:
            from_square = list(disappeared_pieces.keys())[0]
            to_square = list(appeared_pieces.keys())[0]
            moving_piece = disappeared_pieces[from_square]['piece']
            new_piece = appeared_pieces[to_square]['piece']
            
            if 'pawn' in moving_piece and 'pawn' not in new_piece:
                move = {
                    'move': f"{from_square} → {to_square} (Promotion to {new_piece})",
                    'from_square': from_square,
                    'to_square': to_square,
                    'piece': moving_piece,
                    'promotion_to': new_piece,
                    'capture': False,
                    'captured_piece': None,
                    'type': 'promotion'
                }
                moves.append(move)
        
        return moves
    
    def print_board_state(self, detections, title="Board State"):
        """Print current board state as ASCII"""
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        
        for square, piece_info in detections.items():
            col = cols.index(square[0])
            row = 8 - int(square[1])
            piece_name = piece_info['piece']
            
            # Unicode pieces
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
    
    def print_detections(self, detections, title="Detected Pieces"):
        """Print all detected pieces with confidence"""
        print(f"\n📍 {title}:")
        if not detections:
            print("   (none)")
            return
        
        for square in sorted(detections.keys()):
            piece_info = detections[square]
            conf = piece_info['confidence']
            piece = piece_info['piece']
            print(f"   {square}: {piece} (confidence: {conf:.2f})")
        
        print(f"   Total: {len(detections)} pieces")
    
    def compare_boards(self, image1_path, image2_path):
        """Main function: load two images and detect the move"""
        print("\n" + "="*50)
        print("STATIC CHESS MOVE DETECTION")
        print("="*50)
        
        # Detect pieces in both images
        print(f"\nProcessing image 1: {image1_path}")
        state_1, warped_1 = self.detect_pieces_in_image(image1_path)
        self.print_detections(state_1, "BEFORE - Detected Pieces")
        self.print_board_state(state_1, "BEFORE - Board State")
        
        print(f"\n Processing image 2: {image2_path}")
        state_2, warped_2 = self.detect_pieces_in_image(image2_path)
        self.print_detections(state_2, "AFTER - Detected Pieces")
        self.print_board_state(state_2, "AFTER - Board State")
        
        # Detect moves
        print("\nANALYZING MOVES...")
        moves = self.detect_move(state_1, state_2)
        
        if not moves:
            print("\n  No clear move detected!")
            print("\nDifferences between boards:")
            all_squares = set(state_1.keys()) | set(state_2.keys())
            for square in sorted(all_squares):
                before = state_1.get(square, {'piece': 'empty'})
                after = state_2.get(square, {'piece': 'empty'})
                
                before_piece = before['piece']
                after_piece = after['piece']
                
                if before_piece != after_piece:
                    print(f"   {square}: {before_piece} → {after_piece}")
        else:
            print(f"\nMOVE DETECTED ({len(moves)} move(s)):")
            for i, move in enumerate(moves, 1):
                print(f"\n   Move {i}:")
                print(f"   From: {move['from_square']}")
                print(f"   To: {move['to_square']}")
                print(f"   Piece: {move['piece']}")
                print(f"   Type: {move['type']}")
                if move['capture']:
                    print(f"   Captured: {move['captured_piece']}")
                print(f"\n   ➜ {move['move']}")
        
        return moves


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python static_move_detector.py <image1_path> <image2_path>")
        print("\nExample:")
        print("  python static_move_detector.py before.jpg after.jpg")
        sys.exit(1)
    
    image1 = sys.argv[1]
    image2 = sys.argv[2]
    
    try:
        detector = StaticMoveDetector(
            model_path="runs/detect/train/weights/best.pt",
            calibration_path="calibration.pkl"
        )
        detector.compare_boards(image1, image2)
    
    except FileNotFoundError as e:
        print(f" Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()