"""
backend.py
Complete Flask backend with rotation detection and edge padding
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from board_rotation_detector import correct_detections_for_orientation, detect_orientation_from_image
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

calibration = None
model = None

def load_resources():
    global calibration, model
    
    print("Loading calibration...")
    with open('calibration.pkl', 'rb') as f:
        calibration = pickle.load(f)
    print(f"✓ Calibration loaded (board size: {calibration['board_size']})")
    
    print("Loading YOLOv8 model...")
    model = YOLO('runs/detect/train/weights/best.pt')
    print("✓ Model loaded")

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

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model': 'loaded' if model else 'not loaded',
        'calibration': 'loaded' if calibration else 'not loaded'
    })

@app.route('/detect', methods=['POST'])
def detect_pieces():
    """Detect pieces with rotation correction and edge padding"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        
        print(f"\n📷 Processing: {file.filename}")
        
        # Apply perspective transform
        print("   Applying perspective transform...")
        warped = cv2.warpPerspective(
            img, 
            calibration['transform_matrix'],
            (calibration['board_size'], calibration['board_size'])
        )
        
        # DETECT BOARD ORIENTATION
        print(" Detecting board orientation...")
        orientation = detect_orientation_from_image(warped)
        print(f"   Orientation: {orientation}")
        
        # For now, always use "standard" to avoid confusion
        # The correction will be done in frontend if needed
        orientation = "standard"
        
        # ADD PADDING to detect edge pieces
        print("   Adding padding for edge detection...")
        pad_size = 50
        padded_warped = cv2.copyMakeBorder(
            warped, 
            pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_REFLECT
        )
        
        # Run YOLO inference
        print(" Running YOLOv8 inference...")
        results = model.predict(source=padded_warped, conf=0.25, verbose=False)
        
        # Parse detections
        detections = {}
        if results[0].boxes is not None:
            print(f"   Found {len(results[0].boxes)} raw detections")
            
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box[:6]
                
                # Remove padding offset
                x1 = x1 - pad_size
                y1 = y1 - pad_size
                x2 = x2 - pad_size
                y2 = y2 - pad_size
                
                # Skip if completely outside original bounds
                if x2 < 0 or y2 < 0 or x1 > calibration['board_size'] or y1 > calibration['board_size']:
                    continue
                
                # Clamp to board bounds
                x1 = max(0, min(x1, calibration['board_size'] - 1))
                y1 = max(0, min(y1, calibration['board_size'] - 1))
                x2 = max(0, min(x2, calibration['board_size'] - 1))
                y2 = max(0, min(y2, calibration['board_size'] - 1))
                
                piece_type = results[0].names[int(cls)]
                square = bbox_to_square([x1, y1, x2, y2], calibration['square_map'])
                
                if square:
                    detections[square] = {
                        'piece': piece_type,
                        'confidence': float(conf)
                    }
        
        print(f"   After filtering: {len(detections)} valid detections")
        
        # CORRECT FOR ROTATION - DISABLED FOR NOW
        # The frontend will handle rotation visually
        # if orientation == "rotated":
        #     print("   Correcting coordinates for 180° rotation...")
        #     detections = correct_detections_for_orientation(detections, orientation)
        #     print(f"   After rotation correction: {len(detections)} pieces")
        
        # Print final detections
        print(f"\n✓ Final: {len(detections)} pieces detected")
        for square in sorted(detections.keys()):
            piece_info = detections[square]
            print(f"     {square}: {piece_info['piece']} ({piece_info['confidence']:.2f})")
        
        return jsonify({
            'success': True,
            'pieces': detections,
            'count': len(detections),
            'orientation': orientation
        })
    
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/info', methods=['GET'])
def info():
    """Get system info"""
    return jsonify({
        'board_size': calibration['board_size'] if calibration else None,
        'cell_size': calibration['cell_size'] if calibration else None,
        'model': 'YOLOv8',
        'features': ['piece_detection', 'rotation_correction', 'edge_padding']
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    try:
        print("\n" + "="*60)
        print(" Chess Detection Backend (with Rotation Detection)")
        print("="*60 + "\n")
        
        load_resources()
        
        print("\n" + "="*60)
        print("Backend Ready!")
        print("="*60)
        print("\nEndpoints:")
        print("  POST http://localhost:5000/detect  - Detect pieces")
        print("  GET  http://localhost:5000/health  - Health check")
        print("  GET  http://localhost:5000/info    - System info")
        print("\n" + "="*60)
        print("Running on http://localhost:5000")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    
    except Exception as e:
        print(f"\n Startup error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)