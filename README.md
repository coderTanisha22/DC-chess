# DC-Chess: Real Board Chess Position and Move Detection

DC-Chess is a computer vision project that converts photos of a physical chessboard into structured digital board states and move histories. The system combines perspective calibration, YOLOv8 piece detection, board-square mapping, and a web-based sequence analyzer to track game progress from camera images.

This project is designed as an applied AI/ML engineering portfolio project with practical deployment components (Flask/FastAPI APIs + browser UI) and model training/evaluation workflows.

## Project Highlights

- End-to-end pipeline from raw board image to square-level piece positions.
- Perspective calibration to normalize angled board photos.
- YOLOv8-based multi-class piece detection (12 chess piece classes).
- Square mapping (`a1` to `h8`) from bounding box centers.
- Sequential move analysis across multiple uploaded board snapshots.
- API-first architecture with health endpoints and JSON outputs.
- Experimental dual approach:
	- Detector-first workflow (YOLO -> board squares)
	- Tile-classifier workflow (warp -> 64 tiles -> classifier -> FEN)

## Problem Statement

Given one or more images of a real chessboard, detect all visible pieces, map them to legal board coordinates, and infer move transitions between successive board states.

## System Architecture

### 1) Board Calibration and Geometry

- `board_calibration.py`
	- Manual corner selection of board region.
	- Builds perspective transform matrix.
	- Generates `square_map` for all 64 cells.
	- Saves calibration artifacts to `calibration.pkl`.

### 2) Piece Detection Core

- `backend.py`
	- Flask backend with `/detect`, `/health`, and `/info` endpoints.
	- Loads calibration + YOLO model.
	- Applies warp transform.
	- Uses edge padding to improve border-piece recall.
	- Converts detections to chess coordinates.
- `board_rotation_detector.py`
	- Orientation heuristics (brightness/quadrant based).
	- Coordinate-flip utilities for 180-degree board orientation.

### 3) Move Detection Logic

- `static_move_detector.py`
	- Compares before/after board states.
	- Handles simple move, capture, castling, and promotion patterns.
- `simple_move_test.py`
	- Lightweight debugging and move detection test harness.

### 4) Training and Evaluation

- `retrain_model.py`
	- Retrains YOLOv8 on Roboflow-exported dataset config.
- `analysis/plot_pr_curve.py`
	- Generates per-class precision-recall curves and reports mAP.
- `check_training.py`, `test_model_raw.py`, `debug_detection.py`
	- Diagnostic scripts for model quality and pipeline behavior.

### 5) Alternative Tile Classification Path (Experimental)

- `classifier/train_classifier.py`
	- MobileNetV2-based tile classifier (TensorFlow/Keras).
- `inference/full_pipeline.py`
	- Runs tile-level inference and exports FEN.
- `preprocessing/utils.py`
	- Shared helpers for image loading, warping, and tile splitting.
- `web/api.py`
	- FastAPI endpoint for warped-image prediction.

### 6) Frontend

- `frontend.html`
	- Interactive 5-photo sequential analysis UI.
	- Upload, detect, visualize board, and view move history.

## Repository Structure

```text
DC-chess/
├── backend.py                     # Flask inference API (main detector backend)
├── board_calibration.py           # Manual calibration and square-map generation
├── board_rotation_detector.py     # Orientation detection + coordinate flipping
├── static_move_detector.py        # Full move-comparison engine
├── simple_move_test.py            # Minimal move-test script
├── retrain_model.py               # YOLO retraining script
├── debug_detection.py             # Pipeline debugging script
├── test_model_raw.py              # Raw-image detector sanity test
├── check_training.py              # Training artifact/metric checker
├── check_images.py                # Dataset image sanity checks
├── frontend.html                  # Sequential web UI
├── requirements.txt               # Python dependencies
├── analysis/
│   └── plot_pr_curve.py           # PR curve + mAP analysis
├── classifier/
│   └── train_classifier.py        # MobileNetV2 tile classifier trainer
├── detection/
│   ├── detect_and_crop.py         # YOLO detection and crop export
│   └── test_detector.py           # Detector test/validation script
├── inference/
│   └── full_pipeline.py           # Warped-board -> FEN pipeline
├── preprocessing/
│   └── utils.py                   # Shared image and geometry utilities
└── web/
		└── api.py                     # Optional FastAPI wrapper
```

## Tech Stack

- Python, OpenCV, NumPy
- YOLOv8 (Ultralytics), PyTorch
- TensorFlow/Keras (MobileNetV2 classifier path)
- Flask, FastAPI, Uvicorn
- HTML/CSS/JavaScript frontend

## Setup

```bash
git clone <your-repo-url>
cd DC-chess
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical Workflow

### 1) Prepare model and calibration

1. Place/prepare YOLO weights at `runs/detect/train/weights/best.pt`.
2. Run calibration to generate `calibration.pkl`:

```bash
python board_calibration.py
```

### 2) Start backend API

```bash
python backend.py
```

### 3) Use frontend

1. Open `frontend.html` in a browser.
2. Upload chessboard images in sequence.
3. Click detect and inspect piece list + move history.

### 4) Retrain detector (optional)

```bash
python retrain_model.py
```

## API Endpoints (Flask)

- `GET /health`: service/model/calibration readiness.
- `GET /info`: board and feature metadata.
- `POST /detect`: multipart image upload, returns detected pieces and count.

Example response shape:

```json
{
	"success": true,
	"pieces": {
		"e2": {"piece": "white_pawn", "confidence": 0.91}
	},
	"count": 27,
	"orientation": "standard"
}

## Future Enhancements

- Add live camera streaming support (USB/IP camera) so the system can process continuous frames instead of manual image uploads.
- Use WebSocket streaming between frontend and backend for near real-time board updates and lower latency than repeated HTTP uploads.
- Add frame selection and stabilization (process every Nth frame plus motion filtering) to improve detection consistency during hand movement.
- Introduce move-confirmation logic across consecutive frames to reduce false move triggers and improve game-state reliability.