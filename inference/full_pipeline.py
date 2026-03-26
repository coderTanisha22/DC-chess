"""
End-to-end inference (Option A):
- Warp board (8x8)
- Split into 64 tiles
- Classify each tile using a trained classifier model
Produces an 8x8 board array + FEN string.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.utils import split_board_into_tiles, read_image


MODEL_PATH = "models/classifier/piece_classifier.h5"
CLASS_NAMES_PATH = "models/classifier/class_names.npy"


# -------------------------------------------------------------
# FEN Conversion
# -------------------------------------------------------------
def board_to_fen(board):
    """Convert 8×8 piece-name board into FEN string."""
    mapping = {
        "white_king": "K", "white_queen": "Q", "white_rook": "R",
        "white_bishop": "B", "white_knight": "N", "white_pawn": "P",

        "black_king": "k", "black_queen": "q", "black_rook": "r",
        "black_bishop": "b", "black_knight": "n", "black_pawn": "p",

        "empty": "1"
    }

    fen_rows = []

    for r in range(8):
        row_str = ""
        empty_count = 0

        for c in range(8):
            piece = board[r][c]

            if piece == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += mapping.get(piece, "?")

        # If last squares were empty
        if empty_count > 0:
            row_str += str(empty_count)

        fen_rows.append(row_str)

    return "/".join(fen_rows)


# -------------------------------------------------------------
# Inference on Warped Board
# -------------------------------------------------------------
def predict_from_warped(warped_path):
    """Load a warped 8x8 board image and classify each tile."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Classifier model missing: {MODEL_PATH}")

    # Load classifier + class names
    model = load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)

    # Load warped input
    warped = cv2.imread(warped_path)
    if warped is None:
        raise ValueError(f"Failed to load warped board image: {warped_path}")

    # Optional: preview tiles
    split_board_into_tiles(
        warped,
        out_dir="/tmp/tiles_preview",
        tile_size=128,
        basename="tile"
    )

    # Tile dimensions
    H, W = warped.shape[:2]
    cell_size = H // 8  # assumes perfect square warp

    board = [["empty"] * 8 for _ in range(8)]

    # Classify each cell
    for r in range(8):
        for c in range(8):
            tile = warped[
                r * cell_size : (r + 1) * cell_size,
                c * cell_size : (c + 1) * cell_size
            ]

            tile_resized = cv2.resize(tile, (128, 128))
            rgb = cv2.cvtColor(tile_resized, cv2.COLOR_BGR2RGB)
            norm = rgb.astype("float32") / 255.0

            pred = model.predict(np.expand_dims(norm, 0), verbose=False)
            idx = int(np.argmax(pred))
            board[r][c] = class_names[idx]

    fen = board_to_fen(board)
    return board, fen


# -------------------------------------------------------------
# Main Script
# -------------------------------------------------------------
if __name__ == "__main__":
    warped_example = "data/processed/IMG_1336_warped_fast.jpg"

    if not os.path.exists(warped_example):
        print(f"Missing warped board image at: {warped_example}")
    else:
        board, fen = predict_from_warped(warped_example)
        print("FEN:", fen)
