"""
End-to-end inference sketch:
- Option A: warp board -> split into 64 tiles -> classify each tile using classifier model
- Option B: run detector -> map boxes to board cells (requires consistent warp mapping)
This script demonstrates Option A (grid-split approach).
"""
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.utils import split_board_into_tiles, read_image

MODEL_PATH = "models/classifier/piece_classifier.h5"
CLASS_NAMES_PATH = "models/classifier/class_names.npy"

def board_to_fen(board8x8):
    mapping = {
        'white_king':'K','white_queen':'Q','white_rook':'R','white_bishop':'B','white_knight':'N','white_pawn':'P',
        'black_king':'k','black_queen':'q','black_rook':'r','black_bishop':'b','black_knight':'n','black_pawn':'p','empty':'1'
    }
    rows=[]
    for r in range(8):
        row=""
        empty_cnt=0
        for c in range(8):
            ch = board8x8[r][c]
            if ch=='empty':
                empty_cnt+=1
            else:
                if empty_cnt:
                    row+=str(empty_cnt); empty_cnt=0
                row+=mapping.get(ch, '?')
        if empty_cnt:
            row+=str(empty_cnt)
        rows.append(row)
    return "/".join(rows)

def predict_from_warped(warped_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Classifier model not found. Train first.")
    model = load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    warped = cv2.imread(warped_path)
    tiles = split_board_into_tiles(warped, out_dir="/tmp/tiles_preview", tile_size=128, basename="tmp")
    board = [['empty']*8 for _ in range(8)]
    s = warped.shape[0]
    cell = s // 8
    for r in range(8):
        for c in range(8):
            tile = cv2.resize(warped[r*cell:(r+1)*cell, c*cell:(c+1)*cell], (128,128))
            x = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype('float32')/255.0
            p = model.predict(np.expand_dims(x,0))
            idx = int(np.argmax(p, axis=1)[0])
            board[r][c] = class_names[idx]
    fen = board_to_fen(board)
    return board, fen

if __name__ == "__main__":
    warped_example = "data/processed/IMG_1336_warped_fast.jpg"
    if not os.path.exists(warped_example):
        print("Place a warped board at", warped_example)
    else:
        board, fen = predict_from_warped(warped_example)
        print("FEN:", fen)
