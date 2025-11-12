# inference/predict_board_state.py
import cv2, numpy as np, os
from tensorflow.keras.models import load_model
from preprocessing.utils import read_image, detect_board_and_warp, split_board_into_tiles
import tensorflow as tf

MODEL_PATH = "../models/trained_models/piece_classifier.h5"
LABELS_PATH = "../models/trained_models/label_encoder_classes.npy"  # if you saved mapping; otherwise get from flow.class_indices

# simple mapping for FEN
map_to_fen = {
 'white_king':'K','white_queen':'Q','white_rook':'R','white_bishop':'B','white_knight':'N','white_pawn':'P',
 'black_king':'k','black_queen':'q','black_rook':'r','black_bishop':'b','black_knight':'n','black_pawn':'p','empty':'1'
}

def board_to_fen(board8x8):
    fen_rows = []
    for r in range(8):
        row = ""
        empty_count = 0
        for c in range(8):
            ch = board8x8[r][c]
            if ch == 'empty':
                empty_count += 1
            else:
                if empty_count:
                    row += str(empty_count); empty_count = 0
                row += map_to_fen.get(ch, '?')
        if empty_count: row += str(empty_count)
        fen_rows.append(row)
    return "/".join(fen_rows)

def predict_on_image(img_path, model, class_names, tile_size=128, show_overlay=True):
    img = read_image(img_path)
    warped = detect_board_and_warp(img)
    if warped is None:
        print("Board not detected.")
        return None
    # split to tiles in memory
    s = warped.shape[0]
    cell = s // 8
    preds = []
    for r in range(8):
        row_preds = []
        for c in range(8):
            y1, y2 = r*cell, (r+1)*cell
            x1, x2 = c*cell, (c+1)*cell
            tile = cv2.resize(warped[y1:y2, x1:x2], (tile_size,tile_size))
            x = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype('float32')/255.0
            x = np.expand_dims(x, axis=0)
            p = model.predict(x)
            idx = int(np.argmax(p, axis=1)[0])
            cls = class_names[idx]
            row_preds.append(cls)
        preds.append(row_preds)
    fen = board_to_fen(preds)
    print("FEN:", fen)
    if show_overlay:
        overlay = warped.copy()
        for r in range(8):
            for c in range(8):
                cx = int((c+0.5)*cell)
                cy = int((r+0.5)*cell)
                cv2.putText(overlay, preds[r][c].split('_')[-1][0].upper(), (cx-10, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("pred_overlay", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return preds, fen

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    # If you saved class_names as an array, load it; otherwise provide in order from train flow.class_indices
    # class_names = np.load(LABELS_PATH)  # optional
    # Example: if not saved, define manually:
    class_names = ['white_king','white_queen','white_rook','white_bishop','white_knight','white_pawn',
                'black_king','black_queen','black_rook','black_bishop','black_knight','black_pawn','empty']
    predict_on_image("../data/full_boards/board1.jpg", model, class_names)
