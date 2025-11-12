# preprocessing/utils.py
import cv2
import numpy as np
import os
from PIL import Image
import pillow_heif  # required to read HEIC via pillow-heif (if installed)

def read_image(path):
    # uses cv2 for most, pillow_heif for HEIC when pillow can't read
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.heic', '.heif']:
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(
            "RGB", (heif_file.size[0], heif_file.size[1]), heif_file.data, "raw"
        )
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return arr
    else:
        return cv2.imread(path)

def order_points_clockwise(pts):
    # pts: (4,2)
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_board_and_warp(img, debug=False):
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive blur + edge detection
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    # close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_cnt = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            board_cnt = approx.reshape(4,2)
            break
    if board_cnt is None:
        # fallback: try threshold contour
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                board_cnt = approx.reshape(4,2)
                break
    if board_cnt is None:
        return None  # board not found
    warped = four_point_transform(orig, board_cnt)
    # make square by resizing to max dimension and then cropping/padding
    h,w = warped.shape[:2]
    size = max(h,w)
    square = cv2.resize(warped, (size,size))
    return square

def split_board_into_tiles(warped, out_dir, tile_size=128, basename="board"):
    os.makedirs(out_dir, exist_ok=True)
    s = warped.shape[0]
    cell = s // 8
    for r in range(8):
        for c in range(8):
            y1, y2 = r*cell, (r+1)*cell
            x1, x2 = c*cell, (c+1)*cell
            tile = cv2.resize(warped[y1:y2, x1:x2], (tile_size, tile_size))
            fname = os.path.join(out_dir, f"{basename}_r{r}c{c}.jpg")
            cv2.imwrite(fname, tile)
    return out_dir

def process_folder(in_folder, out_folder, tile_size=128):
    os.makedirs(out_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(in_folder) if f.lower().endswith(('.jpg','.jpeg','.png','.heic','.heif'))])
    for f in files:
        p = os.path.join(in_folder, f)
        img = read_image(p)
        warped = detect_board_and_warp(img)
        if warped is None:
            print("Board not found:", f)
            continue
        base = os.path.splitext(f)[0]
        split_board_into_tiles(warped, out_folder, tile_size=tile_size, basename=base)
    print("Done processing folder:", in_folder)
