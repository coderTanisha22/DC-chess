"""
Utility functions:
- read image (basic)
- warp & split board into tiles
"""
import os
import cv2
import numpy as np
from PIL import Image

def read_image(path):
    return cv2.imread(path)

def order_points_clockwise(pts):
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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

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
