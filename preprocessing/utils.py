"""
Utility functions:
- Read image
- Four-point perspective warp
- Split warped chessboard into 8×8 tiles
"""

import os
import cv2
import numpy as np
from PIL import Image


# -------------------------------------------------------------
# Basic Image Loader
# -------------------------------------------------------------
def read_image(path):
    """Read an image from disk (BGR format)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


# -------------------------------------------------------------
# Perspective Transform Helpers
# -------------------------------------------------------------
def order_points_clockwise(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Input shape: (4,2).
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect


def four_point_transform(image, pts):
    """
    Apply perspective warp using 4 corner points.
    Returns a top-down, rectangular warped image.
    """
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect

    # Compute new image size
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# -------------------------------------------------------------
# Board Splitter
# -------------------------------------------------------------
def split_board_into_tiles(warped, out_dir, tile_size=128, basename="board"):
    """
    Split a square warped board image into 64 tiles.
    Saves tiles to out_dir and also returns out_dir.
    """
    if warped is None:
        raise ValueError("Warped board image is None.")

    os.makedirs(out_dir, exist_ok=True)

    H, W = warped.shape[:2]
    cell = H // 8  # assumes perfect square warp

    for r in range(8):
        for c in range(8):
            y1, y2 = r * cell, (r + 1) * cell
            x1, x2 = c * cell, (c + 1) * cell

            tile = warped[y1:y2, x1:x2]

            # Resize for classifier
            tile_resized = cv2.resize(tile, (tile_size, tile_size))

            out_path = os.path.join(out_dir, f"{basename}_r{r}c{c}.jpg")
            cv2.imwrite(out_path, tile_resized)

    return out_dir
