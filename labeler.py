# labeler.py
import os, shutil
from PIL import Image
import csv

UNLABELED = "data/tiles_unlabeled"
LABELED_DIR = "data/tiles_labeled"
CSV_OUT = "data/tiles_labels.csv"

CLASSES = ["white_king","white_queen","white_rook","white_bishop","white_knight","white_pawn",
           "black_king","black_queen","black_rook","black_bishop","black_knight","black_pawn","empty","unknown"]

def make_dirs():
    os.makedirs(LABELED_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(LABELED_DIR, c), exist_ok=True)

def run_labeler():
    make_dirs()
    files = sorted(os.listdir(UNLABELED))
    labels = []
    for f in files:
        path = os.path.join(UNLABELED, f)
        img = Image.open(path)
        img.show()
        lab = input(f"Label for {f} (or 'skip'): ")
        if lab.lower() == 'skip':
            continue
        if lab not in CLASSES:
            print("Unknown class. choose from:", CLASSES)
            continue
        dst = os.path.join(LABELED_DIR, lab, f)
        shutil.move(path, dst)
        labels.append((f, lab))
    # save csv
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename","label"])
        writer.writerows(labels)
    print("Labeling finished. CSV saved to", CSV_OUT)

if __name__ == "__main__":
    run_labeler()
