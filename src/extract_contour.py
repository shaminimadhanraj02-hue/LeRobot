import cv2
import numpy as np
from pathlib import Path

raw_image_path = "data/raw/bolt_nut.png"     #Taking the original image 
output_path = "data/processed/bn_contour.npy"

def main():
    print(f"Reading image from: {raw_image_path}")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
#Converting the raw image to grayscale image
    img = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {raw_image_path}")

    print("Image loaded, shape:", img.shape)

    # threshold to binary
    #if pixel < 127 → 0 (black)
    #if pixel > 127 → 255 (white)
    _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of contours found:", len(contours))
    if len(contours) == 0:
        raise RuntimeError("No contours found in image")

    # largest contour
    cnt = max(contours, key=cv2.contourArea)

    # subsample ~300 points
    step = max(1, len(cnt) // 300)
    pts = cnt[::step, 0, :]   # shape (M,2)

    # normalize to [0,1]
    h, w = img.shape
    pts_norm = np.zeros_like(pts, dtype=np.float32)
    pts_norm[:, 0] = pts[:, 0] / w
    pts_norm[:, 1] = pts[:, 1] / h

    # save
    np.save(output_path, pts_norm)
    print(f"Saved {len(pts_norm)} normalized contour points to {output_path}")

if __name__ == "__main__":
    main()
