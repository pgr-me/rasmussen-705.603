#!/usr/bin/env python3
"""
Preprocess images for a YOLO classifier.
"""
# Standard library imports
import argparse
from pathlib import Path
from typing import List, Tuple

# Third party imports
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Local imports
from preprocessing import bgr_to_rgb, resize, normalize, pca, make_grayscale


SRC_DIR = Path("/workspace/rasmussen-705.603/data/Assignment2/raw")
DST_DIR = Path("/workspace/rasmussen-705.603/data/Assignment2/processed")
DSIZE = (300, 400)

def parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--src_dir", type=Path, default=SRC_DIR)
    parser.add_argument("-o", "--dst_dir", type=Path, default=DST_DIR)
    parser.add_argument("-d", "--dsize", nargs="+", type=int, default=DSIZE)
    return parser.parse_args()


def main(src_dir: Path=SRC_DIR, dst_dir: Path=DST_DIR, dsize: Tuple[int, int]=DSIZE):
    """
    Read a BGR image, convert to RGB, resize it, normalize it, convert to grayscale, then plot each image.
    Args:
        src_dir: Directory of input images.
        dst_dir: Directory for output images.
        width: Target resizing width.
    """
    print(f"Iterate over each image in {src_dir}.")
    for im_src in [x for x in src_dir.iterdir() if x.suffix in [".jpg", ".png"]]:
        print(80 * "~")
        print(f"Process {im_src.name}")

        # Load image
        bgr_img = cv2.imread(str(im_src))
        # Convert BGR to RGB
        rgb_img = bgr_to_rgb(bgr_img)
        # Resize RGB image to desired size
        resized_img = resize(rgb_img, dsize)
        # Normalize resized RGB image
        normed_img = normalize(resized_img)
        # Grayscale normalized image
        gray_img = make_grayscale(normed_img, method='cv2')
        # Save outputs
        ims_dict = dict(zip(["bgr", "rgb", "resized", "normed", "gray"], [bgr_img, rgb_img, resized_img, normed_img, gray_img]))
        for img_name, img in ims_dict.items():
            dst = str(dst_dir / f"{im_src.stem}_{img_name}.png")
            print(f"Save {img_name} to {dst}.")
            cv2.imwrite(dst, img)
        print("")

if __name__ == "__main__":
    args = parser()
    main(args.src_dir, args.dst_dir, args.dsize)
