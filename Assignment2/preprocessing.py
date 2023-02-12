#!/usr/bin/env python3
"""
Image preprocessing functions for a YOLO classifier.
"""
# Standard library imports
from pathlib import Path
from typing import List, Tuple

# Third party imports
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def bgr_to_rgb(bgr_img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to RGB.
    Args:
        bgr_img: Unsigned 8-bit blue-green-red image.
    Returns: Unsigned 8-bit red-blue-green image.
    """
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


test_bgr_img = np.array([[[166, 237, 252], [ 54, 202, 247], [164, 236, 252]], [[ 45, 192, 242], [14, 75, 129], [34, 180, 238]], [[177, 237, 250], [ 45, 194, 240], [151, 227, 247]]], dtype=np.uint8)
assert bgr_to_rgb(test_bgr_img).max() <= 255
assert bgr_to_rgb(test_bgr_img).min() >= 0
assert ~np.allclose(test_bgr_img, bgr_to_rgb(test_bgr_img))


def resize(img: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to desired width and height.
    Args:
        img: Image to resize.
        dsize: Width, height tuple.
    Returns: Resized image.
    """
    return cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)


test_rgb_img = np.array([[[252, 237, 166], [247, 202,  54], [252, 236, 164]], [[242, 192,  45], [129,  75,  14], [238, 180,  34]], [[250, 237, 177], [240, 194,  45], [247, 227, 151]]], dtype=np.uint8)
assert resize(test_rgb_img, (20, 40)).shape != (20, 40, 3)
assert resize(test_rgb_img, (20, 30)).shape == (30, 20, 3)
assert resize(test_rgb_img, (10, 10)).shape == (10, 10, 3)


def normalize(
        rgb_img: np.ndarray,
        means: List[float]=None,
        stds: List[float]=None,
    ) -> np.ndarray:
    """
    Normalize RGB image using band-weighted means and standard deviations.
    Args:
        rgb_img: Unsigned 8-bit RGB array with dimension order of height, width, channels.
        mean_wts: List of means, one for each channel.
        std_wts: List of standard deviations, one for each channel.
    Returns: Unsigned 8-bit band-centered image.
    """
    if means is None:
        means = [0.485, 0.456, 0.406]
    if stds is None:
        stds = [0.229, 0.224, 0.225]
    height, width, channels = rgb_img.shape
    norm_img = np.zeros((height, width, channels))
    for channel in range(channels):
        norm_img[:, :, channel] = (rgb_img[:, :, channel] - means[channel]) / stds[channel]
    return cv2.normalize(norm_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)


test_rgb_img = np.array([[[252, 237, 166], [247, 202,  54], [252, 236, 164]], [[242, 192,  45], [129,  75,  14], [238, 180,  34]], [[250, 237, 177], [240, 194,  45], [247, 227, 151]]], dtype=np.uint8)
assert normalize(test_rgb_img).dtype == np.uint8
assert ~np.allclose(normalize(test_rgb_img), normalize(test_rgb_img, means=[0.2, 0.9, 0.1]))
assert ~np.allclose(normalize(test_rgb_img), normalize(test_rgb_img, stds=[0.4, 0.2, 0.9]))



def pca(rgb_img: np.ndarray) -> np.ndarray:
    """
    Reduce dimensionality of multiband image (e.g., convert RGB to grayscale).
    Args:
        rgb_img: Unsigned 8-bit RGB image to perform PCA on.
    Returns: 64-bit float gray-scale image.
    """
    height, width, channels = rgb_img.shape
    scaler_model = StandardScaler()
    pca_model = PCA(n_components=1)
    X = scaler_model.fit_transform(rgb_img.reshape((height * width, channels), order="F"))
    X_pca = pca_model.fit_transform(X)
    return X_pca.reshape((height, width), order="F")


test_rgb_img = np.array([[[252, 237, 166], [247, 202,  54], [252, 236, 164]], [[242, 192,  45], [129,  75,  14], [238, 180,  34]], [[250, 237, 177], [240, 194,  45], [247, 227, 151]]], dtype=np.uint8)
assert pca(test_rgb_img).dtype == np.float64
assert ~np.allclose(pca(test_rgb_img), test_rgb_img)


def make_grayscale(rgb_img: np.ndarray, method: str="cv2") -> np.ndarray:
    """
    Convert RGB image into grayscale using standard open-cv approach, PCA, or simple mean.
    Args:
        rgb_img: Unsigned 8-bit RGB image to convert to grayscale.
        method: 'cv2', 'pca', or 'simple'.
    Returns: Unsigned 8-bit grayscale image.
    """
    assert method in ("cv2", "pca", "simple"), "Grayscale method must be 'cv2', 'pca', or 'simple'."
    if method == "cv2":
        return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    elif method == "pca":
        scaler = StandardScaler()
        pca = PCA(n_components=1, random_state=777)
        height, width, channels = rgb_img.shape
        X = scaler.fit_transform(rgb_img.reshape((height * width, channels), order="F"))
        X_pca = pca.fit_transform(X)
        gray_img = X_pca.reshape((height, width), order="F")
    else:
        gray_img = rgb_img.mean(axis=2)
    return cv2.normalize(gray_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)#(255 * (gray_img - np.min(gray_img)) / np.ptp(gray_img)).astype(np.uint8)


test_rgb_img = np.array([[[252, 237, 166], [247, 202,  54], [252, 236, 164]], [[242, 192,  45], [129,  75,  14], [238, 180,  34]], [[250, 237, 177], [240, 194,  45], [247, 227, 151]]], dtype=np.uint8)
assert ~np.allclose(make_grayscale(test_rgb_img), make_grayscale(test_rgb_img, method="pca"))
assert ~np.allclose(make_grayscale(test_rgb_img), make_grayscale(test_rgb_img, method="simple"))
assert ~np.allclose(make_grayscale(test_rgb_img, "pca"), make_grayscale(test_rgb_img, method="simple"))


def plot(ims: List[np.ndarray], im_titles: List[str], cmaps: List[str], figsize: Tuple[int, int]=(5, 5)):
    """
    Plot titled images in a single row.
    Args:
        ims: List of images to plot.
        im_titles: List of image titles.
        cmaps: List of color maps.
        figsize: Height and width figure dimensions.
    Adapted from https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html.
    """
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, len(ims)),  # creates 2x2 grid of axes
                     axes_pad=0.4,  # pad between axes in inch.
                     )
    for ix, (ax, im) in enumerate(zip(grid, ims)):
        # Iterating over the grid returns the Axes.
        ax.axis("off")
        ax.title.set_text(im_titles[ix])
        ax.imshow(im, cmap=cmaps[ix])
    plt.show()
