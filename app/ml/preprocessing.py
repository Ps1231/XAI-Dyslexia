"""Image preprocessing module for handwriting analysis.

Provides utilities for loading, cleaning, and normalizing handwritten
images before feature extraction.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from disk.

    Args:
        path: Filesystem path to the image.

    Returns:
        BGR numpy array or None if loading fails.
    """
    if not path:
        return None
    image = cv2.imread(path)
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Args:
        image: Input BGR image.

    Returns:
        Single-channel grayscale image.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Binarize a grayscale image.

    Args:
        image: Single-channel grayscale image.
        method: ``'otsu'`` for global Otsu threshold, ``'adaptive'``
            for adaptive Gaussian thresholding.

    Returns:
        Binary image (0 or 255).
    """
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def denoise(
    image: np.ndarray, method: str = "gaussian", ksize: int = 3
) -> np.ndarray:
    """Remove noise from a grayscale or binary image.

    Args:
        image: Input image.
        method: ``'gaussian'``, ``'median'``, or ``'morphological'``.
        ksize: Kernel size (must be odd).

    Returns:
        Denoised image.
    """
    ksize = max(3, ksize | 1)

    if method == "median":
        return cv2.medianBlur(image, ksize)

    if method == "morphological":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closing

    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct rotation skew using image moments.

    Args:
        image: Binary or grayscale image.

    Returns:
        Deskewed image of the same size.
    """
    if len(image.shape) == 3:
        gray = to_grayscale(image)
    else:
        gray = image

    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def resize_and_pad(
    image: np.ndarray, target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """Resize an image preserving its aspect ratio and pad to target size.

    Args:
        image: Input image.
        target_size: ``(width, height)`` of the output canvas.

    Returns:
        Resized and centered image.
    """
    h, w = image.shape[:2]
    tw, th = target_size

    scale = min(tw / max(w, 1), th / max(h, 1))
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((th, tw), dtype=resized.dtype)
    y_off = (th - new_h) // 2
    x_off = (tw - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Steps: grayscale -> denoise -> binarize -> deskew -> resize_and_pad.

    Args:
        image: Raw BGR or grayscale image.

    Returns:
        Preprocessed binary image of size (128, 128).
    """
    gray = to_grayscale(image)
    denoised = denoise(gray, method="gaussian")
    binary = binarize(denoised, method="otsu")
    deskewed = deskew(binary)
    resized = resize_and_pad(deskewed)
    return resized
