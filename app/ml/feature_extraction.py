"""Feature extraction from preprocessed handwriting images.

Combines HOG descriptors, contour geometry, projection profiles,
morphological statistics, and letter-spacing metrics into a single
feature vector suitable for classical ML classifiers.
"""

from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage


def extract_hog_features(
    image: np.ndarray,
    cell_size: Tuple[int, int] = (8, 8),
    block_size: Tuple[int, int] = (2, 2),
    nbins: int = 9,
) -> np.ndarray:
    """Compute HOG descriptor for an image.

    Args:
        image: Grayscale or binary image.
        cell_size: Size of each cell in pixels.
        block_size: Number of cells per block.
        nbins: Number of orientation bins.

    Returns:
        Flattened HOG feature vector.
    """
    if image is None or image.size == 0:
        return np.zeros(1, dtype=np.float32)

    win_size = image.shape[:2][::-1]
    hog = cv2.HOGDescriptor(
        win_size, block_size, cell_size, cell_size, nbins
    )
    descriptor = hog.compute(image)
    if descriptor is None:
        return np.zeros(1, dtype=np.float32)
    return descriptor.flatten()


def extract_contour_features(image: np.ndarray) -> np.ndarray:
    """Extract geometric features from contours.

    Features: mean aspect ratio, area variance, total convex-hull
    defect count, and mean perimeter-to-area ratio across contours.

    Args:
        image: Binary image.

    Returns:
        numpy array of shape ``(4,)``.
    """
    defaults = np.zeros(4, dtype=np.float64)
    if image is None or image.size == 0:
        return defaults

    binary = image if image.max() <= 1 else (image // 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return defaults

    aspect_ratios: List[float] = []
    areas: List[float] = []
    defect_count = 0
    par_list: List[float] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / max(h, 1)
        aspect_ratios.append(ar)

        area = cv2.contourArea(cnt)
        areas.append(area)

        hull = cv2.convexHull(cnt, returnPoints=False)
        if len(cnt) > 3 and len(hull) > 3:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                defect_count += defects.shape[0]

        perimeter = cv2.arcLength(cnt, True)
        par = perimeter / max(area, 1e-6)
        par_list.append(par)

    return np.array(
        [
            float(np.mean(aspect_ratios)),
            float(np.var(areas)),
            float(defect_count),
            float(np.mean(par_list)),
        ],
        dtype=np.float64,
    )


def extract_projection_features(image: np.ndarray) -> np.ndarray:
    """Compute horizontal and vertical projection profiles.

    Returns statistics: horizontal mean/std/max, vertical mean/std/max
    of the per-row / per-column ink pixel sums.

    Args:
        image: Binary image.

    Returns:
        numpy array of shape ``(6,)``.
    """
    defaults = np.zeros(6, dtype=np.float64)
    if image is None or image.size == 0:
        return defaults

    binary = (image > 0).astype(np.float64)
    h_proj = binary.sum(axis=1)
    v_proj = binary.sum(axis=0)

    return np.array(
        [
            float(h_proj.mean()),
            float(h_proj.std()),
            float(h_proj.max()),
            float(v_proj.mean()),
            float(v_proj.std()),
            float(v_proj.max()),
        ],
        dtype=np.float64,
    )


def extract_morphological_features(image: np.ndarray) -> np.ndarray:
    """Extract ink density, Hu moments, and stroke-width variance.

    Args:
        image: Binary image.

    Returns:
        numpy array of shape ``(9,)``  (1 + 7 + 1).
    """
    defaults = np.zeros(9, dtype=np.float64)
    if image is None or image.size == 0:
        return defaults

    binary = (image > 0).astype(np.float64)

    # Ink density
    ink_density = binary.mean()

    # Hu moments (log-transformed for stability)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    # Stroke-width variance via distance transform
    dist = ndimage.distance_transform_edt(binary > 0)
    sw_var = float(dist.std()) if dist.any() else 0.0

    return np.concatenate([[ink_density], hu_log, [sw_var]]).astype(np.float64)


def extract_letter_spacing_features(image: np.ndarray) -> np.ndarray:
    """Estimate inter-letter spacing and baseline deviation.

    Uses connected components to measure horizontal gaps and vertical
    baseline jitter.

    Args:
        image: Binary image.

    Returns:
        numpy array of shape ``(4,)``: mean spacing, std spacing,
        baseline std, baseline range.
    """
    defaults = np.zeros(4, dtype=np.float64)
    if image is None or image.size == 0:
        return defaults

    binary = (image > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return defaults

    # Filter out the background label (0)
    x_coords = stats[1:, cv2.CC_STAT_LEFT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    centroids_y = centroids[1:, 1]

    sorted_idx = np.argsort(x_coords)
    x_sorted = x_coords[sorted_idx]
    w_sorted = widths[sorted_idx]

    # Inter-letter gaps: distance between end of component i and start of i+1
    if len(x_sorted) > 1:
        gaps = (x_sorted[1:] - (x_sorted[:-1] + w_sorted[:-1])).astype(np.float64)
        gaps = gaps[gaps >= 0]
        mean_gap = float(gaps.mean()) if len(gaps) else 0.0
        std_gap = float(gaps.std()) if len(gaps) else 0.0
    else:
        mean_gap = 0.0
        std_gap = 0.0

    # Baseline deviation
    y_sorted = centroids_y[sorted_idx]
    baseline_std = float(y_sorted.std()) if len(y_sorted) > 1 else 0.0
    baseline_range = float(y_sorted.max() - y_sorted.min()) if len(y_sorted) > 1 else 0.0

    return np.array([mean_gap, std_gap, baseline_std, baseline_range], dtype=np.float64)


def extract_all_features(
    image: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Combine every feature extractor into a single vector.

    Args:
        image: Preprocessed binary image.

    Returns:
        ``(feature_vector, feature_names)`` where *feature_vector* is a
        1-D numpy array and *feature_names* is a list of matching labels.
    """
    hog = extract_hog_features(image)
    contour = extract_contour_features(image)
    projection = extract_projection_features(image)
    morph = extract_morphological_features(image)
    spacing = extract_letter_spacing_features(image)

    feature_vector = np.concatenate([hog, contour, projection, morph, spacing])

    names: List[str] = [f"hog_{i}" for i in range(len(hog))] + [
        "contour_aspect_ratio_mean",
        "contour_area_variance",
        "contour_convex_defects",
        "contour_perimeter_area_ratio",
        "proj_h_mean",
        "proj_h_std",
        "proj_h_max",
        "proj_v_mean",
        "proj_v_std",
        "proj_v_max",
        "morph_ink_density",
        *["morph_hu_moment_" + str(i) for i in range(1, 8)],
        "morph_stroke_width_var",
        "spacing_mean_gap",
        "spacing_std_gap",
        "spacing_baseline_std",
        "spacing_baseline_range",
    ]

    return feature_vector, names
