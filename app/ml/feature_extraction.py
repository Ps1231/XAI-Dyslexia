"""Feature extraction from preprocessed handwriting images.

Combines HOG descriptors, contour geometry, projection profiles,
morphological statistics, and letter-spacing metrics into a single
feature vector suitable for classical ML classifiers.

NOTE: Input images are expected to already be standardized to a fixed
size (e.g. 128x128) by the preprocessing pipeline. No additional
resizing is performed here.
"""

from typing import Any, List, Tuple, Optional

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
        image: Grayscale or binary image (already size-normalized).
        cell_size: Size of each cell in pixels.
        block_size: Number of cells per block.
        nbins: Number of orientation bins.

    Returns:
        Flattened HOG feature vector.
    """
    if image is None or image.size == 0:
        return np.zeros(1, dtype=np.float32)

    h, w = image.shape[:2]
    block_size_px = (block_size[0] * cell_size[0], block_size[1] * cell_size[1])

    win_w = max((w // cell_size[0]) * cell_size[0], block_size_px[0])
    win_h = max((h // cell_size[1]) * cell_size[1], block_size_px[1])
    win_size = (win_w, win_h)

    if win_w != w or win_h != h:
        image = cv2.resize(image, (win_w, win_h))

    block_stride = cell_size
    hog = cv2.HOGDescriptor(win_size, block_size_px, block_stride, cell_size, nbins)
    descriptor = hog.compute(image)
    if descriptor is None:
        return np.zeros(1, dtype=np.float32)
    return descriptor.flatten()


def extract_contour_features(image: np.ndarray) -> np.ndarray:
    """Extract geometric features from contours."""
    defaults = np.zeros(4, dtype=np.float64)
    if image is None or image.size == 0:
        return defaults

    binary = image if image.max() <= 1 else (image // 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return defaults

    aspect_ratios = []
    areas = []
    defect_count = 0
    par_list = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / max(h, 1)
        aspect_ratios.append(ar)

        area = cv2.contourArea(cnt)
        areas.append(area)

        hull = cv2.convexHull(cnt, returnPoints=False)
        if len(cnt) > 3 and len(hull) > 3:
            try:
                defects = cv2.convexityDefects(cnt, hull)
                if defects is not None:
                    defect_count += defects.shape[0]
            except cv2.error:
                pass

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
    """Compute horizontal and vertical projection profiles."""
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
    """Extract ink density, Hu moments, and stroke-width variance."""
    defaults = np.zeros(9, dtype=np.float64)
    if image is None or image.size == 0:
        return defaults

    binary = (image > 0).astype(np.float64)
    ink_density = binary.mean()

    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    dist = ndimage.distance_transform_edt(binary > 0)
    sw_var = float(dist.std()) if dist.any() else 0.0

    return np.concatenate([[ink_density], hu_log, [sw_var]]).astype(np.float64)


def extract_letter_spacing_features(image: np.ndarray) -> np.ndarray:
    """Estimate inter-letter spacing and baseline deviation."""
    defaults = np.zeros(4, dtype=np.float64)
    if image is None or image.size == 0:
        return defaults

    binary = (image > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return defaults

    x_coords = stats[1:, cv2.CC_STAT_LEFT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    centroids_y = centroids[1:, 1]

    sorted_idx = np.argsort(x_coords)
    x_sorted = x_coords[sorted_idx]
    w_sorted = widths[sorted_idx]

    if len(x_sorted) > 1:
        gaps = (x_sorted[1:] - (x_sorted[:-1] + w_sorted[:-1])).astype(np.float64)
        gaps = gaps[gaps >= 0]
        mean_gap = float(gaps.mean()) if len(gaps) else 0.0
        std_gap = float(gaps.std()) if len(gaps) else 0.0
    else:
        mean_gap = 0.0
        std_gap = 0.0

    y_sorted = centroids_y[sorted_idx]
    baseline_std = float(y_sorted.std()) if len(y_sorted) > 1 else 0.0
    baseline_range = float(y_sorted.max() - y_sorted.min()) if len(y_sorted) > 1 else 0.0

    return np.array([mean_gap, std_gap, baseline_std, baseline_range], dtype=np.float64)


def extract_pca_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    n_components: Optional[float] = 0.95,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
    """Apply PCA dimensionality reduction.

    Args:
        X_train: Training feature matrix.
        X_test: Optional test feature matrix.
        n_components: Number of components (int) or variance ratio (float, default 0.95).

    Returns:
        (X_train_pca, X_test_pca, pca_model). X_test_pca is None if X_test not provided.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test) if X_test is not None else None

    print(f"PCA: reduced {X_train.shape[1]} -> {X_train_pca.shape[1]} dimensions "
          f"(explained variance: {sum(pca.explained_variance_ratio_):.3f})")

    return X_train_pca, X_test_pca, pca


def extract_all_features(
    image: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Combine every feature extractor into a single vector.

    Args:
        image: Preprocessed binary image (already size-normalized, e.g. 128x128).

    Returns:
        ``(feature_vector, feature_names)``
    """
    # NO resize here — preprocessing pipeline already standardizes size.
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