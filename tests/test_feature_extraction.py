"""Unit tests for feature extraction module."""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.feature_extraction import (
    extract_hog_features,
    extract_contour_features,
    extract_projection_features,
    extract_morphological_features,
    extract_letter_spacing_features,
    extract_all_features,
)


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        """Create a standard 128x128 binary test image."""
        self.img = np.zeros((128, 128), dtype=np.uint8)
        # Draw a simple rectangle
        self.img[30:100, 30:100] = 255

    def test_extract_all_features_shape(self):
        """extract_all_features must return a non-empty 1-D vector."""
        features, names = extract_all_features(self.img)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)
        self.assertGreater(len(features), 0)
        self.assertEqual(len(features), len(names))

    def test_feature_vector_consistency(self):
        """Same input should produce same feature vector."""
        f1, _ = extract_all_features(self.img)
        f2, _ = extract_all_features(self.img)
        np.testing.assert_array_equal(f1, f2)

    def test_empty_image_handling(self):
        """Empty image should not crash and return defaults."""
        empty = np.zeros((128, 128), dtype=np.uint8)
        features, names = extract_all_features(empty)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(names))

    def test_hog_descriptor_length(self):
        """HOG should return a flat vector of reasonable length."""
        hog = extract_hog_features(self.img)
        self.assertEqual(hog.ndim, 1)
        self.assertGreater(len(hog), 10)

    def test_contour_features_length(self):
        """Contour features should always return 4 values."""
        contour = extract_contour_features(self.img)
        self.assertEqual(len(contour), 4)

    def test_projection_features_length(self):
        """Projection features should always return 6 values."""
        proj = extract_projection_features(self.img)
        self.assertEqual(len(proj), 6)

    def test_morphological_features_length(self):
        """Morphological features should always return 9 values."""
        morph = extract_morphological_features(self.img)
        self.assertEqual(len(morph), 9)

    def test_letter_spacing_features_length(self):
        """Letter spacing features should always return 4 values."""
        spacing = extract_letter_spacing_features(self.img)
        self.assertEqual(len(spacing), 4)

    def test_none_input_safety(self):
        """None input should not crash (returns defaults)."""
        for func in [
            extract_hog_features,
            extract_contour_features,
            extract_projection_features,
            extract_morphological_features,
            extract_letter_spacing_features,
        ]:
            with self.subTest(func=func.__name__):
                result = func(None)
                self.assertIsInstance(result, np.ndarray)
                self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
