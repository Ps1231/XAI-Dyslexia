"""Unit tests for image preprocessing pipeline."""

import sys
import os
import unittest
import numpy as np

# Ensure app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.preprocessing import (
    load_image,
    to_grayscale,
    binarize,
    denoise,
    deskew,
    resize_and_pad,
    preprocess_pipeline,
)


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_pipeline_output_shape(self):
        """preprocess_pipeline must always return (128, 128)."""
        # Create a dummy BGR image
        dummy = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        out = preprocess_pipeline(dummy)
        self.assertEqual(out.shape, (128, 128), f"Expected (128,128), got {out.shape}")

    def test_preprocess_pipeline_grayscale_input(self):
        """Should handle grayscale input without crashing."""
        gray = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
        out = preprocess_pipeline(gray)
        self.assertEqual(out.shape, (128, 128))

    def test_preprocess_pipeline_binary_output(self):
        """Output should be binary (0 or 255)."""
        dummy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = preprocess_pipeline(dummy)
        unique = np.unique(out)
        self.assertTrue(
            set(unique.tolist()).issubset({0, 255}),
            f"Output contains non-binary values: {unique}",
        )

    def test_to_grayscale_idempotent(self):
        """Passing grayscale to to_grayscale should return same array."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        out = to_grayscale(gray)
        np.testing.assert_array_equal(gray, out)

    def test_binarize_otsu(self):
        """Otsu binarization should return only 0 and 255."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = binarize(gray, method="otsu")
        unique = np.unique(binary)
        self.assertTrue(set(unique.tolist()).issubset({0, 255}))

    def test_denoise_methods(self):
        """Denoise should not crash on any method."""
        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        for method in ["gaussian", "median", "morphological"]:
            out = denoise(gray, method=method)
            self.assertEqual(out.shape, gray.shape)

    def test_deskew_empty_image(self):
        """Deskew on empty image should return input without crash."""
        empty = np.zeros((100, 100), dtype=np.uint8)
        out = deskew(empty)
        self.assertEqual(out.shape, empty.shape)

    def test_resize_and_pad_aspect_ratio(self):
        """resize_and_pad should preserve aspect ratio and pad to target."""
        tall = np.ones((200, 100), dtype=np.uint8) * 128
        out = resize_and_pad(tall, target_size=(128, 128))
        self.assertEqual(out.shape, (128, 128))
        # Check that content is centered (not all zeros)
        self.assertTrue(out.sum() > 0)

    def test_load_image_none_path(self):
        """load_image with empty path should return None."""
        self.assertIsNone(load_image(""))


if __name__ == "__main__":
    unittest.main()
