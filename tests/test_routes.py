"""HTTP-level integration tests for Flask routes."""

import sys
import os
import unittest
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app


class TestRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    # --- Main Routes ---
    def test_index_page(self):
        """Home page should return 200."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Explainable AI", resp.data)

    def test_about_page(self):
        """About page should return 200."""
        resp = self.client.get("/about")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Project Overview", resp.data)

    # --- Dysgraphia Routes ---
    def test_dysgraphia_upload_form(self):
        """Dysgraphia upload form should return 200."""
        resp = self.client.get("/dysgraphia/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"handwriting", resp.data.lower())

    def test_dysgraphia_analyze_no_file(self):
        """POST without file should redirect with flash error."""
        resp = self.client.post("/dysgraphia/analyze", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"No file selected", resp.data)

    def test_dysgraphia_analyze_with_dummy_image(self):
        """POST with a valid dummy image should process and redirect to results."""
        import numpy as np
        import cv2

        # Create a minimal valid PNG in memory
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", dummy)
        data = io.BytesIO(buf.tobytes())

        resp = self.client.post(
            "/dysgraphia/analyze",
            data={"handwriting_image": (data, "test.png")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        # Should land on results page (or upload form if no model)
        self.assertTrue(
            b"Prediction" in resp.data or b"upload" in resp.data.lower()
        )

    # --- Dyslexia Routes ---
    def test_dyslexia_test_form(self):
        """Dyslexia reading form should return 200."""
        resp = self.client.get("/dyslexia/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Reading", resp.data)

    def test_dyslexia_screen_valid(self):
        """POST valid reading metrics should redirect to results."""
        resp = self.client.post(
            "/dyslexia/screen",
            data={
                "wpm": "60",
                "errors": "5",
                "reversals": "3",
                "comprehension": "70",
                "spelling_errors": "4",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Risk Level", resp.data)

    def test_dyslexia_screen_missing_fields(self):
        """POST with missing fields should flash error."""
        resp = self.client.post(
            "/dyslexia/screen",
            data={"wpm": "60"},  # other fields missing
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Missing required fields", resp.data)

    def test_dyslexia_aggregate_form(self):
        """Visual Search aggregate form should return 200."""
        resp = self.client.get("/dyslexia/aggregate")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Visual Search", resp.data)

    def test_dyslexia_screen_aggregate_valid(self):
        """POST valid aggregate metrics should redirect to results."""
        resp = self.client.post(
            "/dyslexia/screen_aggregate",
            data={
                "age": "10",
                "gender": "1",
                "total_clicks": "400",
                "total_hits": "350",
                "total_misses": "50",
                "total_score": "1400",
                "mean_accuracy": "85",
                "mean_missrate": "12.5",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Risk Level", resp.data)

    def test_dyslexia_results_no_session(self):
        """Results page without prior screening should redirect."""
        resp = self.client.get("/dyslexia/results", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"No screening results", resp.data)


if __name__ == "__main__":
    unittest.main()
