"""
Automated Pipeline and Endpoint Test Suite.
Verifies landmark extraction, normalization, model inference, and Flask endpoints.
"""

import unittest
import json
import base64
import numpy as np
import cv2
from landmark_processor import LandmarkProcessor
from model_manager import SignModelManager
from dataset_generator import LandmarkDatasetGenerator
from app import app


class TestSignLanguageRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = LandmarkDatasetGenerator()
        cls.processor = LandmarkProcessor()
        cls.manager = SignModelManager(model_dir="model")
        cls.app_client = app.test_client()

    def test_model_loaded(self):
        """Verify model and labels load correctly."""
        self.assertTrue(self.manager.model is not None, "Model should be loaded")
        self.assertGreaterEqual(len(self.manager.classes), 10, "Should have at least the 10 core classes")
        core_classes = ["A", "B", "C", "D", "HELLO", "HELP", "NO", "STOP", "THANK YOU", "YES"]
        for cls_name in core_classes:
            self.assertIn(cls_name, self.manager.classes, f"Core class {cls_name} should be present")

    def test_normalization_invariance(self):
        """Verify translation and scale invariance."""
        hand = self.generator._create_base_hand()
        raw_lms = [{"x": hand[i][0], "y": hand[i][1], "z": hand[i][2]} for i in range(21)]

        norm1 = LandmarkProcessor.normalize_landmarks(raw_lms)
        self.assertEqual(len(norm1), 63)
        # Wrist at origin (first 3 coords should be 0.0)
        self.assertAlmostEqual(norm1[0], 0.0, places=5)
        self.assertAlmostEqual(norm1[1], 0.0, places=5)
        self.assertAlmostEqual(norm1[2], 0.0, places=5)

        # Shift all coordinates by (+10.0, +20.0, -5.0) -> translation invariance test
        shifted_lms = [{"x": lm["x"] + 10.0, "y": lm["y"] + 20.0, "z": lm["z"] - 5.0} for lm in raw_lms]
        norm2 = LandmarkProcessor.normalize_landmarks(shifted_lms)
        for v1, v2 in zip(norm1, norm2):
            self.assertAlmostEqual(v1, v2, places=4)

    def test_inference_accuracy_on_prototypes(self):
        """Verify inference returns correct sign with high confidence for clear prototypes."""
        for sign in ["A", "B", "C", "D", "HELP", "STOP", "YES"]:
            proto = self.generator._generate_sign_prototype(sign)
            raw = [{"x": proto[i][0], "y": proto[i][1], "z": proto[i][2]} for i in range(21)]
            features = LandmarkProcessor.normalize_landmarks(raw)
            result = self.manager.predict(features)
            self.assertEqual(result["predicted_sign"], sign, f"Expected {sign}, got {result['predicted_sign']}")
            self.assertGreater(result["confidence"], 0.70, f"Confidence for {sign} should be > 0.70")

    def test_flask_health_endpoint(self):
        """Verify GET /health."""
        res = self.app_client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["total_classes"], len(self.manager.classes))

    def test_flask_classes_endpoint(self):
        """Verify GET /classes."""
        res = self.app_client.get("/classes")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["num_classes"], len(self.manager.classes))

    def test_flask_predict_endpoint_no_hand(self):
        """Verify POST /predict with blank image returns hand_detected: False."""
        blank_img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank_img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        res = self.app_client.post("/predict", json={"image": b64})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertFalse(data["hand_detected"])
        self.assertIsNone(data["sign"])


if __name__ == "__main__":
    unittest.main()
