"""
Model Manager for Sign Language Recognition.
Loads model, performs inference with probability distributions, and implements
temporal smoothing to provide rock-solid real-time predictions.
"""

import os
import json
import joblib
from collections import deque, Counter
import numpy as np


class SignModelManager:
    """Manages model loading, inference, confidence scoring, and prediction smoothing."""

    def __init__(self, model_dir="model", smoothing_window=5, confidence_threshold=0.70):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "sign_classifier.joblib")
        self.labels_path = os.path.join(model_dir, "labels.json")
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        self.model = None
        self.classes = []
        self.metadata = {}
        self.history_buffer = deque(maxlen=smoothing_window)

        self.load_model()

    def load_model(self):
        """Loads or reloads the Random Forest model and metadata from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.labels_path):
            print(f"Warning: Model files not found in {self.model_dir}. Please train model first.")
            self.model = None
            self.classes = []
            return False

        try:
            self.model = joblib.load(self.model_path)
            with open(self.labels_path, "r") as f:
                self.metadata = json.load(f)
            self.classes = list(self.model.classes_)
            self.history_buffer.clear()
            print(f"Loaded model successfully. Active classes ({len(self.classes)}): {self.classes}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False

    def predict(self, normalized_features):
        """
        Runs inference on 63 normalized features.
        Returns:
            dict containing:
                predicted_sign: str (or "UNCERTAIN" if below threshold)
                smoothed_sign: str (stabilized via temporal window)
                confidence: float (0.0 to 1.0)
                probabilities: dict {class_name: float}
                top_k: list of tuples [(class_name, prob), ...]
        """
        if self.model is None or normalized_features is None:
            return {
                "predicted_sign": None,
                "smoothed_sign": None,
                "confidence": 0.0,
                "probabilities": {},
                "top_k": []
            }

        features_arr = np.array(normalized_features).reshape(1, -1)
        probas = self.model.predict_proba(features_arr)[0]

        best_idx = np.argmax(probas)
        best_confidence = float(probas[best_idx])
        raw_sign = self.classes[best_idx]

        # Probability dictionary
        prob_dict = {
            self.classes[i]: float(probas[i])
            for i in range(len(self.classes))
        }

        # Top 3 predictions
        sorted_indices = np.argsort(probas)[::-1][:3]
        top_k = [
            {"sign": self.classes[i], "probability": round(float(probas[i]), 3)}
            for i in sorted_indices
        ]

        margin = 1.0
        if len(sorted_indices) >= 2:
            margin = float(probas[sorted_indices[0]] - probas[sorted_indices[1]])

        if best_confidence >= self.confidence_threshold and margin >= 0.15:
            accepted_sign = raw_sign
        else:
            accepted_sign = "UNCERTAIN"

        # Update smoothing buffer
        self.history_buffer.append(accepted_sign)

        # Compute smoothed prediction (majority vote in buffer)
        counts = Counter(self.history_buffer)
        most_common_sign, most_common_count = counts.most_common(1)[0]
        # Require at least 50% majority in the window to switch smoothed sign
        if most_common_count >= max(2, len(self.history_buffer) // 2):
            smoothed_sign = most_common_sign
        else:
            smoothed_sign = accepted_sign

        return {
            "predicted_sign": accepted_sign,
            "smoothed_sign": smoothed_sign,
            "confidence": round(best_confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in prob_dict.items()},
            "top_k": top_k
        }

    def reset_buffer(self):
        """Clears the temporal smoothing buffer."""
        self.history_buffer.clear()

    def get_info(self):
        """Returns model status and metadata."""
        return {
            "is_loaded": self.model is not None,
            "classes": self.classes,
            "num_classes": len(self.classes),
            "metadata": self.metadata
        }
