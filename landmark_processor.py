"""
Hand Landmark Detection & Processing Module using MediaPipe Hands.
Extracts 21 3D landmarks and normalizes them for translation & scale invariance.
"""

import cv2
import mediapipe as mp
import numpy as np


class LandmarkProcessor:
    """Processes images to extract and normalize hand landmarks using MediaPipe."""

    # 21 Hand Landmark indices:
    # 0: Wrist
    # 1-4: Thumb (CMC, MCP, IP, TIP)
    # 5-8: Index (MCP, PIP, DIP, TIP)
    # 9-12: Middle (MCP, PIP, DIP, TIP)
    # 13-16: Ring (MCP, PIP, DIP, TIP)
    # 17-20: Pinky (MCP, PIP, DIP, TIP)

    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Palm base
    ]

    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

    def process_frame(self, frame_bgr):
        """
        Process a BGR image frame.
        Returns:
            raw_landmarks: list of dicts [{'x': float, 'y': float, 'z': float}] (normalized to image dimensions)
            normalized_features: list of 63 floats (translation & scale invariant)
            annotated_frame: frame with landmarks drawn (BGR)
            handedness: 'Left' or 'Right'
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, frame_bgr, None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        annotated_frame = frame_bgr.copy()

        if not results.multi_hand_landmarks:
            return None, None, annotated_frame, None

        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        # Draw hand landmarks on the annotated frame
        self.mp_draw.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw_styles.get_default_hand_landmarks_style(),
            self.mp_draw_styles.get_default_hand_connections_style()
        )

        # Extract 21 raw landmarks
        raw_landmarks = []
        for lm in hand_landmarks.landmark:
            raw_landmarks.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z)
            })

        # Calculate normalized feature vector
        normalized_features = self.normalize_landmarks(raw_landmarks)

        return raw_landmarks, normalized_features, annotated_frame, handedness

    @staticmethod
    def normalize_landmarks(raw_landmarks):
        """
        Normalizes 21 3D landmarks for translation and scale invariance.
        1. Translates coordinates so wrist (landmark 0) is at origin (0, 0, 0).
        2. Scales coordinates by maximum distance from wrist.
        Returns 63-element 1D list [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
        """
        if not raw_landmarks or len(raw_landmarks) != 21:
            return None

        # Base landmark is wrist (index 0)
        base_x = raw_landmarks[0]["x"]
        base_y = raw_landmarks[0]["y"]
        base_z = raw_landmarks[0]["z"]

        translated = []
        max_dist = 0.0

        for lm in raw_landmarks:
            dx = lm["x"] - base_x
            dy = lm["y"] - base_y
            dz = lm["z"] - base_z
            dist = (dx**2 + dy**2 + dz**2)**0.5
            if dist > max_dist:
                max_dist = dist
            translated.append((dx, dy, dz))

        if max_dist < 1e-6:
            max_dist = 1.0

        features = []
        for dx, dy, dz in translated:
            features.append(float(dx / max_dist))
            features.append(float(dy / max_dist))
            features.append(float(dz / max_dist))

        return features

    @staticmethod
    def get_feature_column_names():
        """Returns the list of 63 column header names for data export."""
        names = []
        for i in range(21):
            names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
        return names

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "hands"):
            self.hands.close()
