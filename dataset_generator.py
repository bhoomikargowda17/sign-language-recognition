"""
Dataset Generator for Sign Language Hand Landmarks.
Synthesizes a robust foundational dataset of realistic 3D hand landmark variations
for the 10 target classes:
HELLO, YES, NO, THANK YOU, HELP, STOP, A, B, C, D.
Includes multi-prototype variants, bilateral hand mirroring, and 3D augmentation.
"""

import os
import math
import random
import pandas as pd
import numpy as np
from landmark_processor import LandmarkProcessor


class LandmarkDatasetGenerator:
    """Generates biomechanically grounded hand landmark datasets with bilateral augmentation."""

    CLASSES = [
        "HELLO",
        "YES",
        "NO",
        "THANK YOU",
        "HELP",
        "STOP",
        "A",
        "B",
        "C",
        "D"
    ]

    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_csv = os.path.join(self.output_dir, "dataset.csv")

    @staticmethod
    def _create_base_hand():
        """Returns standard neutral 21 landmark coordinates (wrist at bottom center)."""
        return {
            0:  np.array([0.50, 0.85, 0.00]), # Wrist
            # Thumb
            1:  np.array([0.43, 0.76, -0.01]),
            2:  np.array([0.37, 0.67, -0.02]),
            3:  np.array([0.33, 0.58, -0.03]),
            4:  np.array([0.30, 0.50, -0.04]),
            # Index
            5:  np.array([0.44, 0.56, -0.01]),
            6:  np.array([0.43, 0.44, -0.01]),
            7:  np.array([0.42, 0.35, -0.01]),
            8:  np.array([0.41, 0.25, -0.01]),
            # Middle
            9:  np.array([0.50, 0.54, 0.00]),
            10: np.array([0.50, 0.41, 0.00]),
            11: np.array([0.50, 0.31, 0.00]),
            12: np.array([0.50, 0.21, 0.00]),
            # Ring
            13: np.array([0.56, 0.56, 0.01]),
            14: np.array([0.57, 0.44, 0.01]),
            15: np.array([0.58, 0.35, 0.01]),
            16: np.array([0.59, 0.26, 0.01]),
            # Pinky
            17: np.array([0.62, 0.60, 0.02]),
            18: np.array([0.64, 0.50, 0.02]),
            19: np.array([0.65, 0.43, 0.02]),
            20: np.array([0.66, 0.35, 0.02])
        }

    def _generate_sign_prototype(self, sign_name, variant=0):
        """Generates canonical 21-landmark model for a specific sign with variant support."""
        hand = self._create_base_hand()

        if sign_name == "HELLO":
            # HELLO: Open waving hand, fingers spread wide apart in a greeting fan shape
            # Thumb spread far out
            hand[1] = np.array([0.41, 0.76, -0.02])
            hand[2] = np.array([0.33, 0.68, -0.03])
            hand[3] = np.array([0.25, 0.60, -0.04])
            hand[4] = np.array([0.18, 0.52, -0.04]) # Thumb wide out

            # Index spread left
            hand[6] = np.array([0.38, 0.43, -0.01])
            hand[7] = np.array([0.35, 0.32, -0.01])
            hand[8] = np.array([0.32, 0.21, -0.01])

            # Middle straight up
            hand[10] = np.array([0.48, 0.40, 0.0])
            hand[11] = np.array([0.48, 0.28, 0.0])
            hand[12] = np.array([0.48, 0.17, 0.0])

            # Ring spread right
            hand[14] = np.array([0.58, 0.42, 0.01])
            hand[15] = np.array([0.62, 0.31, 0.01])
            hand[16] = np.array([0.65, 0.20, 0.01])

            # Pinky spread far right
            hand[18] = np.array([0.67, 0.49, 0.02])
            hand[19] = np.array([0.73, 0.40, 0.02])
            hand[20] = np.array([0.78, 0.30, 0.02])

        elif sign_name == "STOP":
            # STOP: Rigid flat palm, fingers straight up and held tightly together
            hand[4] = np.array([0.33, 0.52, -0.02]) # thumb held close
            # All 4 fingers touching tightly side-by-side
            hand[6] = np.array([0.44, 0.42, 0.0])
            hand[7] = np.array([0.44, 0.30, 0.0])
            hand[8] = np.array([0.44, 0.19, 0.0])

            hand[10] = np.array([0.48, 0.40, 0.0])
            hand[11] = np.array([0.48, 0.28, 0.0])
            hand[12] = np.array([0.48, 0.17, 0.0])

            hand[14] = np.array([0.52, 0.41, 0.0])
            hand[15] = np.array([0.52, 0.29, 0.0])
            hand[16] = np.array([0.52, 0.18, 0.0])

            hand[18] = np.array([0.56, 0.44, 0.0])
            hand[19] = np.array([0.56, 0.33, 0.0])
            hand[20] = np.array([0.56, 0.23, 0.0])

        elif sign_name == "C":
            # C: Curved hand forming a clear 'C' shape with open gap between thumb and fingers
            # variant 0: opening facing left, variant 1: opening facing right, variant 2: frontal curved
            facing_dir = -1.0 if (variant % 2 == 1) else 1.0
            cx = 0.50

            # Thumb curving up and into the bottom arc of C
            hand[1] = np.array([cx - 0.04 * facing_dir, 0.76, 0.0])
            hand[2] = np.array([cx - 0.08 * facing_dir, 0.70, 0.0])
            hand[3] = np.array([cx - 0.13 * facing_dir, 0.64, 0.0])
            hand[4] = np.array([cx - 0.11 * facing_dir, 0.56, 0.0]) # Thumb tip at bottom of C

            # 4 fingers curved forward/down forming upper arc of C
            for base_idx, mcp_x, tip_x in [
                (5, cx - 0.03 * facing_dir, cx - 0.11 * facing_dir),
                (9, cx, cx - 0.09 * facing_dir),
                (13, cx + 0.03 * facing_dir, cx - 0.07 * facing_dir),
                (17, cx + 0.06 * facing_dir, cx - 0.05 * facing_dir)
            ]:
                hand[base_idx] = np.array([mcp_x, 0.54, 0.0])
                hand[base_idx + 1] = np.array([mcp_x - 0.04 * facing_dir, 0.43, -0.02])
                hand[base_idx + 2] = np.array([mcp_x - 0.08 * facing_dir, 0.35, -0.04])
                hand[base_idx + 3] = np.array([tip_x, 0.33, -0.05]) # Fingertip at top of C

        elif sign_name == "NO":
            # NO: ASL 3-finger pinch: Index and middle extended and touching thumb tip; ring & pinky curled
            # variant 0: pinched closed; variant 1: slightly open beak
            is_closed = (variant % 2 == 0)
            thumb_y = 0.48 if is_closed else 0.51
            tip_y = 0.48 if is_closed else 0.45

            hand[1] = np.array([0.45, 0.75, 0.0])
            hand[2] = np.array([0.43, 0.66, 0.0])
            hand[3] = np.array([0.44, 0.57, 0.0])
            hand[4] = np.array([0.46, thumb_y, 0.02]) # Thumb tip

            # Index extended and touching/close to thumb
            hand[5] = np.array([0.44, 0.55, 0.0])
            hand[6] = np.array([0.43, 0.44, 0.0])
            hand[7] = np.array([0.44, 0.42, 0.01])
            hand[8] = np.array([0.46, tip_y, 0.02]) # Index tip meets thumb

            # Middle extended and touching/close to thumb
            hand[9] = np.array([0.49, 0.54, 0.0])
            hand[10] = np.array([0.48, 0.43, 0.0])
            hand[11] = np.array([0.47, 0.42, 0.01])
            hand[12] = np.array([0.47, tip_y + 0.01, 0.02]) # Middle tip meets thumb

            # Ring and pinky curled into palm
            hand[13] = np.array([0.54, 0.56, 0.0])
            hand[14] = np.array([0.54, 0.62, 0.03])
            hand[15] = np.array([0.53, 0.68, 0.04])
            hand[16] = np.array([0.51, 0.71, 0.03])

            hand[17] = np.array([0.58, 0.59, 0.0])
            hand[18] = np.array([0.58, 0.64, 0.03])
            hand[19] = np.array([0.57, 0.70, 0.04])
            hand[20] = np.array([0.55, 0.73, 0.03])

        elif sign_name == "A":
            # Fist with thumb resting alongside index MCP, all 4 fingers curled into palm
            hand[6] = np.array([0.43, 0.50, 0.03])
            hand[7] = np.array([0.44, 0.58, 0.04])
            hand[8] = np.array([0.44, 0.65, 0.03])

            hand[10] = np.array([0.50, 0.49, 0.03])
            hand[11] = np.array([0.50, 0.57, 0.04])
            hand[12] = np.array([0.50, 0.64, 0.03])

            hand[14] = np.array([0.56, 0.50, 0.03])
            hand[15] = np.array([0.56, 0.58, 0.04])
            hand[16] = np.array([0.55, 0.65, 0.03])

            hand[18] = np.array([0.62, 0.54, 0.03])
            hand[19] = np.array([0.62, 0.61, 0.04])
            hand[20] = np.array([0.61, 0.68, 0.03])

            hand[3] = np.array([0.38, 0.50, -0.01])
            hand[4] = np.array([0.40, 0.42, 0.01])

        elif sign_name == "B":
            # 4 fingers straight up together, thumb crossed flat over palm
            hand[6] = np.array([0.43, 0.43, 0.00])
            hand[7] = np.array([0.43, 0.32, 0.00])
            hand[8] = np.array([0.43, 0.21, 0.00])
            hand[10] = np.array([0.48, 0.41, 0.00])
            hand[11] = np.array([0.48, 0.30, 0.00])
            hand[12] = np.array([0.48, 0.19, 0.00])
            hand[14] = np.array([0.53, 0.42, 0.00])
            hand[15] = np.array([0.53, 0.32, 0.00])
            hand[16] = np.array([0.53, 0.22, 0.00])
            hand[18] = np.array([0.58, 0.46, 0.00])
            hand[19] = np.array([0.58, 0.37, 0.00])
            hand[20] = np.array([0.58, 0.28, 0.00])
            # Thumb folded across palm
            hand[2] = np.array([0.44, 0.68, 0.03])
            hand[3] = np.array([0.48, 0.64, 0.04])
            hand[4] = np.array([0.52, 0.62, 0.04])

        elif sign_name == "D":
            # Index extended straight up, middle/ring/pinky curled touching thumb
            hand[6] = np.array([0.44, 0.42, 0.00])
            hand[7] = np.array([0.44, 0.31, 0.00])
            hand[8] = np.array([0.44, 0.20, 0.00]) # Index UP

            hand[10] = np.array([0.49, 0.50, 0.03])
            hand[11] = np.array([0.48, 0.57, 0.04])
            hand[12] = np.array([0.45, 0.60, 0.03])

            hand[14] = np.array([0.54, 0.52, 0.03])
            hand[15] = np.array([0.53, 0.60, 0.04])
            hand[16] = np.array([0.50, 0.64, 0.03])

            hand[18] = np.array([0.60, 0.55, 0.03])
            hand[19] = np.array([0.58, 0.63, 0.04])
            hand[20] = np.array([0.55, 0.67, 0.03])

            hand[3] = np.array([0.40, 0.64, 0.02])
            hand[4] = np.array([0.44, 0.60, 0.03])

        elif sign_name == "HELP":
            # Thumbs UP: Thumb pointing straight up, four fingers curled in tight fist
            hand[2] = np.array([0.38, 0.63, -0.02])
            hand[3] = np.array([0.38, 0.48, -0.04])
            hand[4] = np.array([0.38, 0.32, -0.05]) # Thumb UP
            for base_idx, mcp_x in [(5, 0.44), (9, 0.50), (13, 0.56), (17, 0.62)]:
                hand[base_idx + 1] = np.array([mcp_x, 0.52, 0.04])
                hand[base_idx + 2] = np.array([mcp_x, 0.60, 0.05])
                hand[base_idx + 3] = np.array([mcp_x, 0.67, 0.04])

        elif sign_name == "YES":
            # Closed fist: all fingers curled tight, thumb folded in front
            for base_idx, mcp_x in [(5, 0.44), (9, 0.50), (13, 0.56), (17, 0.62)]:
                hand[base_idx + 1] = np.array([mcp_x, 0.50, 0.04])
                hand[base_idx + 2] = np.array([mcp_x, 0.59, 0.05])
                hand[base_idx + 3] = np.array([mcp_x, 0.66, 0.04])
            hand[2] = np.array([0.42, 0.65, 0.04])
            hand[3] = np.array([0.47, 0.61, 0.06])
            hand[4] = np.array([0.53, 0.60, 0.06])

        elif sign_name == "THANK YOU":
            # Flat open hand moving forward toward camera (z-depth)
            hand[4] = np.array([0.34, 0.48, 0.01])
            hand[8] = np.array([0.43, 0.28, 0.06])
            hand[12] = np.array([0.49, 0.25, 0.06])
            hand[16] = np.array([0.55, 0.28, 0.06])
            hand[20] = np.array([0.61, 0.34, 0.06])

        elif sign_name == "NEUTRAL":
            # Natural relaxed hand in resting state
            # Fingers loosely relaxed/partially curled, thumb resting comfortably
            hand[1] = np.array([0.43, 0.76, -0.01])
            hand[2] = np.array([0.38, 0.68, -0.01])
            hand[3] = np.array([0.35, 0.60, -0.02])
            hand[4] = np.array([0.34, 0.52, -0.02])

            for base_idx, mcp_x in [(5, 0.44), (9, 0.50), (13, 0.56), (17, 0.62)]:
                hand[base_idx + 1] = np.array([mcp_x, 0.46, 0.01])
                hand[base_idx + 2] = np.array([mcp_x, 0.38, 0.02])
                hand[base_idx + 3] = np.array([mcp_x, 0.31, 0.02])

        return hand

    @staticmethod
    def _apply_3d_transform(hand_dict, yaw_deg, pitch_deg, roll_deg, scale, mirror_h=False, noise_sigma=0.012):
        """Applies 3D rotation, scaling, Gaussian jitter, and optional horizontal reflection."""
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        R_z = np.array([
            [math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll), math.cos(roll), 0],
            [0, 0, 1]
        ])
        R_y = np.array([
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ])
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])
        R = R_z @ R_y @ R_x

        wrist = hand_dict[0]
        augmented_raw = []

        for i in range(21):
            pt = hand_dict[i].copy()
            if mirror_h:
                # Reflect x coordinate around wrist x
                pt[0] = 2.0 * wrist[0] - pt[0]

            rel = (pt - wrist) * scale
            rotated = R @ rel
            jitter = np.random.normal(0, noise_sigma, 3)
            final_pt = wrist + rotated + jitter

            augmented_raw.append({
                "x": float(final_pt[0]),
                "y": float(final_pt[1]),
                "z": float(final_pt[2])
            })

        return augmented_raw

    def generate_dataset(self, samples_per_class=300):
        """
        Generates full dataset and saves to CSV.
        Includes 50% left-hand / mirrored samples and multiple variant configurations.
        """
        records = []
        columns = LandmarkProcessor.get_feature_column_names() + ["label"]

        print(f"Generating enhanced landmark dataset ({samples_per_class} samples/class)...")

        for sign in self.CLASSES:
            for sample_idx in range(samples_per_class):
                variant = sample_idx % 4
                prototype = self._generate_sign_prototype(sign, variant=variant)

                yaw = random.uniform(-22.0, 22.0)
                pitch = random.uniform(-18.0, 18.0)
                roll = random.uniform(-22.0, 22.0)
                scale = random.uniform(0.82, 1.22)
                jitter = random.uniform(0.008, 0.016)

                # 50% chance of horizontal reflection (left hand / mirrored webcam)
                mirror_h = (sample_idx % 2 == 1)

                raw_landmarks = self._apply_3d_transform(prototype, yaw, pitch, roll, scale, mirror_h=mirror_h, noise_sigma=jitter)
                normalized_features = LandmarkProcessor.normalize_landmarks(raw_landmarks)

                if normalized_features and len(normalized_features) == 63:
                    records.append(normalized_features + [sign])

        df = pd.DataFrame(records, columns=columns)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df.to_csv(self.output_csv, index=False)
        print(f"Successfully generated {len(df)} samples saved to {self.output_csv}!")
        return df


if __name__ == "__main__":
    generator = LandmarkDatasetGenerator()
    generator.generate_dataset(samples_per_class=300)
