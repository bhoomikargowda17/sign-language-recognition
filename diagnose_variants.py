from dataset_generator import LandmarkDatasetGenerator
from model_manager import SignModelManager
from landmark_processor import LandmarkProcessor
import numpy as np

gen = LandmarkDatasetGenerator()
mgr = SignModelManager(model_dir='model')

def test_variant(desc, hand_dict):
    raw = [{'x': hand_dict[i][0], 'y': hand_dict[i][1], 'z': hand_dict[i][2]} for i in range(21)]
    feats = LandmarkProcessor.normalize_landmarks(raw)
    pred = mgr.predict(feats)
    print(f"{desc:35s} -> Pred: {pred['predicted_sign']:10s} | Conf: {pred['confidence']*100:5.1f}% | Top2: {pred['top_k'][:2]}")

print("=== TESTING REAL-WORLD VARIATIONS OF C ===")
base = gen._create_base_hand()

# Realistic C: Hand facing left/right sideways (ASL C)
c_side = dict(base)
# Wrist at (0.5, 0.8, 0)
# Thumb curved down-right:
c_side[1] = np.array([0.48, 0.74, 0.0])
c_side[2] = np.array([0.45, 0.68, 0.0])
c_side[3] = np.array([0.42, 0.63, 0.0])
c_side[4] = np.array([0.41, 0.58, 0.0]) # Thumb tip
# 4 fingers curved up-left forming top arc of C:
for mcp_idx, base_x in [(5, 0.49), (9, 0.50), (13, 0.51), (17, 0.52)]:
    c_side[mcp_idx] = np.array([base_x, 0.55, 0.0])
    c_side[mcp_idx+1] = np.array([base_x - 0.05, 0.47, 0.0])
    c_side[mcp_idx+2] = np.array([base_x - 0.08, 0.40, 0.0])
    c_side[mcp_idx+3] = np.array([base_x - 0.06, 0.35, 0.0]) # fingertips curved
test_variant("C (profile sideways)", c_side)

# C with thumb and fingers with wide gap
c_wide = gen._generate_sign_prototype("C")
c_wide[4] = np.array([0.35, 0.55, -0.04]) # thumb lower
c_wide[8] = np.array([0.34, 0.32, -0.04]) # index higher
test_variant("C (wide gap)", c_wide)

# Realistic NO: Index + middle tips touching thumb tip (pinched), ring & pinky curled
print("\n=== TESTING REAL-WORLD VARIATIONS OF NO ===")
no_real = dict(base)
# Thumb pointing up-inward
no_real[2] = np.array([0.44, 0.66, 0.0])
no_real[3] = np.array([0.46, 0.58, 0.0])
no_real[4] = np.array([0.48, 0.52, 0.0]) # Thumb tip
# Index extending and bending to meet thumb
no_real[5] = np.array([0.45, 0.56, 0.0])
no_real[6] = np.array([0.45, 0.47, 0.0])
no_real[7] = np.array([0.47, 0.48, 0.0])
no_real[8] = np.array([0.48, 0.51, 0.0]) # Index tip touches thumb
# Middle extending and bending to meet thumb
no_real[9] = np.array([0.50, 0.54, 0.0])
no_real[10] = np.array([0.49, 0.46, 0.0])
no_real[11] = np.array([0.48, 0.49, 0.0])
no_real[12] = np.array([0.48, 0.52, 0.0]) # Middle tip touches thumb
# Ring and pinky curled into palm
no_real[14] = np.array([0.55, 0.58, 0.02])
no_real[15] = np.array([0.54, 0.65, 0.03])
no_real[16] = np.array([0.52, 0.68, 0.02])
no_real[18] = np.array([0.60, 0.62, 0.02])
no_real[19] = np.array([0.59, 0.68, 0.03])
no_real[20] = np.array([0.56, 0.70, 0.02])
test_variant("NO (natural ASL 3-finger pinch)", no_real)

# NO with fingers slightly separated (about to pinch)
no_open = dict(no_real)
no_open[8] = np.array([0.48, 0.46, 0.0])
no_open[12] = np.array([0.49, 0.47, 0.0])
no_open[4] = np.array([0.47, 0.55, 0.0])
test_variant("NO (open beak / pre-pinch)", no_open)

# HELLO variations:
print("\n=== TESTING REAL-WORLD VARIATIONS OF HELLO ===")
# Natural waving hand: fingers spread, hand tilted ~20 deg
hello_spread = gen._generate_sign_prototype("HELLO")
# Spread fingers:
hello_spread[4] = np.array([0.24, 0.54, -0.02]) # thumb wide
hello_spread[8] = np.array([0.36, 0.23, 0.0])  # index spread left
hello_spread[12] = np.array([0.49, 0.18, 0.0]) # middle up
hello_spread[16] = np.array([0.62, 0.22, 0.0]) # ring spread right
hello_spread[20] = np.array([0.72, 0.30, 0.0]) # pinky spread far right
test_variant("HELLO (fingers naturally spread)", hello_spread)

# HELLO tilted wave (salute / wave tilt)
hello_tilt = dict(hello_spread)
test_variant("HELLO (waving posture)", hello_tilt)
