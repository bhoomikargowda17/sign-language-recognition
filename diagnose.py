from dataset_generator import LandmarkDatasetGenerator
from model_manager import SignModelManager
from landmark_processor import LandmarkProcessor
import numpy as np

gen = LandmarkDatasetGenerator()
mgr = SignModelManager(model_dir='model')

print(f"Active classes: {mgr.classes}")

for name in ['HELLO', 'STOP', 'THANK YOU', 'C', 'NO', 'D', 'A', 'B', 'HELP', 'YES']:
    proto = gen._generate_sign_prototype(name)
    raw = [{'x': proto[i][0], 'y': proto[i][1], 'z': proto[i][2]} for i in range(21)]
    feats = LandmarkProcessor.normalize_landmarks(raw)
    pred = mgr.predict(feats)
    print(f"Proto {name:10s} -> Pred: {pred['predicted_sign']:10s} | Conf: {pred['confidence']*100:5.1f}% | Top2: {pred['top_k'][:2]}")

print("\n--- Testing Flipped (Left Hand) ---")
for name in ['HELLO', 'C', 'NO']:
    proto = gen._generate_sign_prototype(name)
    # Mirror x around center 0.5
    raw_flipped = [{'x': 1.0 - proto[i][0], 'y': proto[i][1], 'z': proto[i][2]} for i in range(21)]
    feats = LandmarkProcessor.normalize_landmarks(raw_flipped)
    pred = mgr.predict(feats)
    print(f"Flipped {name:10s} -> Pred: {pred['predicted_sign']:10s} | Conf: {pred['confidence']*100:5.1f}% | Top2: {pred['top_k'][:2]}")
