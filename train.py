"""
Model Training Script for Sign Language Recognition.
Trains a Scikit-Learn Random Forest Classifier on extracted hand landmark features
and saves the trained model and dynamic class definitions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from landmark_processor import LandmarkProcessor


def train_model(data_path="data/dataset.csv", model_dir="model", n_estimators=150):
    """
    Trains Random Forest model on the landmark dataset and serializes artifacts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please generate or collect data first.")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sign_classifier.joblib")
    labels_path = os.path.join(model_dir, "labels.json")

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise ValueError("Dataset CSV must contain a 'label' column.")

    feature_cols = LandmarkProcessor.get_feature_column_names()
    # Check that all feature columns are present
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing expected feature columns: {missing_cols[:5]}...")

    X = df[feature_cols].values
    y = df["label"].values

    classes = sorted(list(np.unique(y)))
    print(f"Discovered {len(classes)} dynamic sign classes: {classes}")
    print(f"Total samples: {len(X)} across {X.shape[1]} features.")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    print(f"Training Random Forest Classifier (n_estimators={n_estimators})...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=25,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n================ MODEL EVALUATION ================")
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model Artifacts
    print(f"Saving trained model to {model_path}...")
    joblib.dump(clf, model_path)

    metadata = {
        "classes": list(clf.classes_),
        "num_classes": len(clf.classes_),
        "num_features": len(feature_cols),
        "total_samples": len(df),
        "test_accuracy": float(acc),
        "timestamp": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators
    }

    with open(labels_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved class metadata to {labels_path}.")

    return clf, metadata


if __name__ == "__main__":
    train_model()
