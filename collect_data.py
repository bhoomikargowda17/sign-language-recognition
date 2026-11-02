"""
CLI Data Collection Tool for Sign Language Landmarks.
Captures real-world hand landmark samples from webcam and saves to dataset.csv.
"""

import os
import time
import argparse
import cv2
import pandas as pd
from landmark_processor import LandmarkProcessor


def collect_samples(label, num_samples=60, camera_id=0, output_csv="data/dataset.csv"):
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    processor = LandmarkProcessor()
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return

    print(f"\n--- Data Collection for Sign: '{label}' ---")
    print(f"Target: {num_samples} samples.")
    print("Press 's' when ready to begin recording countdown, or 'q' to quit.")

    # Phase 1: Ready Screen
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        raw_lms, norm_feats, annotated, _ = processor.process_frame(frame)

        # Status text
        cv2.putText(annotated, f"Sign: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 's' to start countdown | 'q' to exit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Sign Language Data Collector", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            processor.close()
            return

    # Phase 2: Countdown
    for countdown in range(3, 0, -1):
        start_t = time.time()
        while time.time() - start_t < 1.0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            _, _, annotated, _ = processor.process_frame(frame)
            cv2.putText(annotated, f"Starting in: {countdown}", (frame.shape[1]//2 - 120, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 4)
            cv2.imshow("Sign Language Data Collector", annotated)
            cv2.waitKey(1)

    # Phase 3: Recording
    collected = []
    print("Recording samples now! Hold hand steady with slight natural movement...")

    while len(collected) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        raw_lms, norm_feats, annotated, _ = processor.process_frame(frame)

        if norm_feats and len(norm_feats) == 63:
            collected.append(norm_feats + [label])
            progress_pct = int((len(collected) / num_samples) * 100)
            cv2.putText(annotated, f"Recording: {len(collected)}/{num_samples} ({progress_pct}%)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, "Hand not detected! Place hand in frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Sign Language Data Collector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Collection aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    processor.close()

    if collected:
        cols = LandmarkProcessor.get_feature_column_names() + ["label"]
        new_df = pd.DataFrame(collected, columns=cols)
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_csv(output_csv, index=False)
        print(f"Successfully saved {len(collected)} samples for '{label}' to {output_csv}!")
        print(f"Total dataset now contains {len(combined_df)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hand landmark samples for sign recognition.")
    parser.add_argument("--label", type=str, required=True, help="Label name (e.g. HELLO, YES, STOP)")
    parser.add_argument("--samples", type=int, default=60, help="Number of samples to record")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()
    collect_samples(args.label.strip().upper(), args.samples, args.camera)
