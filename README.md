# AI-Based Real-Time Sign Language Recognition Using Hand Landmark Detection

A complete, production-ready real-time Sign Language Recognition web application built with **Flask**, **MediaPipe Hands**, **Scikit-Learn Random Forest**, and a modern responsive **HTML5/CSS3/JavaScript** user interface with optional browser **Text-to-Speech (TTS)**.

---

## 🚀 Key Features

* **Real-Time Hand Landmark Extraction:** Uses MediaPipe Hands to capture 21 3D hand skeletal landmarks at high framerate.
* **Translation & Scale Invariant Normalization:** Centers landmarks relative to the wrist (landmark 0) and normalizes coordinate vectors by maximum Euclidean distance, ensuring robust recognition regardless of hand distance or position on screen.
* **Machine Learning Classifier:** Trains and runs a Scikit-Learn `RandomForestClassifier` with dynamic class loading (no hardcoded sign classes).
* **10 Initial Configurable Signs:**
  * `HELLO`
  * `YES`
  * `NO`
  * `THANK YOU`
  * `HELP`
  * `STOP`
  * `A`
  * `B`
  * `C`
  * `D`
* **Modern Glassmorphic UI:** Deep dark obsidian background, glowing emerald/cyan neon accents, responsive layout, real-time FPS counter, and animated confidence gauges.
* **Sentence Builder & History:** Accumulates recognized gestures held for $\ge 1.2$s, with Word Space, Backspace, Copy to Clipboard, and Clear controls.
* **Browser Text-to-Speech (TTS):** Uses the browser's native Web Speech Synthesis API with voice selection, pitch/rate adjustments, auto-speak on recognized sign, and manual sentence narration.
* **In-Browser Sample Collector & Hot-Retraining:** Record new gesture samples directly from the webcam in the browser and retrain the Random Forest model on the fly without server restarts.
* **100% Local & Privacy-Friendly:** Runs completely on your local machine with zero external cloud dependencies or API keys.

---

## 📁 Project Architecture

```
sign-language-recognition/
├── app.py                   # Flask backend with REST APIs & video streaming
├── landmark_processor.py    # MediaPipe 21-landmark extractor & 3D normalizer
├── model_manager.py         # Model loading, inference, and temporal smoothing
├── train.py                 # Random Forest training and evaluation script
├── dataset_generator.py     # Biomechanically grounded 3D dataset generator
├── collect_data.py          # Standalone CLI data collector for live webcam
├── test_pipeline.py         # Automated unit and integration test suite
├── requirements.txt         # Python dependencies
├── data/
│   └── dataset.csv          # 2,000+ landmark samples (63 features + label)
├── model/
│   ├── sign_classifier.joblib # Serialized Random Forest model (96% test accuracy)
│   └── labels.json          # Dynamic class metadata & training stats
├── static/
│   ├── css/style.css        # Glassmorphic dark theme stylesheet
│   └── js/app.js            # Video stream, canvas renderer, TTS & controller
└── templates/
    └── index.html           # Main responsive web interface
```

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.11+** installed on Windows, macOS, or Linux.
* A standard webcam or laptop integrated camera.

### 1. Install Dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. (Optional) Generate Dataset & Retrain Model
A pre-trained model with 96% accuracy is already included in `model/`. If you want to regenerate data and train from scratch:
```bash
# Generate synthetic dataset (200 samples per class across 10 classes)
py -3.11 dataset_generator.py

# Train Random Forest Classifier
py -3.11 train.py
```

### 3. Run Automated Tests
Verify all components, normalization invariance, and endpoints:
```bash
py -3.11 test_pipeline.py
```

---

## 🖥️ Running the Web Application

Start the Flask server:
```bash
py -3.11 app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

1. Click **Start Camera** (allow webcam permissions in the browser).
2. Hold your hand in front of the camera.
3. MediaPipe will track the 21 hand landmarks and draw a neon green/cyan skeleton overlay on the video canvas.
4. The recognized sign, classification confidence, and class probabilities will display in real time.
5. If **Auto-Speak** is checked, the browser will announce recognized signs using Speech Synthesis.
6. Hold a sign for 1.2 seconds to append it to the **Sentence Builder**, then click **Speak Sentence** to hear the full sentence.

---

## 🖐️ Hand Signs Reference

| Sign | Pose Description |
| :--- | :--- |
| **HELLO** | Open hand, palm facing camera, fingers extended upward together with slight relaxed wave angle. |
| **YES** | Closed fist with thumb curled across fingers (affirmative nod/fist posture). |
| **NO** | Index and middle fingertips snap down to touch the thumb tip; ring and pinky curled. |
| **THANK YOU** | Flat open hand with fingers straight together, moving forward toward the camera. |
| **HELP** | Closed fist with thumb extended straight UP (thumbs up gesture). |
| **STOP** | Flat open hand pointing upward, palm flat against camera, all 5 fingers firmly extended. |
| **A** | Closed fist with thumb resting vertically alongside the index finger knuckle. |
| **B** | Four fingers pointing straight UP touching side-by-side, thumb folded flat across palm. |
| **C** | Hand arched to form a curved 'C' profile between thumb and four curved fingers. |
| **D** | Index finger pointing straight up, thumb touching middle, ring, and pinky tips in a loop. |

---

## 🎯 Adding Custom Signs

### Via Web Interface (Recommended)
1. In the web app, open the **Record Custom Signs** modal.
2. Enter the new sign name (e.g., `PEACE`).
3. Position your hand and click **Record 30 Samples**.
4. Click **Retrain Model Now**. The model will retrain in seconds and immediately begin recognizing your new sign!

### Via CLI Tool
```bash
py -3.11 collect_data.py --label PEACE --samples 50
py -3.11 train.py
```
