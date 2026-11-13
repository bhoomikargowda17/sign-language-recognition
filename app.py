"""
Flask Backend Application for Real-Time Sign Language Recognition.
Exposes REST endpoints for real-time landmark inference, model retraining,
dynamic class management, and video streaming.
"""

import os
import io
import re
import json
import base64
import threading
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth, messaging
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    firebase_admin = None
    credentials = firestore = auth = messaging = None

from landmark_processor import LandmarkProcessor
from model_manager import SignModelManager
from train import train_model

app = Flask(__name__)
app.config["SECRET_KEY"] = "sign-language-ai-secret-key-2026"
# Initialize Firebase Admin SDK if service account is available (e.g. Render secret files)
firebase_service_account_path = os.environ.get(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    "/etc/secrets/firebase-service-account.json"
)
db = None
try:
    if HAS_FIREBASE and os.path.exists(firebase_service_account_path):
        firebase_cred = credentials.Certificate(firebase_service_account_path)
        firebase_admin.initialize_app(firebase_cred)
        db = firestore.client()
        print("Firebase Admin initialized successfully.")
    else:
        print(f"Notice: Firebase service account not found at {firebase_service_account_path}. Messaging features will run in offline mode.")
except Exception as fb_err:
    print(f"Warning: Failed to initialize Firebase Admin: {fb_err}")


# Initialize singletons
processor = LandmarkProcessor()
model_manager = SignModelManager(model_dir="model")

# Server-side camera & dataset synchronization locks
camera_lock = threading.Lock()
dataset_lock = threading.Lock()
retrain_lock = threading.Lock()
server_camera = None


def decode_base64_image(image_data):
    """Decodes base64 data URL to OpenCV BGR image."""
    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


@app.route("/")
def index():
    """Serves the main application page."""
    model_info = model_manager.get_info()
    return render_template("index.html", model_info=model_info)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint:
    Receives base64 image frame from browser webcam, processes landmarks with MediaPipe,
    and classifies sign using Random Forest.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No image provided"}), 400

    frame = decode_base64_image(data["image"])
    if frame is None:
        return jsonify({"success": False, "error": "Failed to decode image frame"}), 400

    # Flip horizontally for natural mirror feel if requested or by default
    if data.get("flip", False):
        frame = cv2.flip(frame, 1)

    raw_landmarks, norm_features, annotated_frame, handedness = processor.process_frame(frame)

    if raw_landmarks is None or norm_features is None:
        model_manager.reset_buffer()
        return jsonify({
            "success": True,
            "hand_detected": False,
            "sign": None,
            "smoothed_sign": None,
            "confidence": 0.0,
            "probabilities": {},
            "top_k": [],
            "landmarks": None,
            "handedness": None
        })

    # Run inference
    pred = model_manager.predict(norm_features)

    response_payload = {
        "success": True,
        "hand_detected": True,
        "sign": pred["predicted_sign"],
        "raw_sign": pred.get("raw_sign"),
        "smoothed_sign": pred["smoothed_sign"],
        "confidence": pred["confidence"],
        "probabilities": pred["probabilities"],
        "top_k": pred["top_k"],
        "landmarks": raw_landmarks,
        "handedness": handedness,
        "connections": LandmarkProcessor.CONNECTIONS
    }

    # Optionally return annotated frame in base64 if client requests server rendering
    if data.get("return_annotated", False):
        _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
        response_payload["annotated_frame"] = annotated_b64

    return jsonify(response_payload)


@app.route("/classes", methods=["GET"])
def get_classes():
    """Returns the list of currently recognized classes."""
    return jsonify({
        "success": True,
        "classes": model_manager.classes,
        "num_classes": len(model_manager.classes),
        "metadata": model_manager.metadata
    })


@app.route("/collect_sample", methods=["POST"])
def collect_sample():
    """
    Appends a new sample directly to the dataset from the web interface.
    Accepts: { label: str, image: base64_image } or { label: str, landmarks: [...] }
    Includes bilateral mirroring augmentation for left/right hand robustness.
    """
    data = request.get_json(silent=True)
    if not data or "label" not in data:
        return jsonify({"success": False, "error": "Missing label"}), 400

    label = data["label"].strip().upper()
    label = re.sub(r"\s+", " ", label)
    if not label or not re.match(r"^[A-Z0-9 _-]+$", label):
        return jsonify({"success": False, "error": "Invalid label format (letters, numbers, spaces, and hyphens only)"}), 400

    if label in ["UNCERTAIN", "NONE", "NULL", "UNKNOWN"]:
        return jsonify({"success": False, "error": f"'{label}' is a reserved keyword. Please choose another name."}), 400

    norm_features = None

    if "landmarks" in data and len(data["landmarks"]) == 21:
        norm_features = LandmarkProcessor.normalize_landmarks(data["landmarks"])
    elif "image" in data:
        frame = decode_base64_image(data["image"])
        if frame is not None:
            _, norm_features, _, _ = processor.process_frame(frame)

    if norm_features is None or len(norm_features) != 63:
        return jsonify({"success": False, "error": "Could not extract valid hand landmarks from sample"}), 400

    # Bilateral horizontal reflection (-x, y, z) for dual-hand invariance
    mirrored_features = []
    for idx in range(21):
        x = norm_features[idx * 3]
        y = norm_features[idx * 3 + 1]
        z = norm_features[idx * 3 + 2]
        mirrored_features.extend([round(-x, 5), round(y, 5), round(z, 5)])

    # Append to dataset.csv with thread safety
    csv_path = "data/dataset.csv"
    os.makedirs("data", exist_ok=True)
    cols = LandmarkProcessor.get_feature_column_names() + ["label"]
    new_rows_df = pd.DataFrame([
        norm_features + [label],
        mirrored_features + [label]
    ], columns=cols)

    with dataset_lock:
        if os.path.exists(csv_path):
            new_rows_df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            new_rows_df.to_csv(csv_path, mode="w", header=True, index=False)

    return jsonify({
        "success": True,
        "message": f"Sample for '{label}' saved successfully.",
        "label": label
    })


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Triggers model retraining on data/dataset.csv and reloads the active model.
    Guarantees thread-safe single-instance execution.
    """
    if not retrain_lock.acquire(blocking=False):
        return jsonify({
            "success": False,
            "error": "Model retraining is already in progress. Please wait a moment."
        }), 429

    try:
        data_path = "data/dataset.csv"
        if not os.path.exists(data_path):
            return jsonify({"success": False, "error": "Dataset file data/dataset.csv does not exist"}), 400

        with dataset_lock:
            clf, metadata = train_model(data_path=data_path, model_dir="model")

        reload_success = model_manager.load_model()

        return jsonify({
            "success": True,
            "message": "Model retrained and loaded successfully!",
            "accuracy": metadata.get("test_accuracy", 0.0),
            "num_classes": metadata.get("num_classes", 0),
            "classes": metadata.get("classes", []),
            "total_samples": metadata.get("total_samples", 0)
        })
    except Exception as e:
        print(f"Error during retraining: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        retrain_lock.release()


def gen_video_frames():
    """Generator for server-side camera MJPEG video stream."""
    global server_camera
    with camera_lock:
        if server_camera is None or not server_camera.isOpened():
            server_camera = cv2.VideoCapture(0)

    while True:
        with camera_lock:
            if server_camera is None or not server_camera.isOpened():
                break
            success, frame = server_camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        raw_lms, norm_feats, annotated, _ = processor.process_frame(frame)

        if norm_feats:
            pred = model_manager.predict(norm_feats)
            sign = pred.get("smoothed_sign") or "..."
            conf = pred.get("confidence", 0.0)
            cv2.putText(annotated, f"{sign} ({int(conf * 100)}%)", (24, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 3)

        ret, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint for server-side camera."""
    return Response(gen_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "classes": model_manager.classes,
        "total_classes": len(model_manager.classes)
    })



   

@app.route("/firebase-test")
def firebase_test():
    if db is None:
        return jsonify({
            "success": False,
            "error": "Firebase service account not configured on this environment"
        }), 503

    try:
        db.collection("_connection_test").document("status").get()

        return jsonify({
            "success": True,
            "message": "Firebase Firestore connected successfully!"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/send-message", methods=["POST"])
def send_message():
    if db is None:
        return jsonify({
            "success": False,
            "error": "Firebase messaging is not configured on this environment"
        }), 503

    try:
        # Verify Firebase login
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({"success": False, "error": "Login required"}), 401

        id_token = auth_header.split(" ", 1)[1]
        decoded_token = auth.verify_id_token(id_token)

        sender_id = decoded_token["uid"]
        sender_email = decoded_token.get("email", "")

        # Read message details
        data = request.get_json(silent=True) or {}

        recipient_email = data.get("recipientEmail", "").strip().lower()
        message_text = data.get("text", "").strip()

        if not recipient_email or not message_text:
            return jsonify({
                "success": False,
                "error": "Recipient email and message are required"
            }), 400

        if len(message_text) > 1000:
            return jsonify({
                "success": False,
                "error": "Message must be 1000 characters or fewer"
            }), 400

        if recipient_email == sender_email.lower():
            return jsonify({
                "success": False,
                "error": "You cannot send a message to yourself"
            }), 400

        # Find recipient in Firestore
        profiles = (
            db.collection("publicProfiles")
            .where("email", "==", recipient_email)
            .limit(1)
            .stream()
        )

        recipient = next(profiles, None)

        if recipient is None:
            return jsonify({
                "success": False,
                "error": "Recipient not found"
            }), 404

        recipient_id = recipient.id

        # Save message using the Android app's existing format
        message_ref = db.collection("messages").document()

        message_ref.set({
            "senderId": sender_id,
            "senderEmail": sender_email,
            "recipientId": recipient_id,
            "recipientEmail": recipient_email,
            "text": message_text,
            "createdAt": firestore.SERVER_TIMESTAMP
        })
       # Send push notification to recipient
        try:
            recipient_user = db.collection("users").document(recipient_id).get()

            if recipient_user.exists:
                fcm_token = recipient_user.to_dict().get("fcmToken")

                if fcm_token:
                    notification = messaging.Message(
                        notification=messaging.Notification(
                            title="New SignAI Message",
                            body=f"{sender_email} sent you a message."
                        ),
                        token=fcm_token
                    )

                    messaging.send(notification)
                    print("Push notification sent successfully.")

        except Exception as notification_error:
            print("Push notification error:", str(notification_error))


        return jsonify({
            "success": True,
            "message": "Message sent successfully"
        })

    except Exception as e:
        print("Message sending error:", str(e))

        return jsonify({
            "success": False,
            "error": "Unable to send message"
        }), 500




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
   
