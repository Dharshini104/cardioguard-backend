from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
from datetime import datetime
from pymongo import MongoClient

import firebase_admin
from firebase_admin import credentials, messaging

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- LOAD ML MODEL ----------------
try:
    model = joblib.load("cardiac_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ---------------- MONGODB CONNECTION ----------------
MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable not set")

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.admin.command("ping")

    db = mongo_client["cardioguard"]
    predictions_col = db["predictions"]
    alerts_col = db["alerts"]

    print("✅ MongoDB connected")

except Exception as e:
    raise RuntimeError(f"MongoDB connection failed: {e}")

# ---------------- FIREBASE CONFIG ----------------
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    print("✅ Firebase initialized")
except Exception as e:
    raise RuntimeError(f"Firebase initialization failed: {e}")

# ---------------- SEND PUSH NOTIFICATION ----------------
def send_push_notification(token, title, body):

    try:

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            token=token
        )

        response = messaging.send(message)

        print("✅ Firebase notification sent:", response)

        return {"success": True}

    except Exception as e:

        print("❌ Firebase notification failed:", e)

        return {"success": False, "error": str(e)}

# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "CardioGuard Backend Running",
        "database": "MongoDB connected",
        "firebase": "Initialized"
    })

# ---------------- PREDICTION ENDPOINT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        patient = data.get("patient")
        medical = data.get("medical")

        features = [
            float(patient["age"]),
            float(patient["gender"]),
            float(medical["heart_rate"]),
            float(medical["systolic_bp"]),
            float(medical["diastolic_bp"]),
            float(medical["blood_sugar"]),
            float(medical["ck_mb"]),
            float(medical["troponin"])
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        risk = "HIGH RISK" if prediction == 1 else "LOW RISK"
        confidence = round(probability * 100, 2)

        predictions_col.insert_one({
            "patient": patient,
            "medical": medical,
            "risk": risk,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        })

        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({"error": str(e)}), 400

# ---------------- SEND ALERT ENDPOINT ----------------
@app.route("/send-alert", methods=["POST"])
def send_alert():

    try:

        data = request.get_json()

        patient = data.get("patient", {})
        risk = data.get("risk")
        confidence = data.get("confidence")
        device_token = data.get("device_token")

        if not device_token:
            return jsonify({"error": "Device token missing"}), 400

        patient_name = patient.get("name", "Unknown")

        title = "🚨 CardioGuard Emergency Alert"

        body = f"""
Patient: {patient_name}
Risk Level: {risk}
Confidence: {confidence}%

Immediate medical attention required.
"""

        result = send_push_notification(device_token, title, body)

        alerts_col.insert_one({
            "patient": patient,
            "risk": risk,
            "confidence": confidence,
            "device_token": device_token,
            "sent": result["success"],
            "timestamp": datetime.utcnow()
        })

        if result["success"]:
            return jsonify({
                "status": "sent"
            })

        else:
            return jsonify({
                "status": "error",
                "error": result["error"]
            }), 500

    except Exception as e:

        return jsonify({"error": str(e)}), 500

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)