from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import json
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
    alerts_col      = db["alerts"]
    users_col       = db["users"]  # Store FCM tokens here
    print("✅ MongoDB connected successfully")
except Exception as e:
    raise RuntimeError(f"MongoDB connection failed: {e}")

# ---------------- FIREBASE SETUP ----------------
try:
    firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS")
    if not firebase_creds_json:
        raise RuntimeError("FIREBASE_CREDENTIALS env variable not set")

    cred_dict = json.loads(firebase_creds_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase initialized successfully")
except Exception as e:
    raise RuntimeError(f"Firebase initialization failed: {e}")


# ---------------- HELPER: SEND FCM NOTIFICATION ----------------
def send_fcm_alert(fcm_token: str, title: str, body: str, data: dict = {}) -> dict:
    print(f"\n📤 Sending FCM notification to token: {fcm_token[:20]}...")

    if not fcm_token:
        return {"success": False, "error": "No FCM token provided"}

    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data={k: str(v) for k, v in data.items()},  # All values must be strings
            token=fcm_token,
            android=messaging.AndroidConfig(
                priority="high",  # High priority = immediate delivery
                notification=messaging.AndroidNotification(
                    sound="default",
                    priority="high",
                    channel_id="emergency_alerts"  # Match this in your Android app
                )
            ),
        )

        response = messaging.send(message)
        print(f"✅ FCM notification sent: {response}")
        return {"success": True, "message_id": response}

    except messaging.UnregisteredError:
        msg = "FCM token is no longer valid (app uninstalled or token expired)"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    except messaging.InvalidArgumentError as e:
        msg = f"Invalid FCM argument: {str(e)}"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    except Exception as e:
        msg = f"FCM error: {str(e)}"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}


# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "CardioGuard Backend Running ✅",
        "database": "MongoDB Atlas connected",
        "notifications": "Firebase FCM ready"
    })


# ---------------- REGISTER FCM TOKEN ----------------
# Call this from Android when app starts or token refreshes
@app.route("/register-token", methods=["POST"])
def register_token():
    try:
        data = request.get_json()
        patient_email = data.get("email", "").strip()
        fcm_token     = data.get("fcm_token", "").strip()

        if not patient_email or not fcm_token:
            return jsonify({"error": "email and fcm_token are required"}), 400

        # Upsert — update token if user exists, insert if not
        users_col.update_one(
            {"email": patient_email},
            {"$set": {
                "email":     patient_email,
                "fcm_token": fcm_token,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )

        print(f"✅ FCM token registered for {patient_email}")
        return jsonify({"status": "Token registered successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- PREDICTION ENDPOINT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        patient = data.get("patient")
        medical = data.get("medical")

        if not patient or not medical:
            return jsonify({"error": "Missing patient or medical data"}), 400

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

        X        = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction  = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        risk       = "HIGH RISK" if prediction == 1 else "LOW RISK"
        confidence = round(probability * 100, 2)

        predictions_col.insert_one({
            "patient":    patient,
            "medical":    medical,
            "risk":       risk,
            "confidence": confidence,
            "timestamp":  datetime.utcnow()
        })

        return jsonify({"risk": risk, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------- SEND ALERT ENDPOINT ----------------
@app.route("/send-alert", methods=["POST"])
def send_alert():
    try:
        data = request.get_json()
        print(f"\n🚨 /send-alert called with data: {data}")

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        patient        = data.get("patient", {})
        risk           = data.get("risk", "HIGH RISK")
        confidence     = data.get("confidence", "N/A")
        relation_email = data.get("relation_email", "").strip()

        patient_name  = patient.get("name",   "Unknown Patient")
        patient_age   = patient.get("age",    "N/A")
        gender_val    = patient.get("gender", -1)
        gender_str    = "Male" if gender_val == 1 else "Female" if gender_val == 0 else "N/A"
        patient_email = patient.get("email",  "")
        district      = patient.get("district", "N/A")
        state         = patient.get("state",    "N/A")

        results = []

        # ── 1. Notify the PATIENT ──────────────────────────────────────────
        patient_user = users_col.find_one({"email": patient_email})
        if patient_user and patient_user.get("fcm_token"):
            result = send_fcm_alert(
                fcm_token=patient_user["fcm_token"],
                title="🚨 HIGH Cardiac Risk Detected!",
                body=f"Your assessment shows {confidence}% HIGH risk. Please seek immediate medical attention.",
                data={
                    "risk":       risk,
                    "confidence": str(confidence),
                    "type":       "emergency_alert"
                }
            )
            results.append({"target": "patient", **result})
        else:
            print(f"⚠️ No FCM token found for patient: {patient_email}")
            results.append({"target": "patient", "success": False, "error": "No FCM token registered"})

        # ── 2. Notify the EMERGENCY CONTACT ───────────────────────────────
        if relation_email:
            relation_user = users_col.find_one({"email": relation_email})
            if relation_user and relation_user.get("fcm_token"):
                result = send_fcm_alert(
                    fcm_token=relation_user["fcm_token"],
                    title=f"🚨 Emergency: {patient_name} needs help!",
                    body=f"{patient_name} ({patient_age}y, {gender_str}) from {district}, {state} has HIGH cardiac risk ({confidence}%). Immediate attention required!",
                    data={
                        "risk":           risk,
                        "confidence":     str(confidence),
                        "patient_name":   patient_name,
                        "patient_email":  patient_email,
                        "type":           "emergency_alert"
                    }
                )
                results.append({"target": "emergency_contact", **result})
            else:
                print(f"⚠️ No FCM token for emergency contact: {relation_email}")
                results.append({"target": "emergency_contact", "success": False, "error": "No FCM token registered"})

        # ── 3. Log to MongoDB ──────────────────────────────────────────────
        alerts_col.insert_one({
            "patient":        patient,
            "risk":           risk,
            "confidence":     confidence,
            "relation_email": relation_email,
            "results":        results,
            "timestamp":      datetime.utcnow()
        })

        any_success = any(r.get("success") for r in results)

        if any_success:
            return jsonify({
                "status":  "sent",
                "results": results,
                "message": "Emergency alert delivered via push notification"
            })
        else:
            return jsonify({
                "status":  "failed",
                "results": results,
                "error":   "No notifications delivered — check FCM tokens are registered"
            }), 500

    except Exception as e:
        print(f"❌ /send-alert exception: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
