from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from pymongo import MongoClient
from datetime import datetime

# ---------------- APP INIT ----------------
app = Flask(__name__)
CORS(app)

# ---------------- MONGODB ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["cardioguard"]
predictions = db["predictions"]

# ---------------- LOAD MODEL ----------------
model = joblib.load("cardiac_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HOME ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CardioGuard Backend Running"})

# ---------------------------------------------------
@app.route("/patient", methods=["POST"])
def save_patient():
    data = request.json

    patient = {
        "name": data["name"],
        "gender": data["gender"],
        "created_at": datetime.utcnow()
    }

    db.patients.insert_one(patient)
    return jsonify({"status": "saved"})

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📥 Prediction request received")

        data = request.json
        patient = data["patient"]
        medical = data["medical"]

        # FEATURE ORDER MUST MATCH TRAINING
        features = [
            patient["age"],
            patient["gender"],
            medical["heart_rate"],
            medical["systolic_bp"],
            medical["diastolic_bp"],
            medical["blood_sugar"],
            medical["ck_mb"],
            medical["troponin"]
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0][pred])

        risk = "HIGH RISK" if pred == 1 else "NEGATIVE"
        confidence = round(prob * 100, 2)

        # ---------------- SAVE TO MONGODB ----------------
        record = {
            "age": patient["age"],
            "gender": patient["gender"],
            "heart_rate": medical["heart_rate"],
            "systolic_bp": medical["systolic_bp"],
            "diastolic_bp": medical["diastolic_bp"],
            "blood_sugar": medical["blood_sugar"],
            "ck_mb": medical["ck_mb"],
            "troponin": medical["troponin"],
            "risk": risk,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }

        predictions.insert_one(record)

        print("✅ Prediction saved to MongoDB")

        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 400

# ---------------- HISTORY ----------------
@app.route("/history", methods=["GET"])
def history():
    data = list(predictions.find({}, {"_id": 0}).sort("timestamp", -1))
    return jsonify(data)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
