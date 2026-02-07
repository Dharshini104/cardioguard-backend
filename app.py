from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
from pymongo import MongoClient
from datetime import datetime

# ------------------- APP SETUP -------------------
app = Flask(__name__)
CORS(app)

# ------------------- LOAD MODEL -------------------
model = joblib.load("cardiac_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------- MONGODB -------------------
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://card:Srec123@cluster0.mongodb.net/cardioguard?retryWrites=true&w=majority"
)

client = MongoClient(MONGO_URI)
db = client["cardioguard"]
predictions = db["predictions"]

# ------------------- HEALTH CHECK -------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CardioGuard Backend Running ✅"})

# ------------------- PREDICT -------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        patient = data["patient"]
        medical = data["medical"]

        # 🔥 FORCE NUMERIC CONVERSION (CRITICAL)
        features = [
            float(patient["age"]),
            float(patient["gender"]),          # 0 = Female, 1 = Male
            float(medical["heart_rate"]),
            float(medical["systolic_bp"]),
            float(medical["diastolic_bp"]),
            float(medical["blood_sugar"]),
            float(medical["ck_mb"]),
            float(medical["troponin"])
        ]

        # 🔥 SAME ORDER AS TRAINING
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # 🔥 CORRECT PREDICTION LOGIC
        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0][1])

        risk = "HIGH RISK" if pred == 1 else "LOW RISK"
        confidence = round(prob * 100, 2)

        # ------------------- SAVE TO DB -------------------
        record = {
            "patient": patient,
            "medical": medical,
            "risk": risk,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }

        predictions.insert_one(record)

        # ------------------- RESPONSE -------------------
        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- HISTORY -------------------
@app.route("/history", methods=["GET"])
def history():
    data = list(predictions.find({}, {"_id": 0}).sort("timestamp", -1))
    return jsonify(data)

# ------------------- RUN -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
