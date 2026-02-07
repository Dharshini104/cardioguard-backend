from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL ----------------
model = joblib.load("cardiac_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- MONGODB ATLAS ----------------
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://cardioguard_user:Srec123@cardioguard-cluster.f7hi6um.mongodb.net/cardioguard?retryWrites=true&w=majority"
)

client = MongoClient(MONGO_URI)
db = client["cardioguard"]
predictions = db["predictions"]

# ---------------- HOME ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CardioGuard Backend Running"})

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        patient = data["patient"]
        medical = data["medical"]

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

        risk = "HIGH RISK" if pred == 1 else "LOW RISK"
        confidence = round(prob * 100, 2)

        # ---------- SAVE TO MONGODB ----------
        record = {
            "patient": patient,
            "medical": medical,
            "risk": risk,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }

        predictions.insert_one(record)

        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------- HISTORY ----------------
@app.route("/history", methods=["GET"])
def history():
    data = list(predictions.find({}, {"_id": 0}).sort("timestamp", -1))
    return jsonify(data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
