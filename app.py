from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

# ---------------- MONGODB IMPORT ----------------
from pymongo import MongoClient

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("cardiac_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- MONGODB ATLAS CONNECTION ----------------
# MONGO_URI must be added in Render Environment Variables
MONGO_URI = os.environ.get("MONGO_URI")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["cardioguard"]
predictions_col = db["predictions"]

# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CardioGuard Backend Running ✅"})

# ---------------- PREDICTION API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        patient = data["patient"]
        medical = data["medical"]

        # --------- FEATURE EXTRACTION (ORDER MATTERS) ---------
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

        # --------- MODEL PREDICTION ---------
        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0][1])

        risk = "HIGH RISK" if pred == 1 else "LOW RISK"
        confidence = round(prob * 100, 2)

        # --------- STORE RESULT IN MONGODB ---------
        record = {
            "patient": patient,
            "medical": medical,
            "risk": risk,
            "confidence": confidence
        }

        predictions_col.insert_one(record)

        # --------- RESPONSE ---------
        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
