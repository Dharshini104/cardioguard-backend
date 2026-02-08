from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

# ---------- MONGODB ----------
from pymongo import MongoClient

# ---------- APP SETUP ----------
app = Flask(__name__)
CORS(app)

# ---------- LOAD ML MODEL ----------
model = joblib.load("cardiac_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------- MONGODB ATLAS CONNECTION ----------
# DO NOT hardcode the URI
MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable not set")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["cardioguard"]
predictions_col = db["predictions"]

# ---------- HEALTH CHECK ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "CardioGuard Backend Running ✅",
        "database": "MongoDB Atlas connected"
    })

# ---------- PREDICTION ENDPOINT ----------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # --------- INPUT DATA ---------
        patient = data["patient"]
        medical = data["medical"]

        # --------- FEATURE ORDER (VERY IMPORTANT) ---------
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
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        risk = "HIGH RISK" if prediction == 1 else "LOW RISK"
        confidence = round(probability * 100, 2)

        # --------- STORE IN MONGODB ATLAS ---------
        predictions_col.insert_one({
            "patient": patient,
            "medical": medical,
            "risk": risk,
            "confidence": confidence
        })

        # --------- RESPONSE ---------
        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# ---------- RUN SERVER ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
