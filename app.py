from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("cardiac_model_real.pkl")
scaler = joblib.load("scaler_real.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CardioGuard Backend Running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        patient = data["patient"]
        medical = data["medical"]

        # ✅ FORCE NUMERIC CONVERSION (CRITICAL)
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

        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # ✅ CORRECT PREDICTION
        pred = int(model.predict(X_scaled)[0])

        # ✅ ALWAYS TAKE PROBABILITY OF CLASS 1 (HIGH RISK)
        prob = float(model.predict_proba(X_scaled)[0][1])

        risk = "HIGH RISK" if pred == 1 else "LOW RISK"
        confidence = round(prob * 100, 2)

        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
