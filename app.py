from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("cardiac_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "CardioGuard Backend Running"})

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

        return jsonify({
            "risk": risk,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
