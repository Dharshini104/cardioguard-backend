from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from pymongo import MongoClient

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
    print("✅ MongoDB connected successfully")
except Exception as e:
    raise RuntimeError(f"MongoDB connection failed: {e}")

# ---------------- EMAIL CONFIG ----------------
# Set these in your Render environment variables:
#   GMAIL_USER  →  your_gmail@gmail.com
#   GMAIL_PASS  →  your 16-char Gmail App Password (not your login password)
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_PASS = os.environ.get("GMAIL_PASS", "")

# ---------------- HOSPITAL EMAIL MAP ----------------
HOSPITAL_EMAILS = {
    "Tiruppur":        "tiruppur.govthospital@tnhealth.in",
    "Coimbatore":      "coimbatore.ghospital@tnhealth.in",
    "Chennai":         "chennai.rajivgandhi@tnhealth.in",
    "Madurai":         "madurai.rajajihospital@tnhealth.in",
    "Salem":           "salem.govthospital@tnhealth.in",
    "Erode":           "erode.govthospital@tnhealth.in",
    "Vellore":         "vellore.govthospital@tnhealth.in",
    "Tiruchirappalli": "trichy.maraimalai@tnhealth.in",
    "Thanjavur":       "thanjavur.govthospital@tnhealth.in",
    "Pune":            "pune.sassoon@maharashtra.gov.in",
    "Mumbai City":     "mumbai.kem@maharashtra.gov.in",
    "Nagpur":          "nagpur.mayo@maharashtra.gov.in",
    "Hyderabad":       "hyderabad.osmania@telangana.gov.in",
    "Bengaluru Urban": "bangalore.victoria@karnataka.gov.in",
}
DEFAULT_HOSPITAL_EMAIL = "emergency@cardioguard.in"

# ---------------- HELPER: SEND EMAIL ----------------
def send_email(to_addresses: list, subject: str, body: str) -> dict:
    """
    Sends a plain-text email via Gmail SMTP SSL.
    Returns {"success": True} or {"success": False, "error": "..."}
    """
    if not GMAIL_USER or not GMAIL_PASS:
        return {"success": False, "error": "Gmail credentials not configured in environment variables"}

    try:
        msg = MIMEMultipart()
        msg["From"]    = GMAIL_USER
        msg["To"]      = ", ".join(to_addresses)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, to_addresses, msg.as_string())

        print(f"✅ Email sent to: {to_addresses}")
        return {"success": True}

    except smtplib.SMTPAuthenticationError:
        error = "Gmail authentication failed. Check GMAIL_USER and GMAIL_PASS (use App Password, not login password)"
        print(f"❌ {error}")
        return {"success": False, "error": error}

    except smtplib.SMTPException as e:
        error = f"SMTP error: {str(e)}"
        print(f"❌ {error}")
        return {"success": False, "error": error}

    except Exception as e:
        error = f"Unexpected error: {str(e)}"
        print(f"❌ {error}")
        return {"success": False, "error": error}


# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "CardioGuard Backend Running ✅",
        "database": "MongoDB Atlas connected",
        "email_configured": bool(GMAIL_USER and GMAIL_PASS)
    })


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

        # Feature order MUST match training order
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

        # Store in MongoDB
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

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        patient        = data.get("patient", {})
        risk           = data.get("risk", "HIGH RISK")
        confidence     = data.get("confidence", "N/A")
        relation_email = data.get("relation_email", "").strip()
        # Android also sends hospital_email it already resolved — use it as fallback
        android_hospital_email = data.get("hospital_email", "").strip()

        # Resolve hospital email from district map (backend is authoritative)
        district = patient.get("district", "")
        hospital_email = HOSPITAL_EMAILS.get(district, android_hospital_email or DEFAULT_HOSPITAL_EMAIL)

        # Build recipient list — skip empty/missing emails
        recipients = list({e for e in [relation_email, hospital_email] if e})

        if not recipients:
            return jsonify({"error": "No valid recipient emails provided"}), 400

        # Validate that we have the minimum patient info
        patient_name = patient.get("name", "Unknown Patient")
        patient_age  = patient.get("age", "N/A")
        gender_val   = patient.get("gender", -1)
        gender_str   = "Male" if gender_val == 1 else "Female" if gender_val == 0 else "N/A"
        patient_email = patient.get("email", "N/A")
        state        = patient.get("state", "N/A")
        timestamp    = datetime.now().strftime("%d %b %Y, %I:%M %p")

        subject = f"🚨 URGENT: High Cardiac Risk Alert — {patient_name}"

        body = f"""
⚠️  CARDIAC RISK ALERT  ⚠️
{'='*45}

PATIENT DETAILS
{'─'*45}
Name     : {patient_name}
Age      : {patient_age} years
Gender   : {gender_str}
Email    : {patient_email}
Location : {district}, {state}

ASSESSMENT RESULTS
{'─'*45}
Risk Level  : {risk}
Confidence  : {confidence}%

⚠️  IMMEDIATE ACTION REQUIRED  ⚠️
{'─'*45}
This patient has been assessed with HIGH cardiac
risk by the CardioGuard system.

Immediate medical evaluation and intervention
is strongly recommended.

Please contact the patient urgently:
  Email : {patient_email}

{'='*45}
Generated by CardioGuard App
Date: {timestamp}
{'='*45}
        """.strip()

        # Send the email
        result = send_email(recipients, subject, body)

        # Log alert to MongoDB regardless of email success
        alerts_col.insert_one({
            "patient": patient,
            "risk": risk,
            "confidence": confidence,
            "relation_email": relation_email,
            "hospital_email": hospital_email,
            "recipients": recipients,
            "email_sent": result["success"],
            "email_error": result.get("error"),
            "timestamp": datetime.utcnow()
        })

        if result["success"]:
            return jsonify({
                "status": "sent",
                "recipients": recipients,
                "message": f"Emergency alert sent to {len(recipients)} recipient(s)"
            })
        else:
            return jsonify({
                "status": "error",
                "error": result["error"],
                "recipients_attempted": recipients
            }), 500

    except Exception as e:
        print(f"❌ /send-alert exception: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
