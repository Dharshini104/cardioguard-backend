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
    alerts_col      = db["alerts"]
    print("✅ MongoDB connected successfully")
except Exception as e:
    raise RuntimeError(f"MongoDB connection failed: {e}")

# ---------------- EMAIL CONFIG ----------------
GMAIL_USER = os.environ.get("GMAIL_USER", "").strip()
GMAIL_PASS = os.environ.get("GMAIL_PASS", "").strip()

# Print at startup so you can see in Render logs whether they loaded
print(f"📧 GMAIL_USER loaded: {'YES → ' + GMAIL_USER if GMAIL_USER else 'NO ❌ (not set)'}")
print(f"🔑 GMAIL_PASS loaded: {'YES (hidden)' if GMAIL_PASS else 'NO ❌ (not set)'}")

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
    print(f"\n📤 Attempting to send email...")
    print(f"   From    : {GMAIL_USER}")
    print(f"   To      : {to_addresses}")
    print(f"   Subject : {subject}")

    if not GMAIL_USER:
        msg = "GMAIL_USER is not set in environment variables"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    if not GMAIL_PASS:
        msg = "GMAIL_PASS is not set in environment variables"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    if len(GMAIL_PASS.replace(" ", "")) != 16:
        msg = f"GMAIL_PASS looks wrong — should be 16 chars, got {len(GMAIL_PASS.replace(' ', ''))}. Use a Gmail App Password, not your login password."
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    try:
        msg = MIMEMultipart()
        msg["From"]    = GMAIL_USER
        msg["To"]      = ", ".join(to_addresses)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        print("   Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            print("   Connected. Logging in...")
            server.login(GMAIL_USER, GMAIL_PASS)
            print("   Logged in. Sending...")
            server.sendmail(GMAIL_USER, to_addresses, msg.as_string())

        print(f"✅ Email sent successfully to {to_addresses}")
        return {"success": True}

    except smtplib.SMTPAuthenticationError as e:
        msg = (
            "Gmail authentication failed (535). "
            "Make sure: (1) 2-Step Verification is ON in your Google account, "
            "(2) You created an App Password at myaccount.google.com/apppasswords, "
            "(3) GMAIL_PASS is the 16-char App Password without spaces, "
            "(4) GMAIL_USER is the exact Gmail address you created the App Password for."
        )
        print(f"❌ SMTPAuthenticationError: {e}\n   → {msg}")
        return {"success": False, "error": msg}

    except smtplib.SMTPRecipientsRefused as e:
        msg = f"Recipient refused: {e}"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    except smtplib.SMTPException as e:
        msg = f"SMTP error: {str(e)}"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    except OSError as e:
        msg = f"Network/connection error reaching smtp.gmail.com: {str(e)}"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}

    except Exception as e:
        msg = f"Unexpected error: {str(e)}"
        print(f"❌ {msg}")
        return {"success": False, "error": msg}


# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "CardioGuard Backend Running ✅",
        "database": "MongoDB Atlas connected",
        "gmail_user_set": bool(GMAIL_USER),
        "gmail_pass_set": bool(GMAIL_PASS),
        "gmail_pass_length": len(GMAIL_PASS.replace(" ", "")) if GMAIL_PASS else 0
    })


# ---------------- EMAIL TEST ENDPOINT ----------------
# Call this from browser to test email without needing the app:
# GET https://cardioguard-backend-9g9x.onrender.com/test-email?to=youremail@gmail.com
@app.route("/test-email", methods=["GET"])
def test_email():
    to = request.args.get("to", "").strip()
    if not to:
        return jsonify({"error": "Provide ?to=youremail@gmail.com in the URL"}), 400

    result = send_email(
        to_addresses=[to],
        subject="✅ CardioGuard Email Test",
        body=(
            "This is a test email from your CardioGuard backend.\n\n"
            "If you received this, email sending is working correctly!\n\n"
            f"Sent at: {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
        )
    )

    if result["success"]:
        return jsonify({"status": "✅ Email sent successfully", "to": to})
    else:
        return jsonify({"status": "❌ Email failed", "error": result["error"]}), 500


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
        android_hospital_email = data.get("hospital_email", "").strip()

        # Resolve hospital email from district
        district       = patient.get("district", "")
        hospital_email = HOSPITAL_EMAILS.get(district, android_hospital_email or DEFAULT_HOSPITAL_EMAIL)

        # Build recipients — deduplicated, no empty strings
        recipients = list({e for e in [relation_email, hospital_email] if e and "@" in e})
        print(f"   Recipients resolved: {recipients}")

        if not recipients:
            return jsonify({"error": "No valid recipient email addresses"}), 400

        patient_name  = patient.get("name",    "Unknown Patient")
        patient_age   = patient.get("age",     "N/A")
        gender_val    = patient.get("gender",  -1)
        gender_str    = "Male" if gender_val == 1 else "Female" if gender_val == 0 else "N/A"
        patient_email = patient.get("email",   "N/A")
        state         = patient.get("state",   "N/A")
        timestamp     = datetime.now().strftime("%d %b %Y, %I:%M %p")

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

Immediate medical evaluation is strongly recommended.
Please contact the patient urgently at:
Email : {patient_email}

{'='*45}
Generated by CardioGuard App
Date: {timestamp}
{'='*45}
        """.strip()

        result = send_email(recipients, subject, body)

        # Always log to MongoDB — even on failure
        alerts_col.insert_one({
            "patient":       patient,
            "risk":          risk,
            "confidence":    confidence,
            "relation_email": relation_email,
            "hospital_email": hospital_email,
            "recipients":    recipients,
            "email_sent":    result["success"],
            "email_error":   result.get("error"),
            "timestamp":     datetime.utcnow()
        })

        if result["success"]:
            return jsonify({
                "status":     "sent",
                "recipients": recipients,
                "message":    f"Alert sent to {len(recipients)} recipient(s)"
            })
        else:
            # Return 200 with error detail so Android knows what went wrong
            return jsonify({
                "status": "error",
                "error":  result["error"]
            }), 500

    except Exception as e:
        print(f"❌ /send-alert exception: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)