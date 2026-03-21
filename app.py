from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from models import db, User, PatientDetail, Prediction, Appointment, Prescription
from health_score import calculate_health_score
import pickle, os, datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # .env file se automatically GROQ_API_KEY load hogi

app = Flask(__name__)
app.secret_key = "medintel_secret_key_2024"
database_url = os.environ.get("DATABASE_URL", "sqlite:///database.db")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
with app.app_context():
    db.create_all()
    # Create admin if not exists
    from models import User
    try:
        admin = User.query.filter_by(role="admin").first()
        if not admin:
            admin = User(
                name="Admin",
                email="admin@medintel.com",
                role="admin",
                specialization="General Physician"
            )
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()
    except:
        pass

# ─── ML Model Training ────────────────────────────────────────────────────────

def train_heart_model():
    print("Training heart disease model...")
    df = pd.read_csv("heart.csv")
    X  = df.drop("target", axis=1)
    y  = df["target"]
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    os.makedirs("ml_models", exist_ok=True)
    with open("ml_models/heart_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Heart model trained and saved.")
    return model

def train_diabetes_model():
    """Train diabetes classifier using synthetic Pima-style data."""
    print("Training diabetes model...")
    np.random.seed(42)
    n = 768

    pregnancies    = np.random.poisson(3.8, n)
    glucose        = np.where(np.random.rand(n) < 0.35,
                              np.random.normal(140, 25, n),
                              np.random.normal(100, 20, n)).clip(40, 200)
    blood_pressure = np.random.normal(69, 19, n).clip(0, 130)
    skin_thickness = np.random.normal(20, 16, n).clip(0, 99)
    insulin        = np.random.exponential(80, n).clip(0, 846)
    bmi            = np.random.normal(32, 7, n).clip(10, 70)
    dpf            = np.random.exponential(0.47, n).clip(0.08, 2.42)
    age            = np.random.normal(33, 11, n).clip(21, 81)

    X = np.column_stack([pregnancies, glucose, blood_pressure,
                         skin_thickness, insulin, bmi, dpf, age])

    prob = (
        0.4 * (glucose > 125) +
        0.2 * (bmi > 30) +
        0.15 * (age > 40) +
        0.1  * (dpf > 0.5) +
        0.1  * (blood_pressure > 80) +
        0.05 * np.random.rand(n)
    )
    y = (prob > 0.5).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    os.makedirs("ml_models", exist_ok=True)
    with open("ml_models/diabetes_model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)

    print("✅ Diabetes model trained and saved.")
    return model, scaler

def load_model(name):
    path = os.path.join("ml_models", name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            return pickle.load(f)
    if os.path.exists(name) and os.path.getsize(name) > 0:
        with open(name, "rb") as f:
            return pickle.load(f)
    if name == "heart_model.pkl":
        return train_heart_model()
    if name == "diabetes_model.pkl":
        return train_diabetes_model()
    return None

heart_model                      = load_model("heart_model.pkl")
diabetes_payload                 = load_model("diabetes_model.pkl")
diabetes_model, diabetes_scaler  = diabetes_payload if isinstance(diabetes_payload, tuple) else (None, None)

# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email    = request.form["email"].strip().lower()
        existing = User.query.filter_by(email=email).first()
        if existing:
            return render_template("register.html", error="Email already registered.")

        role = request.form.get("role", "patient")
        if role not in ("patient", "doctor"):
            role = "patient"

        spec = request.form.get("specialization", "General Physician") if role == "doctor" else "General Physician"

        user = User(
            name           = request.form["name"].strip(),
            email          = email,
            role           = role,
            specialization = spec
        )
        user.set_password(request.form["password"])
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        user  = User.query.filter_by(email=email).first()
        if user and user.check_password(request.form["password"]):
            session["user_id"] = user.id
            session["role"]    = user.role
            session["name"]    = user.name.title() if user.name else "User"
            session["email"]   = user.email
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ─── Dashboard — Role Based ───────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = db.session.get(User, session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    # Admin dashboard
    if user.role == "admin":
        return redirect(url_for("admin"))

    # Doctor dashboard
    if user.role == "doctor":
        patients     = User.query.filter_by(role="patient").all()
        appointments = Appointment.query.filter_by(doctor_id=user.id).all()
        pending      = [a for a in appointments if a.status == "Pending"]
        confirmed    = [a for a in appointments if a.status == "Confirmed"]

        patient_data = []
        for p in patients:
            det   = PatientDetail.query.filter_by(user_id=p.id).first()
            score = calculate_health_score(det) if det else None
            patient_data.append({"user": p, "details": det, "score": score})

        return render_template("doctor_dashboard.html",
            user           = user,
            patient_data   = patient_data,
            appointments   = appointments,
            pending        = pending,
            confirmed      = confirmed,
            total_patients = len(patients)
        )

    # Patient dashboard
    details = PatientDetail.query.filter_by(user_id=user.id).first()
    score   = calculate_health_score(details) if details else None
    return render_template("dashboard.html", user=user, details=details, score=score)

# ─── Health Data ──────────────────────────────────────────────────────────────

@app.route("/health-data", methods=["GET", "POST"])
def health_data():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        f       = request.form
        details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
        if not details:
            details = PatientDetail(user_id=session["user_id"])
            db.session.add(details)
        try:
            details.age      = int(f["age"])
            details.weight   = float(f["weight"])
            details.height   = float(f["height"])
            details.bp       = f["bp"]
            details.sugar    = float(f["sugar"])
            details.smoking  = f.get("smoking", "no")
            details.activity = f.get("activity", "moderate")
            db.session.commit()
        except (ValueError, KeyError):
            pass
        return redirect(url_for("dashboard"))
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    return render_template("health_data.html", details=details)

# ─── BMI ──────────────────────────────────────────────────────────────────────

@app.route("/bmi")
def bmi():
    if "user_id" not in session:
        return redirect(url_for("login"))
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    bmi_val = None
    if details and details.height and details.weight and details.height > 0:
        h       = details.height / 100
        bmi_val = round(details.weight / (h * h), 1)
    return render_template("bmi.html", bmi=bmi_val, details=details)

# ─── AI Prediction (Heart + Diabetes) ────────────────────────────────────────

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user_id" not in session:
        return redirect(url_for("login"))
    result = {}
    if request.method == "POST":
        f = request.form

        def fi(key, default=0):
            v = f.get(key, "").strip()
            try: return int(float(v)) if v else default
            except: return default

        def ff(key, default=0.0):
            v = f.get(key, "").strip()
            try: return float(v) if v else default
            except: return default

        heart_features = [
            fi("age", 40),       fi("sex", 1),       fi("cp", 0),
            fi("trestbps", 120), fi("chol", 200),    fi("fbs", 0),
            fi("restecg", 0),    fi("thalach", 150), fi("exang", 0),
            ff("oldpeak", 0.0),  fi("slope", 1),     fi("ca", 0),
            fi("thal", 1)
        ]

        diabetes_features = [
            fi("pregnancies", 0),
            ff("glucose", 100.0),
            ff("blood_pressure", 70.0),
            ff("skin_thickness", 20.0),
            ff("insulin", 80.0),
            ff("bmi_val", 25.0),
            ff("dpf", 0.5),
            fi("age", 40)
        ]

        heart_risk = heart_prob = 0
        if heart_model:
            heart_risk = int(heart_model.predict([heart_features])[0])
            heart_prob = round(heart_model.predict_proba([heart_features])[0][1] * 100, 1)

        diabetes_risk = diabetes_prob = 0
        if diabetes_model and diabetes_scaler:
            df_scaled     = diabetes_scaler.transform([diabetes_features])
            diabetes_risk = int(diabetes_model.predict(df_scaled)[0])
            diabetes_prob = round(diabetes_model.predict_proba(df_scaled)[0][1] * 100, 1)

        pred = Prediction(
            user_id       = session["user_id"],
            heart_risk    = heart_risk,
            diabetes_risk = diabetes_risk
        )
        db.session.add(pred)
        db.session.commit()

        result = {
            "heart_risk":    heart_risk,    "heart_prob":    heart_prob,
            "diabetes_risk": diabetes_risk, "diabetes_prob": diabetes_prob
        }
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    return render_template("prediction.html", result=result, details=details)

# ─── Chatbot (Groq AI) ────────────────────────────────────────────────────────

@app.route("/chatbot")
def chatbot():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("chatbot.html")

@app.route("/chatbot/respond", methods=["POST"])
def chatbot_respond():
    data    = request.json
    msg     = data.get("message", "").strip()
    history = data.get("history", [])
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not msg:
        return jsonify({"reply": "Please type a message.", "source": "rules"})

    patient_context = ""
    if "user_id" in session:
        details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
        user    = db.session.get(User, session["user_id"])
        if details:
            h   = (details.height / 100) if details.height else None
            bmi = round(details.weight / (h * h), 1) if (h and details.weight) else None
            patient_context = f"""
The patient's current health data:
- Name: {user.name if user else 'Unknown'}
- Age: {details.age or 'Unknown'}
- Weight: {details.weight or 'Unknown'} kg
- Height: {details.height or 'Unknown'} cm
- BMI: {bmi or 'Unknown'}
- Blood Pressure: {details.bp or 'Unknown'}
- Blood Sugar: {details.sugar or 'Unknown'} mg/dL
- Smoking: {details.smoking or 'Unknown'}
- Activity Level: {details.activity or 'Unknown'}
Use this data to give personalised advice when relevant.
"""

    system_prompt = f"""You are MedIntel Health Assistant, a friendly and knowledgeable AI health chatbot.

Your role:
- Answer health questions clearly, concisely, and helpfully
- Give personalised advice when patient health data is available
- Use bullet points for lists, keep responses focused (3-6 sentences max)
- Always recommend consulting a real doctor for diagnosis or treatment
- You CAN discuss symptoms, medications, diet, exercise, mental health, and lifestyle
- For emergencies, always tell them to call 108 (India ambulance) immediately
- Be warm, empathetic, and encouraging
{patient_context}
Today's date: {datetime.date.today().strftime('%B %d, %Y')}"""

    if api_key:
        try:
            client   = Groq(api_key=api_key)
            messages = [{"role": "system", "content": system_prompt}]
            for turn in history[-10:]:
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append({"role": "user", "content": msg})

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            return jsonify({"reply": reply, "source": "groq"})
        except Exception as e:
            print(f"Groq API error: {e}")

    lower = msg.lower()
    rules = {
        "fever":          "🌡️ Drink plenty of fluids and rest. Take paracetamol for high temperature. See a doctor if fever exceeds 103°F or lasts more than 3 days.",
        "headache":       "🤕 Rest in a quiet dark room, stay hydrated, and avoid screens. If caused by heat or exercise, drink ORS or electrolytes immediately.",
        "diabetes":       "🩸 Monitor blood sugar regularly, eat low-GI foods, exercise daily, and avoid sugary drinks. Regular HbA1c checks are important.",
        "heart":          "❤️ Reduce salt and saturated fats, quit smoking, exercise 30 min/day, and get regular BP and cholesterol checks.",
        "cold":           "🤧 Rest, drink warm fluids (ginger tea, soups), and take vitamin C. Most colds resolve in 7–10 days.",
        "bmi":            "⚖️ BMI = weight(kg) ÷ height²(m). Healthy range is 18.5–24.9. Check your BMI in the BMI section of the app.",
        "emergency":      "🚨 Call 108 (Ambulance) immediately for any medical emergency.",
        "blood pressure": "💉 Normal BP is 120/80 mmHg. Reduce salt, manage stress, exercise, and sleep 7–8 hours.",
        "sleep":          "😴 Adults need 7–9 hours of sleep. Consistent schedule and no screens before bed helps greatly.",
        "stress":         "🧘 Try deep breathing, meditation, or light exercise. Persistent stress raises cardiovascular risk.",
        "diet":           "🥗 Eat more vegetables, whole grains, and lean protein. Limit processed foods, sugar, and excess salt.",
        "cholesterol":    "🫀 Keep LDL below 100 mg/dL. Reduce fried foods; increase fibre, nuts, and omega-3 rich foods.",
        "water":          "💧 Aim for 8–10 glasses (2–2.5 litres) of water daily. More if you exercise or are in a hot climate.",
        "exercise":       "🏃 Aim for 150 min/week of moderate activity. Mix cardio with strength training for best results.",
        "vitamin":        "💊 A balanced diet covers most vitamin needs. Common deficiencies in India: Vitamin D, B12, and Iron.",
        "pain":           "🩹 Note the location, intensity, and duration of pain. Sudden severe pain always warrants emergency care.",
    }
    for key, reply in rules.items():
        if key in lower:
            return jsonify({"reply": reply, "source": "rules"})

    return jsonify({
        "reply": (
            "🤔 I'm not sure about that. I can help with symptoms, diet, heart health, diabetes, "
            "BMI, blood pressure, sleep, stress, and more.\n\n"
            "💡 Tip: Set your GROQ_API_KEY in .env file to enable full AI responses!"
        ),
        "source": "rules"
    })

# ─── Appointments ─────────────────────────────────────────────────────────────

@app.route("/appointment", methods=["GET", "POST"])
def appointment():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") == "doctor":
        return redirect(url_for("dashboard"))

    user    = db.session.get(User, session["user_id"])
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    doctors = User.query.filter_by(role="doctor").all()

    suggested_doctor_id = None
    suggestion_reason   = None

    if details and doctors:
        if details.bp:
            try:
                sys_bp = int(details.bp.split("/")[0])
                if sys_bp >= 140:
                    suggested_doctor_id = doctors[0].id
                    suggestion_reason = f"Your blood pressure ({details.bp}) is high — a cardiologist can help."
            except (ValueError, IndexError):
                pass

        if not suggested_doctor_id and details.sugar and details.sugar >= 126:
            suggested_doctor_id = doctors[1].id if len(doctors) > 1 else doctors[0].id
            suggestion_reason = f"Your blood sugar ({details.sugar} mg/dL) is in diabetic range — an endocrinologist can help."

        if not suggested_doctor_id and details.height and details.weight:
            h   = details.height / 100
            bmi = details.weight / (h * h)
            if bmi >= 30:
                suggested_doctor_id = doctors[0].id
                suggestion_reason = f"Your BMI ({round(bmi,1)}) is in obese range — a general physician can help."

        if not suggested_doctor_id and details.smoking == "yes":
            suggested_doctor_id = doctors[0].id
            suggestion_reason = "Smoking significantly increases health risks — a doctor can help you quit safely."

    if request.method == "POST":
        f = request.form
        try:
            appt = Appointment(
                patient_id = session["user_id"],
                doctor_id  = int(f["doctor_id"]),
                date       = f["date"],
                time       = f["time"],
                status     = "Pending"
            )
            db.session.add(appt)
            db.session.commit()
        except (ValueError, KeyError):
            pass
        return redirect(url_for("appointment"))

    appointments = Appointment.query.filter_by(patient_id=session["user_id"]).all()
    appt_data    = []
    for appt in appointments:
        doc = db.session.get(User, appt.doctor_id)
        appt_data.append({"appt": appt, "doctor": doc})

    return render_template("appointment.html",
        user                = user,
        details             = details,
        doctors             = doctors,
        appt_data           = appt_data,
        suggested_doctor_id = suggested_doctor_id,
        suggestion_reason   = suggestion_reason
    )

@app.route("/appointment/cancel/<int:appt_id>")
def cancel_appointment(appt_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    appt = Appointment.query.get_or_404(appt_id)
    if appt.patient_id == session["user_id"] and appt.status == "Pending":
        appt.status = "Cancelled"
        db.session.commit()
    return redirect(url_for("appointment"))

# ─── Patient — View My Prescriptions ─────────────────────────────────────────

@app.route("/my-prescriptions")
def my_prescriptions():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "patient":
        return redirect(url_for("dashboard"))
    prescriptions = Prescription.query.filter_by(
        patient_id=session["user_id"]
    ).order_by(Prescription.date.desc()).all()

    pres_data = []
    for pres in prescriptions:
        doc = db.session.get(User, pres.doctor_id)
        pres_data.append({"pres": pres, "doctor": doc})

    return render_template("my_prescriptions.html", pres_data=pres_data)

# ─── Doctor — Patient Detail View ────────────────────────────────────────────

@app.route("/doctor/patient/<int:patient_id>")
def doctor_patient(patient_id):
    if "user_id" not in session or session.get("role") != "doctor":
        return redirect(url_for("login"))
    patient       = User.query.get_or_404(patient_id)
    details       = PatientDetail.query.filter_by(user_id=patient_id).first()
    predictions   = Prediction.query.filter_by(user_id=patient_id).order_by(Prediction.date.desc()).limit(5).all()
    prescriptions = Prescription.query.filter_by(patient_id=patient_id).order_by(Prescription.date.desc()).all()
    score         = calculate_health_score(details) if details else None
    return render_template("doctor_patient.html",
        patient=patient, details=details,
        predictions=predictions, prescriptions=prescriptions, score=score
    )

# ─── Doctor — Appointment Management ─────────────────────────────────────────

@app.route("/doctor/appointment/<int:appt_id>/<action>")
def doctor_appointment_action(appt_id, action):
    if "user_id" not in session or session.get("role") != "doctor":
        return redirect(url_for("login"))
    appt = Appointment.query.get_or_404(appt_id)
    if appt.doctor_id != session["user_id"]:
        return redirect(url_for("dashboard"))
    if action == "confirm":
        appt.status = "Confirmed"
    elif action == "cancel":
        appt.status = "Cancelled"
    db.session.commit()
    return redirect(url_for("dashboard"))

# ─── Doctor — Write Prescription ─────────────────────────────────────────────

@app.route("/doctor/prescription/<int:patient_id>", methods=["GET", "POST"])
def write_prescription(patient_id):
    if "user_id" not in session or session.get("role") != "doctor":
        return redirect(url_for("login"))
    patient = User.query.get_or_404(patient_id)
    details = PatientDetail.query.filter_by(user_id=patient_id).first()
    if request.method == "POST":
        pres = Prescription(
            doctor_id  = session["user_id"],
            patient_id = patient_id,
            medicines  = request.form.get("medicines", "").strip(),
            notes      = request.form.get("notes", "").strip()
        )
        db.session.add(pres)
        db.session.commit()
        return redirect(url_for("doctor_patient", patient_id=patient_id))
    prescriptions = Prescription.query.filter_by(
        doctor_id=session["user_id"], patient_id=patient_id
    ).order_by(Prescription.date.desc()).all()
    return render_template("doctor_prescription.html",
        patient=patient, details=details, prescriptions=prescriptions
    )

# ─── Emergency ────────────────────────────────────────────────────────────────

@app.route("/emergency")
def emergency():
    if "user_id" not in session:
        return redirect(url_for("login"))
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    score   = calculate_health_score(details) if details else 100
    alert   = score < 50
    return render_template("emergency.html",
        score=score, alert=alert, details=details
    )

# ─── Admin ────────────────────────────────────────────────────────────────────

@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))
    recent_users = User.query.order_by(User.created.desc()).limit(8).all()
    return render_template("admin.html",
        total_users        = User.query.count(),
        total_appointments = Appointment.query.count(),
        total_predictions  = Prediction.query.count(),
        total_prescriptions= Prescription.query.count(),
        total_patients     = User.query.filter_by(role="patient").count(),
        total_doctors      = User.query.filter_by(role="doctor").count(),
        total_admins       = User.query.filter_by(role="admin").count(),
        recent_users       = recent_users
    )

# ─── Health Score API ─────────────────────────────────────────────────────────

@app.route("/api/health-score")
def api_health_score():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    if not details:
        return jsonify({"score": None, "message": "No health data found"})
    score, breakdown = calculate_health_score(details, breakdown=True)
    return jsonify({"score": score, "breakdown": breakdown})

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)