from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from models import db, User, PatientDetail, Prediction, Appointment, Prescription
from health_score import calculate_health_score
import pickle, os, datetime, random, string, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from groq import Groq
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "medintel_secret_key_2024")
database_url = os.environ.get("DATABASE_URL", "sqlite:///database.db")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# ─── Google OAuth Setup ───────────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

with app.app_context():
    db.create_all()
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

# ─── OTP Helpers ──────────────────────────────────────────────────────────────

# In-memory OTP store: { email: { otp, expiry, purpose } }
_otp_store = {}

def generate_otp():
    return "".join(random.choices(string.digits, k=6))

def send_otp_email(to_email, otp, purpose="login"):
    sender_email    = os.environ.get("MAIL_USERNAME")
    sender_password = os.environ.get("MAIL_PASSWORD")

    if not sender_email or not sender_password:
        # Dev mode — print OTP to console so you can still test
        print(f"⚠️  MAIL not configured. OTP for {to_email}: {otp}")
        return True, True  # (sent_ok, is_dev_mode)

    action = "log in to" if purpose == "login" else "verify your email for"
    html_body = f"""
    <div style="font-family:'DM Sans',sans-serif;max-width:480px;margin:auto;padding:32px;
                background:#f0fdfa;border-radius:16px;border:1px solid #99f6e4;">
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:40px;">🩺</div>
        <h2 style="font-family:Georgia,serif;color:#0f766e;margin:8px 0 4px;">MedIntel</h2>
        <p style="color:#64748b;font-size:13px;margin:0;">Smart Health Monitor</p>
      </div>
      <h3 style="color:#0f172a;margin-bottom:8px;">Your One-Time Password</h3>
      <p style="color:#64748b;font-size:14px;margin-bottom:20px;">
        Use the code below to {action} MedIntel. It expires in <strong>10 minutes</strong>.
      </p>
      <div style="background:#fff;border:2px dashed #14b8a6;border-radius:12px;
                  padding:20px;text-align:center;margin-bottom:20px;">
        <span style="font-size:42px;font-weight:900;letter-spacing:14px;color:#0d9488;
                     font-family:Georgia,serif;">{otp}</span>
      </div>
      <p style="color:#94a3b8;font-size:12px;text-align:center;margin:0;">
        If you did not request this, please ignore this email.
      </p>
    </div>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "MedIntel – Your OTP Code"
    msg["From"]    = sender_email
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.sendmail(sender_email, to_email, msg.as_string())
        return True, False
    except Exception as e:
        print(f"Email send error: {e}")
        return False, False

# ─── OTP Routes ───────────────────────────────────────────────────────────────

@app.route("/send-otp", methods=["POST"])
def send_otp():
    data    = request.get_json()
    email   = (data.get("email") or "").strip().lower()
    purpose = data.get("purpose", "login")   # "login" or "register"

    if not email:
        return jsonify({"ok": False, "msg": "Email is required."})

    user = User.query.filter_by(email=email).first()

    if purpose == "login" and not user:
        return jsonify({"ok": False, "msg": "No account found with this email."})
    if purpose == "register" and user:
        return jsonify({"ok": False, "msg": "Email already registered. Please log in instead."})

    otp    = generate_otp()
    expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)

    _otp_store[email] = {"otp": otp, "expiry": expiry.isoformat(), "purpose": purpose}

    ok, is_dev = send_otp_email(email, otp, purpose)
    if ok:
        resp = {"ok": True, "msg": f"OTP sent to {email}"}
        if is_dev:
            resp["dev_otp"] = otp
            resp["msg"] = f"[DEV MODE] Mail not configured. OTP: {otp}"
        return jsonify(resp)
    return jsonify({"ok": False, "msg": "Failed to send email. Please configure MAIL_USERNAME and MAIL_PASSWORD in .env"})

@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data  = request.get_json()
    code  = (data.get("otp") or "").strip()
    email = (data.get("email") or "").strip().lower()

    record = _otp_store.get(email)
    if not record:
        return jsonify({"ok": False, "msg": "No OTP found for this email. Please request a new one."})

    expiry  = datetime.datetime.fromisoformat(record["expiry"])
    purpose = record["purpose"]

    if datetime.datetime.utcnow() > expiry:
        _otp_store.pop(email, None)
        return jsonify({"ok": False, "msg": "OTP expired. Please request a new one."})

    if code != record["otp"]:
        return jsonify({"ok": False, "msg": "Incorrect OTP. Please try again."})

    _otp_store.pop(email, None)

    if purpose == "login":
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"ok": False, "msg": "User not found."})
        session["user_id"] = user.id
        session["role"]    = user.role
        session["name"]    = user.name.title() if user.name else "User"
        session["email"]   = user.email
        return jsonify({"ok": True, "redirect": url_for("dashboard")})

    if purpose == "register":
        session["verified_email"] = email
        return jsonify({"ok": True, "redirect": url_for("register_complete")})

    return jsonify({"ok": False, "msg": "Unknown purpose."})

@app.route("/register/complete", methods=["GET", "POST"])
def register_complete():
    email = session.get("verified_email")
    if not email:
        return redirect(url_for("register"))
    google_name = session.get("google_name", "")
    is_google   = bool(google_name)
    if request.method == "POST":
        role = request.form.get("role", "patient")
        if role not in ("patient", "doctor"):
            role = "patient"
        spec = request.form.get("specialization", "General Physician") if role == "doctor" else "General Physician"
        name = request.form.get("name", "").strip() or google_name
        user = User(name=name, email=email, role=role, specialization=spec)
        raw_password = session.get("google_password") or request.form.get("password", "")
        user.set_password(raw_password)
        db.session.add(user)
        db.session.commit()
        session.pop("verified_email", None)
        session.pop("google_name", None)
        session.pop("google_password", None)
        session["user_id"] = user.id
        session["role"]    = user.role
        session["name"]    = user.name.title()
        session["email"]   = user.email
        return redirect(url_for("dashboard"))
    return render_template("register_complete.html", email=email, google_name=google_name, is_google=is_google)

# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET"])
def register():
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

# ─── Google OAuth ─────────────────────────────────────────────────────────────

@app.route("/auth/google")
def google_login():
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def google_callback():
    try:
        token    = google.authorize_access_token()
        userinfo = token.get("userinfo") or google.userinfo()
        email    = userinfo["email"].lower()
        name     = userinfo.get("name", email.split("@")[0])
        user = User.query.filter_by(email=email).first()
        if user:
            # Existing user — log straight in
            session["user_id"] = user.id
            session["role"]    = user.role
            session["name"]    = user.name.title()
            session["email"]   = user.email
            return redirect(url_for("dashboard"))
        else:
            # New Google user — let them choose their role
            session["verified_email"]  = email
            session["google_name"]     = name
            session["google_password"] = os.urandom(24).hex()
            return redirect(url_for("register_complete"))
    except Exception as e:
        print(f"Google OAuth error: {e}")
        return render_template("login.html", error="Google login failed. Please try again.")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ─── Dashboard ────────────────────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = db.session.get(User, session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))
    if user.role == "admin":
        return redirect(url_for("admin"))
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
            user=user, patient_data=patient_data, appointments=appointments,
            pending=pending, confirmed=confirmed, total_patients=len(patients)
        )
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

# ─── AI Prediction ────────────────────────────────────────────────────────────

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
            fi("age",40), fi("sex",1), fi("cp",0), fi("trestbps",120),
            fi("chol",200), fi("fbs",0), fi("restecg",0), fi("thalach",150),
            fi("exang",0), ff("oldpeak",0.0), fi("slope",1), fi("ca",0), fi("thal",1)
        ]
        diabetes_features = [
            fi("pregnancies",0), ff("glucose",100.0), ff("blood_pressure",70.0),
            ff("skin_thickness",20.0), ff("insulin",80.0),
            ff("bmi_val",25.0), ff("dpf",0.5), fi("age",40)
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
            user_id=session["user_id"],
            heart_risk=heart_risk, diabetes_risk=diabetes_risk
        )
        db.session.add(pred)
        db.session.commit()
        result = {
            "heart_risk": heart_risk, "heart_prob": heart_prob,
            "diabetes_risk": diabetes_risk, "diabetes_prob": diabetes_prob
        }
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    return render_template("prediction.html", result=result, details=details)

# ─── Chatbot ──────────────────────────────────────────────────────────────────

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
- BMI: {bmi or 'Unknown'}
- Blood Pressure: {details.bp or 'Unknown'}
- Blood Sugar: {details.sugar or 'Unknown'} mg/dL
- Smoking: {details.smoking or 'Unknown'}
- Activity Level: {details.activity or 'Unknown'}
"""
    system_prompt = f"""You are MedIntel Health Assistant, a friendly and knowledgeable AI health chatbot.
- Answer health questions clearly and helpfully
- Give personalised advice when patient health data is available
- Always recommend consulting a real doctor for diagnosis or treatment
- For emergencies, always tell them to call 108 (India ambulance) immediately
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
                model="llama-3.1-8b-instant", messages=messages,
                max_tokens=600, temperature=0.7
            )
            return jsonify({"reply": response.choices[0].message.content, "source": "groq"})
        except Exception as e:
            print(f"Groq API error: {e}")

    lower = msg.lower()
    rules = {
        "fever": "🌡️ Drink plenty of fluids and rest. Take paracetamol for high temperature.",
        "headache": "🤕 Rest in a quiet dark room, stay hydrated, and avoid screens.",
        "diabetes": "🩸 Monitor blood sugar regularly, eat low-GI foods, exercise daily.",
        "heart": "❤️ Reduce salt and saturated fats, quit smoking, exercise 30 min/day.",
        "cold": "🤧 Rest, drink warm fluids, and take vitamin C.",
        "bmi": "⚖️ BMI = weight(kg) ÷ height²(m). Healthy range is 18.5–24.9.",
        "emergency": "🚨 Call 108 (Ambulance) immediately for any medical emergency.",
        "blood pressure": "💉 Normal BP is 120/80 mmHg. Reduce salt, manage stress, exercise.",
        "sleep": "😴 Adults need 7–9 hours of sleep.",
        "stress": "🧘 Try deep breathing, meditation, or light exercise.",
        "diet": "🥗 Eat more vegetables, whole grains, and lean protein.",
        "water": "💧 Aim for 8–10 glasses of water daily.",
        "exercise": "🏃 Aim for 150 min/week of moderate activity.",
        "vitamin": "💊 Common deficiencies in India: Vitamin D, B12, and Iron.",
    }
    for key, reply in rules.items():
        if key in lower:
            return jsonify({"reply": reply, "source": "rules"})
    return jsonify({"reply": "🤔 I can help with symptoms, diet, heart health, diabetes, BMI, blood pressure, sleep, and more.", "source": "rules"})

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
                    suggestion_reason = f"Your blood pressure ({details.bp}) is high."
            except (ValueError, IndexError):
                pass
        if not suggested_doctor_id and details.sugar and details.sugar >= 126:
            suggested_doctor_id = doctors[1].id if len(doctors) > 1 else doctors[0].id
            suggestion_reason = f"Your blood sugar ({details.sugar} mg/dL) is in diabetic range."
        if not suggested_doctor_id and details.height and details.weight:
            h = details.height / 100
            bmi_v = details.weight / (h * h)
            if bmi_v >= 30:
                suggested_doctor_id = doctors[0].id
                suggestion_reason = f"Your BMI ({round(bmi_v,1)}) is in obese range."
        if not suggested_doctor_id and details.smoking == "yes":
            suggested_doctor_id = doctors[0].id
            suggestion_reason = "Smoking significantly increases health risks."
    if request.method == "POST":
        f = request.form
        try:
            appt = Appointment(
                patient_id=session["user_id"], doctor_id=int(f["doctor_id"]),
                date=f["date"], time=f["time"], status="Pending"
            )
            db.session.add(appt)
            db.session.commit()
        except (ValueError, KeyError):
            pass
        return redirect(url_for("appointment"))
    appointments = Appointment.query.filter_by(patient_id=session["user_id"]).all()
    appt_data = [{"appt": a, "doctor": db.session.get(User, a.doctor_id)} for a in appointments]
    return render_template("appointment.html",
        user=user, details=details, doctors=doctors, appt_data=appt_data,
        suggested_doctor_id=suggested_doctor_id, suggestion_reason=suggestion_reason
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

@app.route("/my-prescriptions")
def my_prescriptions():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "patient":
        return redirect(url_for("dashboard"))
    prescriptions = Prescription.query.filter_by(
        patient_id=session["user_id"]
    ).order_by(Prescription.date.desc()).all()
    pres_data = [{"pres": p, "doctor": db.session.get(User, p.doctor_id)} for p in prescriptions]
    return render_template("my_prescriptions.html", pres_data=pres_data)

# ─── Doctor Routes ────────────────────────────────────────────────────────────

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

@app.route("/doctor/prescription/<int:patient_id>", methods=["GET", "POST"])
def write_prescription(patient_id):
    if "user_id" not in session or session.get("role") != "doctor":
        return redirect(url_for("login"))
    patient = User.query.get_or_404(patient_id)
    details = PatientDetail.query.filter_by(user_id=patient_id).first()
    if request.method == "POST":
        pres = Prescription(
            doctor_id=session["user_id"], patient_id=patient_id,
            medicines=request.form.get("medicines", "").strip(),
            notes=request.form.get("notes", "").strip()
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
    return render_template("emergency.html", score=score, alert=alert, details=details)

# ─── Admin ────────────────────────────────────────────────────────────────────

@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))
    recent_users = User.query.order_by(User.created.desc()).limit(8).all()
    return render_template("admin.html",
        total_users         = User.query.count(),
        total_appointments  = Appointment.query.count(),
        total_predictions   = Prediction.query.count(),
        total_prescriptions = Prescription.query.count(),
        total_patients      = User.query.filter_by(role="patient").count(),
        total_doctors       = User.query.filter_by(role="doctor").count(),
        total_admins        = User.query.filter_by(role="admin").count(),
        recent_users        = recent_users
    )

@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    if session.get("role") != "admin":
        return jsonify({"ok": False, "msg": "Unauthorized"}), 403
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"ok": False, "msg": "User not found"}), 404
    if user.role == "admin":
        return jsonify({"ok": False, "msg": "Cannot delete admin account!"}), 400
    # Delete all related records first
    PatientDetail.query.filter_by(user_id=user_id).delete()
    Prediction.query.filter_by(user_id=user_id).delete()
    Appointment.query.filter_by(patient_id=user_id).delete()
    Appointment.query.filter_by(doctor_id=user_id).delete()
    Prescription.query.filter_by(patient_id=user_id).delete()
    Prescription.query.filter_by(doctor_id=user_id).delete()
    db.session.delete(user)
    db.session.commit()
    return jsonify({"ok": True, "msg": f"{user.name} deleted successfully!"})

@app.route("/api/health-score")
def api_health_score():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    details = PatientDetail.query.filter_by(user_id=session["user_id"]).first()
    if not details:
        return jsonify({"score": None, "message": "No health data found"})
    score, breakdown = calculate_health_score(details, breakdown=True)
    return jsonify({"score": score, "breakdown": breakdown})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port = 5001)