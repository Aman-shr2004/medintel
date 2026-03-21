from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role             = db.Column(db.String(20), default="patient")  # patient / doctor / admin
    specialization   = db.Column(db.String(100), default="General Physician")  # Doctor specialization
    created          = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw): self.password = generate_password_hash(raw)
    def check_password(self, raw): return check_password_hash(self.password, raw)

class PatientDetail(db.Model):
    __tablename__ = "patient_details"
    id       = db.Column(db.Integer, primary_key=True)
    user_id  = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True)
    age      = db.Column(db.Integer)
    weight   = db.Column(db.Float)   # kg
    height   = db.Column(db.Float)   # cm
    bp       = db.Column(db.String(20))  # e.g. "120/80"
    sugar    = db.Column(db.Float)   # mg/dL fasting
    smoking  = db.Column(db.String(10), default="no")     # yes / no
    activity = db.Column(db.String(20), default="moderate")  # low / moderate / high

class Prediction(db.Model):
    __tablename__ = "predictions"
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey("users.id"))
    heart_risk    = db.Column(db.Integer)    # 0 or 1
    diabetes_risk = db.Column(db.Integer)    # 0 or 1
    date          = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    __tablename__ = "appointments"
    id         = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    doctor_id  = db.Column(db.Integer, db.ForeignKey("users.id"))
    date       = db.Column(db.String(20))
    time       = db.Column(db.String(20))
    status     = db.Column(db.String(20), default="Pending")  # Pending / Confirmed / Cancelled

class Prescription(db.Model):
    __tablename__ = "prescriptions"
    id         = db.Column(db.Integer, primary_key=True)
    doctor_id  = db.Column(db.Integer, db.ForeignKey("users.id"))
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    medicines  = db.Column(db.Text)
    notes      = db.Column(db.Text)
    date       = db.Column(db.DateTime, default=datetime.utcnow)