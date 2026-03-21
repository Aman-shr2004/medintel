# MedIntel – Smart Health Monitor

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_key_here
```

### 3. Run the app
```bash
python run.py
```
The app will start at **http://localhost:5000**

A default admin account is created automatically:
- Email: `admin@medintel.com`
- Password: `admin123`

### 4. Project Structure
```
medintel/
├── app.py              # Main Flask app & all routes
├── models.py           # Database models (SQLAlchemy)
├── health_score.py     # AI health score engine
├── run.py              # Startup script (creates DB + admin)
├── requirements.txt
├── .env                # GROQ_API_KEY goes here
├── heart.csv           # Heart disease training data
├── ml_models/          # Trained ML models (auto-generated)
├── static/css/
│   └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── login.html / register.html
    ├── dashboard.html
    ├── doctor_dashboard.html
    ├── admin.html
    └── ... (all other pages)
```

## Roles
- **Patient** – health data, BMI, AI predictions, appointments, prescriptions
- **Doctor** – view patients, manage appointments, write prescriptions  
- **Admin** – system overview panel

## Bugs Fixed
1. `base.html` – Missing `<script>` tag (JS was running as raw HTML text)
2. `app.py` – Admin users had no dashboard redirect (infinite loop)
3. `app.py` – `prediction` route didn't pass `details` to template on GET
4. `app.py` – `if not user:` block was broken after a bad edit
5. `app.py` – `load_model()` now falls back to root directory for `.pkl` files
6. `app.py` – All deprecated `User.query.get()` calls updated to `db.session.get()`
