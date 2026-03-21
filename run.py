import os
from app import app, db

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        from models import User
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
            print("✅ Admin created!")
        else:
            print("✅ Admin already exists!")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
