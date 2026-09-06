from werkzeug.security import generate_password_hash

from app import app

from models import db, User


with app.app_context():

    existing_admin = User.query.filter_by(
        email="admin@breastcancer.com"
    ).first()

    if existing_admin:

        print("Admin already exists.")

    else:

        admin = User(

            name="Administrator",

            email="admin@breastcancer.com",

            password=generate_password_hash(
                "admin123"
            ),

            role="admin"

        )

        db.session.add(admin)

        db.session.commit()

        print("Admin account created successfully.")

        print("Email: admin@breastcancer.com")

        print("Password: admin123")