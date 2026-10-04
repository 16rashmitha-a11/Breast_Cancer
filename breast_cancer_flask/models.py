import sqlite3

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from config import DATABASE_PATH


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():

    connection = get_db()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password_hash TEXT NOT NULL,

            role TEXT NOT NULL DEFAULT 'patient',

            phone TEXT,

            age INTEGER,

            gender TEXT,

            address TEXT,

            profile_completed INTEGER NOT NULL DEFAULT 0,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------------
    # PREDICTIONS TABLE
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            original_filename TEXT,

            image_path TEXT,

            gradcam_path TEXT,

            binary_prediction TEXT,

            binary_confidence REAL,

            subclass_prediction TEXT,

            subclass_confidence REAL,

            subclass_probabilities TEXT,

            activation_percentage REAL,

            stage TEXT,

            stage_confidence REAL,

            stage_score REAL,

            stage_notice TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(user_id)
                REFERENCES users(id)
        )
    """)

    connection.commit()

    # ========================================================
    # MIGRATION FOR OLD DATABASE
    # ========================================================

    # USERS TABLE MIGRATION

    columns = cursor.execute(
        "PRAGMA table_info(users)"
    ).fetchall()

    existing_columns = {
        column["name"]
        for column in columns
    }

    required_user_columns = {

        "phone": "TEXT",

        "age": "INTEGER",

        "gender": "TEXT",

        "address": "TEXT",

        "profile_completed":
            "INTEGER NOT NULL DEFAULT 0"
    }

    for column_name, column_type in required_user_columns.items():

        if column_name not in existing_columns:

            cursor.execute(
                f"""
                ALTER TABLE users
                ADD COLUMN {column_name} {column_type}
                """
            )

    # --------------------------------------------------------
    # PREDICTIONS TABLE MIGRATION
    # --------------------------------------------------------

    prediction_columns = cursor.execute(
        "PRAGMA table_info(predictions)"
    ).fetchall()

    existing_prediction_columns = {
        column["name"]
        for column in prediction_columns
    }

    required_prediction_columns = {

        "activation_percentage":
            "REAL",

        "stage":
            "TEXT",

        "stage_confidence":
            "REAL",

        "stage_score":
            "REAL",

        "stage_notice":
            "TEXT"
    }

    for column_name, column_type in required_prediction_columns.items():

        if column_name not in existing_prediction_columns:

            cursor.execute(
                f"""
                ALTER TABLE predictions
                ADD COLUMN {column_name} {column_type}
                """
            )

    connection.commit()

    connection.close()


# ============================================================
# CREATE USER
# ============================================================

def create_user(
    name,
    email,
    password,
    role="patient"
):

    connection = get_db()

    try:

        password_hash = generate_password_hash(
            password
        )

        connection.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password_hash,
                role
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                email,
                password_hash,
                role
            )
        )

        connection.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        connection.close()


# ============================================================
# LOGIN
# ============================================================

def get_user(
    email,
    password
):

    connection = get_db()

    user = connection.execute(
        """
        SELECT *
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    connection.close()

    if user is None:
        return None

    try:

        if check_password_hash(
            user["password_hash"],
            password
        ):
            return user

    except (
        ValueError,
        TypeError
    ):

        return None

    return None


# ============================================================
# GET USER BY ID
# ============================================================

def get_user_by_id(user_id):

    connection = get_db()

    user = connection.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    connection.close()

    return user


# ============================================================
# UPDATE PATIENT PROFILE
# ============================================================

def update_profile(
    user_id,
    phone,
    age,
    gender,
    address
):

    connection = get_db()

    connection.execute(
        """
        UPDATE users
        SET
            phone = ?,
            age = ?,
            gender = ?,
            address = ?,
            profile_completed = 1
        WHERE id = ?
        """,
        (
            phone,
            age,
            gender,
            address,
            user_id
        )
    )

    connection.commit()

    connection.close()


# ============================================================
# CHECK PROFILE COMPLETION
# ============================================================

def is_profile_completed(user_id):

    connection = get_db()

    result = connection.execute(
        """
        SELECT profile_completed
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    connection.close()

    if result is None:
        return False

    return bool(
        result["profile_completed"]
    )


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    user_id,
    original_filename,
    image_path,
    gradcam_path,
    binary_prediction,
    binary_confidence,
    subclass_prediction,
    subclass_confidence,
    subclass_probabilities,
    activation_percentage=None,
    stage=None,
    stage_confidence=None,
    stage_score=None,
    stage_notice=None
):

    connection = get_db()

    cursor = connection.execute(
        """
        INSERT INTO predictions
        (
            user_id,
            original_filename,
            image_path,
            gradcam_path,
            binary_prediction,
            binary_confidence,
            subclass_prediction,
            subclass_confidence,
            subclass_probabilities,
            activation_percentage,
            stage,
            stage_confidence,
            stage_score,
            stage_notice
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            original_filename,
            image_path,
            gradcam_path,
            binary_prediction,
            binary_confidence,
            subclass_prediction,
            subclass_confidence,
            subclass_probabilities,
            activation_percentage,
            stage,
            stage_confidence,
            stage_score,
            stage_notice
        )
    )

    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id


# ============================================================
# GET USER PREDICTIONS
# ============================================================

def get_user_predictions(user_id):

    connection = get_db()

    predictions = connection.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    ).fetchall()

    connection.close()

    return predictions


# ============================================================
# GET SINGLE PREDICTION
# ============================================================

def get_prediction(prediction_id):

    connection = get_db()

    prediction = connection.execute(
        """
        SELECT
            predictions.*,

            users.name AS patient_name,

            users.email AS patient_email,

            users.phone AS patient_phone,

            users.age AS patient_age,

            users.gender AS patient_gender,

            users.address AS patient_address

        FROM predictions

        LEFT JOIN users
        ON predictions.user_id = users.id

        WHERE predictions.id = ?
        """,
        (prediction_id,)
    ).fetchone()

    connection.close()

    return prediction


# ============================================================
# ADMIN - GET ALL PATIENTS
# ============================================================

def get_all_patients():

    connection = get_db()

    patients = connection.execute(
        """
        SELECT
            users.*,

            COUNT(predictions.id)
            AS prediction_count

        FROM users

        LEFT JOIN predictions
        ON users.id = predictions.user_id

        WHERE users.role = 'patient'

        GROUP BY users.id

        ORDER BY users.id ASC
        """
    ).fetchall()

    connection.close()

    return patients


# ============================================================
# ADMIN - GET ALL PREDICTIONS
# ============================================================

def get_all_predictions():

    connection = get_db()

    predictions = connection.execute(
        """
        SELECT
            predictions.*,

            users.name AS patient_name,

            users.email AS patient_email

        FROM predictions

        LEFT JOIN users
        ON predictions.user_id = users.id

        ORDER BY predictions.created_at DESC
        """
    ).fetchall()

    connection.close()

    return predictions


# ============================================================
# COUNT PATIENTS
# ============================================================

def count_patients():

    connection = get_db()

    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM users
        WHERE role = 'patient'
        """
    ).fetchone()[0]

    connection.close()

    return count


# ============================================================
# COUNT PREDICTIONS
# ============================================================

def count_predictions():

    connection = get_db()

    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        """
    ).fetchone()[0]

    connection.close()

    return count