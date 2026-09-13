import os
import json
import uuid

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file,
    send_from_directory,
)

from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY,
    ORIGINAL_FOLDER,
    GRADCAM_FOLDER,
    REPORT_FOLDER,
    ALLOWED_EXTENSIONS,
)

from models import (
    init_db,
    create_user,
    get_user,
    get_user_by_id,
    save_prediction,
    get_user_predictions,
    get_prediction,
    get_all_patients,
    get_all_predictions,
    count_patients,
    count_predictions,
)

from ml_service import predict_image
from report_service import generate_report


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)
app.secret_key = SECRET_KEY


# =========================================================
# DATABASE
# =========================================================

init_db()


# =========================================================
# DIRECTORIES
# =========================================================

os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# =========================================================
# CANCER INFORMATION
# =========================================================

CANCER_INFORMATION = {

    "adenosis": {
        "title": "Adenosis",
        "category": "Benign Breast Condition",
        "description": (
            "Adenosis is a benign breast condition in which the glands "
            "(lobules) become enlarged or more numerous."
        ),
        "features": [
            "Usually benign",
            "May produce a lump or area of thickening",
            "Often discovered during breast imaging or examination",
        ],
        "precautions": [
            "Do not assume every breast lump is cancer.",
            "Any persistent or unusual breast change should be evaluated by a healthcare professional.",
        ],
        "detection": [
            "Awareness of normal breast appearance and feel",
            "Appropriate clinical examination",
            "Imaging when indicated",
        ],
        "diagnosis": [
            "Clinical examination",
            "Mammography or ultrasound when appropriate",
            "Biopsy or histopathological examination when needed",
        ],
    },

    "fibroadenoma": {
        "title": "Fibroadenoma",
        "category": "Benign Breast Tumor",
        "description": (
            "Fibroadenoma is a common benign breast tumor, particularly "
            "in younger people. It is not breast cancer."
        ),
        "features": [
            "Usually a firm, smooth lump",
            "Often moves somewhat under the skin",
            "May be painless",
        ],
        "precautions": [
            "Do not try to diagnose a lump yourself.",
            "A new or changing lump should be medically evaluated.",
        ],
        "detection": [
            "Awareness of breast changes",
            "Clinical evaluation of a new lump",
            "Imaging when recommended",
        ],
        "diagnosis": [
            "Clinical examination",
            "Ultrasound and/or mammography depending on age and circumstances",
            "Biopsy may be recommended when the diagnosis is uncertain",
        ],
    },

    "tubular_adenoma": {
        "title": "Tubular Adenoma",
        "category": "Benign Breast Tumor",
        "description": (
            "Tubular adenoma is a rare benign breast tumor made primarily "
            "of closely packed tubular structures."
        ),
        "features": [
            "Rare benign breast tumor",
            "May present as a breast lump",
            "Contains closely packed tubular structures",
        ],
        "precautions": [
            "A benign-looking lump still needs appropriate evaluation.",
            "Follow medical recommendations if imaging shows an abnormality.",
        ],
        "detection": [
            "Recognition of a new breast lump",
            "Clinical examination and appropriate imaging",
        ],
        "diagnosis": [
            "Clinical examination",
            "Breast imaging",
            "Histopathological examination",
        ],
    },

    "ductal_carcinoma": {
        "title": "Ductal Carcinoma",
        "category": "Malignant Breast Tumor",
        "description": (
            "Ductal carcinoma begins in the cells lining the milk ducts. "
            "Invasive ductal carcinoma has grown beyond the duct into "
            "surrounding breast tissue."
        ),
        "features": [
            "Breast lump or thickening",
            "Change in breast size or shape",
            "Nipple changes",
            "Abnormal nipple discharge",
            "Skin dimpling or other skin changes",
        ],
        "precautions": [
            "Pay attention to unusual breast changes.",
            "Follow recommended screening appropriate to age and individual risk.",
            "Seek medical evaluation rather than relying on self-diagnosis.",
        ],
        "detection": [
            "Screening mammography can detect some breast cancers before symptoms appear.",
            "Prompt evaluation of suspicious breast changes",
            "Clinical examination",
        ],
        "diagnosis": [
            "Clinical examination",
            "Mammography",
            "Ultrasound and/or MRI when indicated",
            "Biopsy",
            "Microscopic examination by a pathologist",
        ],
    },

    "lobular_carcinoma": {
        "title": "Lobular Carcinoma",
        "category": "Malignant Breast Tumor",
        "description": (
            "Lobular carcinoma begins in the milk-producing lobules. "
            "Invasive lobular carcinoma occurs when cancer cells spread "
            "beyond the lobules into surrounding tissue."
        ),
        "features": [
            "Breast thickening or swelling",
            "Change in breast shape",
            "Sometimes no obvious lump",
            "Other persistent breast changes",
        ],
        "precautions": [
            "Do not ignore persistent breast changes.",
            "People with significant family or genetic risk should discuss appropriate surveillance with a healthcare professional.",
        ],
        "detection": [
            "Appropriate breast screening",
            "Prompt investigation of unusual changes",
        ],
        "diagnosis": [
            "Clinical examination",
            "Mammography",
            "Ultrasound or MRI when indicated",
            "Biopsy and histopathology",
        ],
    },

    "mucinous_carcinoma": {
        "title": "Mucinous Carcinoma",
        "category": "Malignant Breast Tumor",
        "description": (
            "Mucinous carcinoma is a less common type of invasive breast "
            "cancer in which cancer cells are associated with substantial "
            "amounts of mucus, known as mucin."
        ),
        "features": [
            "Breast lump",
            "Breast thickening",
            "Changes detected during imaging",
        ],
        "precautions": [
            "Follow recommended breast screening.",
            "Report new or persistent breast changes.",
        ],
        "detection": [
            "Breast screening",
            "Timely investigation of abnormalities",
        ],
        "diagnosis": [
            "Breast imaging",
            "Biopsy",
            "Histopathological examination",
            "Biomarker testing may help characterize the cancer",
        ],
    },

    "papillary_carcinoma": {
        "title": "Papillary Carcinoma",
        "category": "Malignant Breast Tumor",
        "description": (
            "Papillary carcinoma is a breast cancer characterized by "
            "distinctive papillary structures when examined microscopically."
        ),
        "features": [
            "Breast lump",
            "Nipple changes",
            "Nipple discharge",
            "May sometimes be detected through imaging before noticeable symptoms",
        ],
        "precautions": [
            "Do not ignore nipple discharge or a new breast lump.",
            "Obtain professional assessment of persistent changes.",
        ],
        "detection": [
            "Appropriate screening",
            "Clinical evaluation of suspicious symptoms",
            "Breast imaging",
        ],
        "diagnosis": [
            "Mammography",
            "Ultrasound",
            "Biopsy",
            "Histopathological examination",
        ],
    },

    "phyllodes_tumor": {
        "title": "Phyllodes Tumor",
        "category": "Fibroepithelial Breast Tumor",
        "description": (
            "Phyllodes tumors develop from the connective or stromal tissue "
            "of the breast. They can be benign, borderline, or malignant."
        ),
        "features": [
            "Rapidly enlarging breast lump",
            "Breast enlargement or asymmetry",
            "Skin stretching over a large mass",
        ],
        "precautions": [
            "A rapidly growing breast lump should be medically assessed.",
            "Rapid growth does not automatically mean that a tumor is cancerous.",
        ],
        "detection": [
            "Prompt evaluation of a growing breast mass",
            "Breast imaging",
            "Tissue diagnosis when necessary",
        ],
        "diagnosis": [
            "Clinical examination",
            "Ultrasound and/or mammography",
            "Core biopsy or other tissue sampling",
            "Histopathological examination",
        ],
    },
}


# =========================================================
# HELPER - ALLOWED FILE
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# HELPER - PATIENT AUTHENTICATION
# =========================================================

def patient_required():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "patient":
        return redirect(url_for("home"))

    return None


# =========================================================
# HELPER - ADMIN AUTHENTICATION
# =========================================================

def admin_required():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        return redirect(url_for("home"))

    return None


# =========================================================
# HELPER - CANCER INFORMATION
# =========================================================

def get_cancer_info(name):

    if not name:
        return None

    key = str(name).strip().lower()
    key = key.replace(" ", "_")
    key = key.replace("-", "_")

    return CANCER_INFORMATION.get(key)


# =========================================================
# VERY IMPORTANT
# PREPARE PREDICTION FOR HTML TEMPLATES
# =========================================================

def prepare_prediction_for_template(prediction_data):

    """
    Converts database prediction data into a normal dictionary
    and creates all variables required by the HTML templates.

    This fixes:
        UndefinedError: 'probabilities' is undefined
        UndefinedError: 'value' is undefined
        UndefinedError: 'binary_confidence_percent' is undefined
        UndefinedError: 'subclass_confidence_percent' is undefined
    """

    if prediction_data is None:
        return None

    # -----------------------------------------------------
    # Convert sqlite3.Row / dict-like object to dictionary
    # -----------------------------------------------------

    try:
        prediction = dict(prediction_data)
    except Exception:
        prediction = prediction_data

    # -----------------------------------------------------
    # BINARY CONFIDENCE
    # -----------------------------------------------------

    try:
        binary_confidence = float(
            prediction.get("binary_confidence", 0) or 0
        )
    except (TypeError, ValueError):
        binary_confidence = 0.0

    # If model stored probability as 0-1,
    # convert it to percentage.
    if binary_confidence <= 1:
        binary_confidence_percent = binary_confidence * 100
    else:
        binary_confidence_percent = binary_confidence

    # Keep inside 0-100
    binary_confidence_percent = max(
        0.0,
        min(100.0, binary_confidence_percent)
    )

    # -----------------------------------------------------
    # SUBCLASS CONFIDENCE
    # -----------------------------------------------------

    try:
        subclass_confidence = float(
            prediction.get("subclass_confidence", 0) or 0
        )
    except (TypeError, ValueError):
        subclass_confidence = 0.0

    if subclass_confidence <= 1:
        subclass_confidence_percent = subclass_confidence * 100
    else:
        subclass_confidence_percent = subclass_confidence

    subclass_confidence_percent = max(
        0.0,
        min(100.0, subclass_confidence_percent)
    )

    # -----------------------------------------------------
    # ADD CLEAN PERCENTAGE VARIABLES
    # -----------------------------------------------------

    prediction["binary_confidence_percent"] = round(
        binary_confidence_percent,
        2
    )

    prediction["subclass_confidence_percent"] = round(
        subclass_confidence_percent,
        2
    )

    # -----------------------------------------------------
    # PARSE SUBCLASS PROBABILITIES
    # -----------------------------------------------------

    raw_probabilities = prediction.get(
        "subclass_probabilities",
        {}
    )

    probabilities = {}

    if raw_probabilities:

        try:

            if isinstance(raw_probabilities, str):

                probabilities = json.loads(
                    raw_probabilities
                )

            elif isinstance(raw_probabilities, dict):

                probabilities = raw_probabilities

        except Exception as error:

            print(
                "Probability JSON parsing error:",
                repr(error)
            )

            probabilities = {}

    # -----------------------------------------------------
    # MAKE SURE probabilities IS ALWAYS DEFINED
    # -----------------------------------------------------

    if not isinstance(probabilities, dict):
        probabilities = {}

    # -----------------------------------------------------
    # CONVERT PROBABILITIES TO PERCENTAGES
    # -----------------------------------------------------

    probability_percentages = {}

    for class_name, probability in probabilities.items():

        try:

            probability = float(probability)

            if probability <= 1:
                percentage = probability * 100
            else:
                percentage = probability

            percentage = max(
                0.0,
                min(100.0, percentage)
            )

            probability_percentages[
                class_name
            ] = round(
                percentage,
                2
            )

        except (TypeError, ValueError):

            probability_percentages[
                class_name
            ] = 0.0

    # -----------------------------------------------------
    # BOTH VARIABLES ARE PROVIDED
    #
    # Some templates may use:
    # probabilities
    #
    # Others may use:
    # probability_percentages
    # -----------------------------------------------------

    prediction["probabilities"] = probabilities

    prediction[
        "probability_percentages"
    ] = probability_percentages

    # -----------------------------------------------------
    # SORTED PROBABILITIES
    # -----------------------------------------------------

    prediction[
        "sorted_probabilities"
    ] = sorted(
        probability_percentages.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------------------------------
    # IMAGE FILENAMES
    # -----------------------------------------------------

    image_path = prediction.get(
        "image_path",
        ""
    )

    if image_path:
        prediction[
            "original_image_filename"
        ] = os.path.basename(
            image_path
        )
    else:
        prediction[
            "original_image_filename"
        ] = None

    # -----------------------------------------------------
    # GRAD-CAM FILENAME
    # -----------------------------------------------------

    gradcam_path = prediction.get(
        "gradcam_path"
    )

    if gradcam_path:

        prediction[
            "gradcam_filename"
        ] = os.path.basename(
            gradcam_path
        )

    else:

        prediction[
            "gradcam_filename"
        ] = None

    # -----------------------------------------------------
    # CANCER INFORMATION
    # -----------------------------------------------------

    prediction[
        "cancer_info"
    ] = get_cancer_info(
        prediction.get(
            "subclass_prediction"
        )
    )

    # -----------------------------------------------------
    # RETURN FINAL DICTIONARY
    # -----------------------------------------------------

    return prediction


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not name or not email or not password:

            flash(
                "Please fill all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        success = create_user(
            name,
            email,
            password,
            "patient"
        )

        if not success:

            flash(
                "Email already registered.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "auth/register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        user = get_user(
            email,
            password
        )

        if user is None:

            flash(
                "Invalid email or password.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        session.clear()

        session["user_id"] = user["id"]
        session["name"] = user["name"]
        session["role"] = user["role"]

        if user["role"] == "admin":

            return redirect(
                url_for("admin_dashboard")
            )

        return redirect(
            url_for("patient_dashboard")
        )

    return render_template(
        "auth/login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# PATIENT DASHBOARD
# =========================================================

@app.route("/dashboard")
def patient_dashboard():

    check = patient_required()

    if check:
        return check

    return render_template(
        "patient/dashboard.html",
        user_name=session.get("name")
    )


# =========================================================
# NEW PREDICTION PAGE
# =========================================================

@app.route("/prediction")
def prediction():

    check = patient_required()

    if check:
        return check

    return render_template(
        "patient/upload.html"
    )


# =========================================================
# MULTIPLE IMAGE PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    check = patient_required()

    if check:
        return check

    # -----------------------------------------------------
    # GET MULTIPLE IMAGES
    # -----------------------------------------------------

    files = request.files.getlist(
        "images"
    )

    # Fallback for old HTML
    if not files:

        single_file = request.files.get(
            "image"
        )

        if single_file:
            files = [single_file]

    # -----------------------------------------------------
    # VALID FILES
    # -----------------------------------------------------

    valid_files = []

    for file in files:

        if file and file.filename:

            if allowed_file(
                file.filename
            ):

                valid_files.append(
                    file
                )

    if not valid_files:

        flash(
            "Please select at least one valid image.",
            "error"
        )

        return redirect(
            url_for("prediction")
        )

    # -----------------------------------------------------
    # STORE ALL PREDICTION IDS
    # -----------------------------------------------------

    batch_prediction_ids = []

    failed_images = []

    total_images = len(
        valid_files
    )

    print("")
    print(
        "=========================================="
    )
    print(
        "MULTI IMAGE ANALYSIS STARTED"
    )
    print(
        "TOTAL IMAGES:",
        total_images
    )
    print(
        "=========================================="
    )

    # -----------------------------------------------------
    # PROCESS EACH IMAGE
    # -----------------------------------------------------

    for index, file in enumerate(
        valid_files,
        start=1
    ):

        original_filename = secure_filename(
            file.filename
        )

        print("")
        print(
            f"Processing image {index}/{total_images}: "
            f"{original_filename}"
        )

        unique_name = (
            str(uuid.uuid4())
            + "_"
            + original_filename
        )

        image_path = os.path.join(
            ORIGINAL_FOLDER,
            unique_name
        )

        try:

            # SAVE IMAGE
            file.save(
                image_path
            )

            print(
                "Image saved:",
                image_path
            )

            # -------------------------------------------------
            # AI PREDICTION
            # -------------------------------------------------

            result = predict_image(
                image_path
            )

            print(
                "Binary:",
                result.get(
                    "binary_prediction"
                )
            )

            print(
                "Binary confidence:",
                result.get(
                    "binary_confidence"
                )
            )

            print(
                "Subclass:",
                result.get(
                    "subclass_prediction"
                )
            )

            print(
                "Subclass confidence:",
                result.get(
                    "subclass_confidence"
                )
            )

            # -------------------------------------------------
            # GRAD-CAM
            # -------------------------------------------------

            gradcam_path = result.get(
                "gradcam_path"
            )

            if gradcam_path:

                print(
                    "Grad-CAM:",
                    gradcam_path
                )

            else:

                print(
                    "Grad-CAM was not generated."
                )

            # -------------------------------------------------
            # SUBCLASS PROBABILITIES
            # -------------------------------------------------

            subclass_probabilities = result.get(
                "subclass_probabilities",
                {}
            )

            # -------------------------------------------------
            # SAVE DATABASE RECORD
            # -------------------------------------------------

            prediction_id = save_prediction(

                user_id=session[
                    "user_id"
                ],

                original_filename=original_filename,

                image_path=image_path,

                gradcam_path=gradcam_path,

                binary_prediction=result[
                    "binary_prediction"
                ],

                binary_confidence=result[
                    "binary_confidence"
                ],

                subclass_prediction=result[
                    "subclass_prediction"
                ],

                subclass_confidence=result[
                    "subclass_confidence"
                ],

                subclass_probabilities=json.dumps(
                    subclass_probabilities
                )
            )

            batch_prediction_ids.append(
                prediction_id
            )

            print(
                "Prediction saved:",
                prediction_id
            )

            # -------------------------------------------------
            # GENERATE REPORT
            # -------------------------------------------------

            try:

                saved_prediction = get_prediction(
                    prediction_id
                )

                if saved_prediction:

                    generated_report = generate_report(
                        saved_prediction
                    )

                    print(
                        "Report created:",
                        generated_report
                    )

            except Exception as report_error:

                print(
                    "REPORT GENERATION ERROR:",
                    repr(report_error)
                )

        except Exception as error:

            print("")
            print(
                f"ERROR PROCESSING IMAGE "
                f"{index}/{total_images}"
            )

            print(
                repr(error)
            )

            failed_images.append(
                original_filename
            )

            continue

    # =====================================================
    # NO SUCCESSFUL IMAGE
    # =====================================================

    if not batch_prediction_ids:

        flash(
            "Image analysis failed. Please try again.",
            "error"
        )

        return redirect(
            url_for("prediction")
        )

    # =====================================================
    # STORE IDS
    # =====================================================

    session[
        "batch_prediction_ids"
    ] = [
        int(prediction_id)
        for prediction_id in batch_prediction_ids
    ]

    session.modified = True

    print("")
    print(
        "=========================================="
    )
    print(
        "MULTI IMAGE ANALYSIS COMPLETED"
    )
    print(
        "SUCCESSFUL:",
        len(batch_prediction_ids)
    )
    print(
        "FAILED:",
        len(failed_images)
    )
    print(
        "PREDICTION IDS:",
        batch_prediction_ids
    )
    print(
        "=========================================="
    )

    if failed_images:

        flash(
            "Some images could not be analyzed: "
            + ", ".join(
                failed_images
            ),
            "error"
        )

    return redirect(
        url_for(
            "prediction_results"
        )
    )


# =========================================================
# MULTIPLE PREDICTION RESULTS
# =========================================================

@app.route(
    "/prediction-results"
)
def prediction_results():

    check = patient_required()

    if check:
        return check

    prediction_ids = session.get(
        "batch_prediction_ids",
        []
    )

    print("")
    print(
        "=========================================="
    )
    print(
        "LOADING MULTI IMAGE RESULTS"
    )
    print(
        "SESSION IDS:",
        prediction_ids
    )
    print(
        "=========================================="
    )

    if not prediction_ids:

        return render_template(
            "patient/prediction_results.html",
            predictions=[],
            total=0
        )

    predictions = []

    for prediction_id in prediction_ids:

        prediction_data = get_prediction(
            prediction_id
        )

        if prediction_data is None:
            continue

        if prediction_data[
            "user_id"
        ] != session[
            "user_id"
        ]:

            continue

        prepared = prepare_prediction_for_template(
            prediction_data
        )

        if prepared:

            predictions.append(
                prepared
            )

    print(
        "RESULTS FOUND:",
        len(predictions)
    )

    session.pop(
        "batch_prediction_ids",
        None
    )

    return render_template(
        "patient/prediction_results.html",
        predictions=predictions,
        total=len(predictions)
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    check = patient_required()

    if check:
        return check

    raw_predictions = get_user_predictions(
        session["user_id"]
    )

    predictions = []

    for prediction_data in raw_predictions:

        prepared = prepare_prediction_for_template(
            prediction_data
        )

        if prepared:
            predictions.append(
                prepared
            )

    return render_template(
        "patient/history.html",
        predictions=predictions
    )


# =========================================================
# PATIENT PREDICTION DETAIL
# =========================================================

@app.route(
    "/prediction/<int:prediction_id>"
)
def prediction_detail(
    prediction_id
):

    check = patient_required()

    if check:
        return check

    prediction_data = get_prediction(
        prediction_id
    )

    if prediction_data is None:

        flash(
            "Prediction not found.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    if prediction_data[
        "user_id"
    ] != session[
        "user_id"
    ]:

        flash(
            "Access denied.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    # -----------------------------------------------------
    # PREPARE EVERYTHING FOR TEMPLATE
    # -----------------------------------------------------

    prediction_data = prepare_prediction_for_template(
        prediction_data
    )

    return render_template(
        "patient/prediction_detail.html",
        prediction=prediction_data,
        original_image_filename=prediction_data[
            "original_image_filename"
        ],
        gradcam_filename=prediction_data[
            "gradcam_filename"
        ],
        cancer_info=prediction_data[
            "cancer_info"
        ]
    )


# =========================================================
# CANCER INFORMATION
# =========================================================

@app.route(
    "/cancer-information"
)
def cancer_information():

    check = patient_required()

    if check:
        return check

    return render_template(
        "patient/cancer_information.html",
        cancers=CANCER_INFORMATION
    )


# =========================================================
# CANCER INFORMATION DETAIL
# =========================================================

@app.route(
    "/cancer-information/<name>"
)
def cancer_information_detail(
    name
):

    check = patient_required()

    if check:
        return check

    cancer_info = get_cancer_info(
        name
    )

    if cancer_info is None:

        return redirect(
            url_for(
                "cancer_information"
            )
        )

    return render_template(
        "patient/cancer_detail.html",
        cancer=cancer_info
    )


# =========================================================
# PROFILE
# =========================================================

@app.route("/profile")
def profile():

    check = patient_required()

    if check:
        return check

    user = get_user_by_id(
        session["user_id"]
    )

    return render_template(
        "patient/profile.html",
        user=user
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route(
    "/admin/dashboard"
)
def admin_dashboard():

    check = admin_required()

    if check:
        return check

    return render_template(
        "admin/dashboard.html",

        patient_count=count_patients(),

        prediction_count=count_predictions()
    )


# =========================================================
# ADMIN PATIENTS
# =========================================================

@app.route(
    "/admin/patients"
)
def admin_patients():

    check = admin_required()

    if check:
        return check

    patients = get_all_patients()

    return render_template(
        "admin/patients.html",
        patients=patients
    )


# =========================================================
# ADMIN PREDICTIONS
# =========================================================

@app.route(
    "/admin/predictions"
)
def admin_predictions():

    check = admin_required()

    if check:
        return check

    raw_predictions = get_all_predictions()

    predictions = []

    for prediction_data in raw_predictions:

        prepared = prepare_prediction_for_template(
            prediction_data
        )

        if prepared:

            predictions.append(
                prepared
            )

    return render_template(
        "admin/predictions.html",
        predictions=predictions
    )


# =========================================================
# ADMIN PATIENT DETAIL
# =========================================================

@app.route(
    "/admin/patient/<int:user_id>"
)
def admin_patient_detail(
    user_id
):

    check = admin_required()

    if check:
        return check

    user = get_user_by_id(
        user_id
    )

    if user is None:

        return redirect(
            url_for(
                "admin_patients"
            )
        )

    raw_predictions = get_user_predictions(
        user_id
    )

    predictions = []

    for prediction_data in raw_predictions:

        prepared = prepare_prediction_for_template(
            prediction_data
        )

        if prepared:

            predictions.append(
                prepared
            )

    return render_template(
        "admin/patient_detail.html",
        user=user,
        predictions=predictions
    )


# =========================================================
# ADMIN PREDICTION DETAIL
# =========================================================

@app.route(
    "/admin/prediction/<int:prediction_id>"
)
def admin_prediction_detail(
    prediction_id
):

    check = admin_required()

    if check:
        return check

    prediction_data = get_prediction(
        prediction_id
    )

    if prediction_data is None:

        flash(
            "Prediction not found.",
            "error"
        )

        return redirect(
            url_for(
                "admin_predictions"
            )
        )

    # =====================================================
    # THIS IS THE IMPORTANT FIX
    # =====================================================

    prediction_data = prepare_prediction_for_template(
        prediction_data
    )

    print("")
    print(
        "=========================================="
    )
    print(
        "ADMIN PREDICTION DETAIL"
    )
    print(
        "Prediction ID:",
        prediction_data.get("id")
    )
    print(
        "Binary:",
        prediction_data.get(
            "binary_prediction"
        )
    )
    print(
        "Binary confidence:",
        prediction_data.get(
            "binary_confidence_percent"
        )
    )
    print(
        "Subclass:",
        prediction_data.get(
            "subclass_prediction"
        )
    )
    print(
        "Subclass confidence:",
        prediction_data.get(
            "subclass_confidence_percent"
        )
    )
    print(
        "Probabilities:",
        prediction_data.get(
            "probabilities"
        )
    )
    print(
        "=========================================="
    )

    return render_template(
        "admin/prediction_detail.html",
        prediction=prediction_data
    )


# =========================================================
# SERVE ORIGINAL IMAGE
# =========================================================

@app.route(
    "/uploaded-image/<path:filename>"
)
def uploaded_image(filename):

    return send_from_directory(
        ORIGINAL_FOLDER,
        filename
    )


# =========================================================
# SERVE GRAD-CAM
# =========================================================

@app.route(
    "/gradcam-image/<path:filename>"
)
def gradcam_image(filename):

    return send_from_directory(
        GRADCAM_FOLDER,
        filename
    )


# =========================================================
# PATIENT DOWNLOAD REPORT
# =========================================================

@app.route(
    "/download-report/<int:prediction_id>"
)
def download_report(
    prediction_id
):

    check = patient_required()

    if check:
        return check

    prediction_data = get_prediction(
        prediction_id
    )

    if prediction_data is None:

        flash(
            "Prediction not found.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    if prediction_data[
        "user_id"
    ] != session[
        "user_id"
    ]:

        flash(
            "Access denied.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    report_filename = (
        "breast_cancer_report_"
        + str(prediction_id)
        + ".pdf"
    )

    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )

    if not os.path.exists(
        report_path
    ):

        try:

            report_path = generate_report(
                prediction_data
            )

        except Exception as error:

            print(
                "REPORT RE-GENERATION ERROR:",
                repr(error)
            )

            flash(
                "Unable to generate the report.",
                "error"
            )

            return redirect(
                url_for(
                    "prediction_detail",
                    prediction_id=prediction_id
                )
            )

    if not os.path.exists(
        report_path
    ):

        flash(
            "Report could not be created.",
            "error"
        )

        return redirect(
            url_for(
                "prediction_detail",
                prediction_id=prediction_id
            )
        )

    return send_file(
        report_path,
        as_attachment=True,
        download_name=report_filename,
        mimetype="application/pdf"
    )


# =========================================================
# ADMIN DOWNLOAD REPORT
# =========================================================

@app.route(
    "/admin/download-report/<int:prediction_id>"
)
def admin_download_report(
    prediction_id
):

    check = admin_required()

    if check:
        return check

    prediction_data = get_prediction(
        prediction_id
    )

    if prediction_data is None:

        flash(
            "Prediction not found.",
            "error"
        )

        return redirect(
            url_for(
                "admin_predictions"
            )
        )

    report_filename = (
        "breast_cancer_report_"
        + str(prediction_id)
        + ".pdf"
    )

    report_path = os.path.join(
        REPORT_FOLDER,
        report_filename
    )

    if not os.path.exists(
        report_path
    ):

        try:

            report_path = generate_report(
                prediction_data
            )

        except Exception as error:

            print(
                "ADMIN REPORT ERROR:",
                repr(error)
            )

            flash(
                "Unable to generate report.",
                "error"
            )

            return redirect(
                url_for(
                    "admin_prediction_detail",
                    prediction_id=prediction_id
                )
            )

    if not os.path.exists(
        report_path
    ):

        flash(
            "Report could not be created.",
            "error"
        )

        return redirect(
            url_for(
                "admin_prediction_detail",
                prediction_id=prediction_id
            )
        )

    return send_file(
        report_path,
        as_attachment=True,
        download_name=report_filename,
        mimetype="application/pdf"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )