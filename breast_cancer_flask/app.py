import os
import json
import uuid
import traceback
from functools import wraps

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
    update_profile
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

    # =====================================================
    # BENIGN - ADENOSIS
    # =====================================================

    "adenosis": {
        "title": "Adenosis",
        "category": "Benign Condition",

        "description": (
            "Adenosis is a benign breast condition in which the "
            "lobules, which are the milk-producing glands of the breast, "
            "become enlarged or more numerous than usual."
        ),

        "overview": (
            "Adenosis refers to an increase in the size or number of "
            "lobular structures in breast tissue. It is generally a "
            "non-cancerous condition. Some forms, such as sclerosing "
            "adenosis, can produce changes that may appear similar to "
            "other breast abnormalities on examination or imaging."
        ),

        "features": [
            "Usually a benign breast condition",
            "Involves enlarged or increased lobular structures",
            "May produce a lump, thickening or area of breast discomfort",
            "May be discovered during breast imaging",
            "Some forms can resemble other breast abnormalities"
        ],

        "symptoms": [
            "Some people may have no noticeable symptoms",
            "A localized area of thickening may be present",
            "A breast lump or tenderness may sometimes occur",
            "Imaging may identify the condition before symptoms are noticed"
        ],

        "detection": [
            "Clinical breast examination when indicated",
            "Mammography when recommended",
            "Breast ultrasound when appropriate",
            "Further tissue evaluation when imaging findings require clarification"
        ],

        "diagnosis": [
            "Clinical examination",
            "Mammography or ultrasound when appropriate",
            "Additional imaging when required",
            "Biopsy or histopathological examination when necessary"
        ],

        "management": (
            "Management depends on the specific type of adenosis and the "
            "clinical findings. Some cases require observation or routine "
            "follow-up, while an abnormal or uncertain finding may require "
            "additional evaluation."
        ),

        "precautions": [
            "Do not assume that every breast lump is cancer.",
            "Do not attempt to diagnose a breast abnormality yourself.",
            "Any persistent or unusual breast change should be evaluated by a qualified healthcare professional.",
            "Follow the recommended imaging or follow-up schedule provided by your healthcare professional."
        ]
    },


    # =====================================================
    # BENIGN - FIBROADENOMA
    # =====================================================

    "fibroadenoma": {
        "title": "Fibroadenoma",
        "category": "Benign Breast Tumor",

        "description": (
            "Fibroadenoma is a common benign breast tumor composed of "
            "glandular and connective tissue. It is not breast cancer."
        ),

        "overview": (
            "Fibroadenomas are common benign breast tumors and are often "
            "seen in younger people. They may feel firm, smooth and mobile "
            "under the skin. Their size can vary and some may change over time."
        ),

        "features": [
            "Common benign breast tumor",
            "Often firm and smooth",
            "May feel mobile beneath the skin",
            "May be painless",
            "Can vary in size"
        ],

        "symptoms": [
            "A palpable breast lump",
            "The lump may be painless",
            "The lump may feel firm or rubbery",
            "Some fibroadenomas may increase or decrease in size"
        ],

        "detection": [
            "Awareness of a new breast lump",
            "Clinical examination",
            "Breast ultrasound when recommended",
            "Mammography when appropriate"
        ],

        "diagnosis": [
            "Clinical examination",
            "Ultrasound and/or mammography depending on age and circumstances",
            "Core needle biopsy when the diagnosis is uncertain",
            "Histopathological examination when tissue diagnosis is required"
        ],

        "management": (
            "Many fibroadenomas can be monitored when their clinical and "
            "imaging features are reassuring. Additional evaluation or "
            "removal may be considered depending on the size, growth, "
            "symptoms and clinical assessment."
        ),

        "precautions": [
            "Do not try to diagnose a new breast lump yourself.",
            "A new or changing breast lump should be medically evaluated.",
            "Follow the imaging and follow-up recommendations provided by your healthcare professional.",
            "Seek assessment if a previously stable lump changes noticeably."
        ]
    },


    # =====================================================
    # BENIGN - PHYLLODES TUMOR
    # =====================================================

    "phyllodes_tumor": {
        "title": "Phyllodes Tumor",
        "category": "Fibroepithelial Breast Tumor",

        "description": (
            "Phyllodes tumors are fibroepithelial breast tumors that "
            "develop from connective tissue. They may be benign, borderline "
            "or malignant."
        ),

        "overview": (
            "Phyllodes tumors are uncommon breast tumors. They can grow "
            "relatively quickly and are classified according to their "
            "pathological characteristics. Because their behavior can vary, "
            "professional evaluation and tissue diagnosis are important."
        ),

        "features": [
            "Uncommon fibroepithelial breast tumor",
            "May grow relatively quickly",
            "Can present as a large breast mass",
            "Can be benign, borderline or malignant",
            "Pathological examination is important for classification"
        ],

        "symptoms": [
            "A rapidly enlarging breast lump",
            "Breast enlargement or asymmetry",
            "A feeling of fullness or pressure",
            "Skin stretching over a large mass may occur"
        ],

        "detection": [
            "Clinical assessment of a growing breast mass",
            "Breast ultrasound",
            "Mammography when appropriate",
            "Tissue sampling when required"
        ],

        "diagnosis": [
            "Clinical examination",
            "Ultrasound and/or mammography",
            "Core needle biopsy or other tissue sampling",
            "Histopathological examination",
            "Pathological classification as benign, borderline or malignant"
        ],

        "management": (
            "Management depends on the pathological classification, tumor "
            "size, location and other clinical factors. Surgical treatment "
            "may be considered for confirmed phyllodes tumors, with the "
            "specific approach determined by the treating medical team."
        ),

        "precautions": [
            "A rapidly growing breast lump should be medically assessed.",
            "Rapid growth does not automatically mean that a tumor is malignant.",
            "Do not rely on an AI prediction to determine whether a phyllodes tumor is benign or malignant.",
            "Follow professional recommendations for tissue diagnosis and follow-up."
        ]
    },


    # =====================================================
    # BENIGN - TUBULAR ADENOMA
    # =====================================================

    "tubular_adenoma": {
        "title": "Tubular Adenoma",
        "category": "Benign Breast Tumor",

        "description": (
            "Tubular adenoma is a rare benign breast tumor characterized "
            "by closely packed tubular structures within the breast tissue."
        ),

        "overview": (
            "Tubular adenoma is an uncommon benign breast tumor. It may "
            "present as a localized breast mass and can have imaging or "
            "histological characteristics that require professional "
            "evaluation to distinguish it from other breast lesions."
        ),

        "features": [
            "Rare benign breast tumor",
            "Contains closely packed tubular structures",
            "May present as a localized breast lump",
            "Usually considered a benign lesion",
            "Histopathological examination may be required"
        ],

        "symptoms": [
            "A localized breast lump",
            "The lump may be painless",
            "Some cases may be discovered during imaging",
            "Symptoms vary depending on the size and location of the lesion"
        ],

        "detection": [
            "Recognition of a new breast lump",
            "Clinical breast examination",
            "Breast ultrasound",
            "Mammography when appropriate",
            "Histopathological assessment when required"
        ],

        "diagnosis": [
            "Clinical examination",
            "Breast imaging",
            "Core biopsy or tissue sampling when necessary",
            "Histopathological examination"
        ],

        "management": (
            "Management depends on the clinical and pathological findings. "
            "A healthcare professional may recommend observation, follow-up "
            "or additional treatment depending on the individual case."
        ),

        "precautions": [
            "A benign-looking breast lump still requires appropriate evaluation.",
            "Do not assume that a lump is tubular adenoma without professional assessment.",
            "Follow recommendations if imaging identifies an abnormality.",
            "Report any significant change in a known breast lesion."
        ]
    },


    # =====================================================
    # MALIGNANT - DUCTAL CARCINOMA
    # =====================================================

    "ductal_carcinoma": {
        "title": "Ductal Carcinoma",
        "category": "Malignant Breast Tumor",

        "description": (
            "Ductal carcinoma refers to carcinoma associated with the "
            "milk ducts of the breast. Invasive ductal carcinoma occurs "
            "when cancer cells grow beyond the duct into surrounding tissue."
        ),

        "overview": (
            "Ductal carcinoma is one of the major pathological patterns "
            "used in breast cancer classification. Invasive ductal carcinoma "
            "can spread from the ducts into surrounding breast tissue and "
            "may potentially spread to other parts of the body."
        ),

        "features": [
            "Associated with the ductal structures of the breast",
            "May be invasive or non-invasive depending on the pathological diagnosis",
            "May present as a breast mass",
            "May cause changes in breast appearance",
            "Requires professional pathological evaluation"
        ],

        "symptoms": [
            "A new breast lump or thickening",
            "Change in breast size or shape",
            "Nipple changes",
            "Abnormal nipple discharge",
            "Skin dimpling or other skin changes",
            "Some breast cancers may cause no obvious symptoms initially"
        ],

        "detection": [
            "Routine breast screening when appropriate",
            "Clinical evaluation of suspicious breast changes",
            "Mammography",
            "Ultrasound when indicated",
            "MRI in selected clinical situations"
        ],

        "diagnosis": [
            "Clinical examination",
            "Mammography",
            "Ultrasound and/or MRI when indicated",
            "Biopsy",
            "Microscopic examination by a pathologist",
            "Additional pathological and biomarker testing when required"
        ],

        "management": (
            "Treatment depends on the confirmed pathological diagnosis, "
            "stage, tumor characteristics and individual patient factors. "
            "Possible approaches can include surgery, radiation therapy, "
            "systemic therapy or other treatments selected by the medical team."
        ),

        "precautions": [
            "Pay attention to new or unusual breast changes.",
            "Follow recommended screening appropriate to age and individual risk.",
            "Do not rely on an AI prediction as a confirmed diagnosis.",
            "Seek professional assessment for a persistent lump, nipple change or other concerning breast symptom."
        ]
    },


    # =====================================================
    # MALIGNANT - LOBULAR CARCINOMA
    # =====================================================

    "lobular_carcinoma": {
        "title": "Lobular Carcinoma",
        "category": "Malignant Breast Tumor",

        "description": (
            "Lobular carcinoma is a carcinoma associated with the "
            "milk-producing lobular structures of the breast. Invasive "
            "lobular carcinoma occurs when cancer cells grow beyond the "
            "lobules into surrounding breast tissue."
        ),

        "overview": (
            "Lobular carcinoma develops in or is associated with the "
            "lobular structures of the breast. Invasive lobular carcinoma "
            "may grow in patterns that make it less obvious as a discrete "
            "lump, so clinical assessment and appropriate imaging are important."
        ),

        "features": [
            "Associated with the lobular structures of the breast",
            "Invasive forms can extend into surrounding breast tissue",
            "May not always form a clearly defined lump",
            "Requires pathological confirmation",
            "Additional testing may be required to characterize the tumor"
        ],

        "symptoms": [
            "Breast thickening",
            "Breast swelling",
            "Change in breast shape",
            "A less clearly defined area of abnormality",
            "Other persistent breast changes"
        ],

        "detection": [
            "Appropriate breast screening",
            "Clinical assessment of persistent changes",
            "Mammography",
            "Ultrasound or MRI when clinically indicated"
        ],

        "diagnosis": [
            "Clinical examination",
            "Breast imaging",
            "Biopsy",
            "Histopathological examination",
            "Additional pathological testing when required"
        ],

        "management": (
            "Management depends on the confirmed diagnosis, stage, "
            "biological characteristics and individual clinical factors. "
            "Treatment decisions should be made by the patient's healthcare team."
        ),

        "precautions": [
            "Do not ignore persistent breast changes.",
            "Seek professional evaluation if a new or unusual breast change develops.",
            "People with significant family or genetic risk should discuss appropriate surveillance with a healthcare professional.",
            "AI classification should not be considered a substitute for pathological diagnosis."
        ]
    },


    # =====================================================
    # MALIGNANT - MUCINOUS CARCINOMA
    # =====================================================

    "mucinous_carcinoma": {
        "title": "Mucinous Carcinoma",
        "category": "Malignant Breast Tumor",

        "description": (
            "Mucinous carcinoma is a less common type of invasive breast "
            "cancer in which tumor cells are associated with substantial "
            "amounts of extracellular mucin."
        ),

        "overview": (
            "Mucinous carcinoma is characterized by the presence of "
            "mucin surrounding or associated with tumor cells. It is "
            "identified through pathological examination and may have "
            "distinctive microscopic characteristics."
        ),

        "features": [
            "Less common type of invasive breast cancer",
            "Tumor cells are associated with extracellular mucin",
            "Has characteristic microscopic features",
            "Requires pathological confirmation",
            "Biological characteristics can vary between tumors"
        ],

        "symptoms": [
            "Breast lump",
            "Breast thickening",
            "Changes detected during breast imaging",
            "Other persistent breast changes"
        ],

        "detection": [
            "Breast screening when appropriate",
            "Clinical assessment of suspicious breast changes",
            "Mammography",
            "Ultrasound or other imaging when indicated"
        ],

        "diagnosis": [
            "Breast imaging",
            "Biopsy",
            "Histopathological examination",
            "Microscopic identification of mucinous features",
            "Biomarker testing may help characterize the cancer"
        ],

        "management": (
            "Treatment is based on the confirmed pathological diagnosis, "
            "stage, tumor characteristics and individual clinical factors. "
            "The treating healthcare team determines the appropriate management plan."
        ),

        "precautions": [
            "Follow recommended breast screening.",
            "Report new or persistent breast changes to a healthcare professional.",
            "Do not assume that a particular subtype can be confirmed from an AI prediction alone.",
            "Professional pathological evaluation is required for diagnosis."
        ]
    },


    # =====================================================
    # MALIGNANT - PAPILLARY CARCINOMA
    # =====================================================

    "papillary_carcinoma": {
        "title": "Papillary Carcinoma",
        "category": "Malignant Breast Tumor",

        "description": (
            "Papillary carcinoma is a breast cancer characterized by "
            "distinctive papillary growth patterns when the tissue is "
            "examined microscopically."
        ),

        "overview": (
            "Papillary carcinoma includes breast lesions with papillary "
            "architecture identified through pathological examination. "
            "The exact type and behavior of a papillary lesion depend on "
            "its pathological characteristics."
        ),

        "features": [
            "Characterized by papillary growth patterns",
            "Requires microscopic pathological evaluation",
            "May occur in different pathological forms",
            "May present as a breast lesion or mass",
            "Additional pathological testing may be required"
        ],

        "symptoms": [
            "Breast lump",
            "Nipple changes",
            "Nipple discharge",
            "Breast thickening",
            "Some lesions may be detected during imaging before symptoms appear"
        ],

        "detection": [
            "Appropriate breast screening",
            "Clinical evaluation of suspicious symptoms",
            "Mammography",
            "Ultrasound",
            "Additional imaging when clinically indicated"
        ],

        "diagnosis": [
            "Mammography",
            "Ultrasound",
            "Biopsy",
            "Histopathological examination",
            "Additional pathological or biomarker testing when required"
        ],

        "management": (
            "Management depends on the exact pathological diagnosis, "
            "extent of disease, tumor characteristics and individual "
            "clinical factors. Treatment should be determined by qualified healthcare professionals."
        ),

        "precautions": [
            "Do not ignore nipple discharge or a new breast lump.",
            "Obtain professional assessment of persistent breast changes.",
            "Follow recommended screening and clinical follow-up.",
            "An AI classification should not be treated as a confirmed medical diagnosis."
        ]
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
#
# IMPORTANT:
# This is intentionally a normal helper function.
# Your existing routes use:
#
#     check = patient_required()
#
# so DO NOT use @patient_required on routes.
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
# HELPER - CONVERT DATABASE ROW
# =========================================================

def row_to_dict(row):

    if row is None:
        return None

    try:
        return dict(row)
    except Exception:
        return row


# =========================================================
# PREPARE PREDICTION FOR HTML TEMPLATES
# =========================================================

def prepare_prediction_for_template(prediction_data):

    if prediction_data is None:
        return None

    prediction = row_to_dict(prediction_data)

    # -----------------------------------------------------
    # BINARY CONFIDENCE
    # -----------------------------------------------------

    try:
        binary_confidence = float(
            prediction.get("binary_confidence", 0) or 0
        )
    except (TypeError, ValueError):
        binary_confidence = 0.0

    if binary_confidence <= 1:
        binary_confidence_percent = binary_confidence * 100
    else:
        binary_confidence_percent = binary_confidence

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

            probability_percentages[class_name] = round(
                percentage,
                2
            )

        except (TypeError, ValueError):

            probability_percentages[class_name] = 0.0

    prediction["probabilities"] = probabilities

    prediction["probability_percentages"] = (
        probability_percentages
    )

    prediction["sorted_probabilities"] = sorted(
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

        prediction["original_image_filename"] = (
            os.path.basename(image_path)
        )

    else:

        prediction["original_image_filename"] = None

    # -----------------------------------------------------
    # GRAD-CAM FILENAME
    # -----------------------------------------------------

    gradcam_path = prediction.get(
        "gradcam_path"
    )

    if gradcam_path:

        prediction["gradcam_filename"] = (
            os.path.basename(gradcam_path)
        )

    else:

        prediction["gradcam_filename"] = None

    # -----------------------------------------------------
    # CANCER INFORMATION
    # -----------------------------------------------------

    prediction["cancer_info"] = get_cancer_info(
        prediction.get(
            "subclass_prediction"
        )
    )

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

    raw_predictions = get_user_predictions(
        session["user_id"]
    )

    total_predictions = len(
        raw_predictions
    )

    total_benign = sum(
        1
        for prediction in raw_predictions
        if str(
            prediction["binary_prediction"]
        ).strip().lower() == "benign"
    )

    total_malignant = sum(
        1
        for prediction in raw_predictions
        if str(
            prediction["binary_prediction"]
        ).strip().lower() == "malignant"
    )

    latest_prediction = None

    if raw_predictions:

        try:

            latest_prediction = max(
                raw_predictions,
                key=lambda prediction: prediction["id"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            latest_prediction = raw_predictions[0]

    latest_confidence = 0.0

    if latest_prediction is not None:

        try:

            latest_confidence = float(
                latest_prediction[
                    "binary_confidence"
                ] or 0
            )

            if latest_confidence <= 1:
                latest_confidence *= 100

            latest_confidence = max(
                0.0,
                min(100.0, latest_confidence)
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            latest_confidence = 0.0

    return render_template(
        "patient/dashboard.html",
        user_name=session.get("name"),
        total_predictions=total_predictions,
        total_benign=total_benign,
        total_malignant=total_malignant,
        latest_confidence=round(
            latest_confidence,
            2
        )
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

    files = request.files.getlist(
        "images"
    )

    if not files:

        single_file = request.files.get(
            "image"
        )

        if single_file:
            files = [single_file]

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

            file.save(
                image_path
            )

            print(
                "Image saved:",
                image_path
            )

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

            subclass_probabilities = result.get(
                "subclass_probabilities",
                {}
            )

            prediction_id = save_prediction(
    user_id=session["user_id"],
    original_filename=original_filename,
    image_path=image_path,
    gradcam_path=gradcam_path,

    binary_prediction=result["binary_prediction"],
    binary_confidence=result["binary_confidence"],

    subclass_prediction=result["subclass_prediction"],
    subclass_confidence=result["subclass_confidence"],

    subclass_probabilities=json.dumps(
        subclass_probabilities
    ),

    activation_percentage=result.get(
        "activation_percentage"
    ),

    stage=result.get(
        "stage"
    ),

    stage_confidence=result.get(
        "stage_confidence"
    ),

    stage_score=result.get(
        "stage_score"
    ),

    stage_notice=result.get(
        "stage_notice"
    )
)

            batch_prediction_ids.append(
                prediction_id
            )

            print(
                "Prediction saved:",
                prediction_id
            )

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
            print("==========================================")
            print(f"ERROR PROCESSING IMAGE "
                  f"{index}/{total_images}"
    )    
            print(
                "FILE:",
                original_filename)
            print(
                "ERROR TYPE:",
                type(error).__name__
                )
            print(
                "ERROR:",
                str(error)
                )
            print("FULL TRACEBACK:")
            traceback.print_exc()
            print("==========================================")
            failed_images.append(
            original_filename
    )
            continue

    if not batch_prediction_ids:

        flash(
            "Image analysis failed. Please try again.",
            "error"
        )

        return redirect(
            url_for("prediction")
        )

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

# ============================================================
# CANCER INFORMATION
# ============================================================

@app.route("/cancer-information")
def cancer_information():

    check = patient_required()

    if check:
        return check

    return render_template(
        "patient/cancer_info.html",
        cancers=CANCER_INFORMATION
    )


@app.route(
    "/cancer-information/<name>",
    endpoint="cancer_information_detail"
)
def cancer_information_detail(name):

    check = patient_required()

    if check:
        return check

    cancer = CANCER_INFORMATION.get(name)

    if cancer is None:
        return "Cancer information not found", 404

    return render_template(
        "patient/cancer_detail.html",
        cancer=cancer,
        cancer_name=name
    )


# =========================================================
# PROFILE
# =========================================================

@app.route("/profile", methods=["GET", "POST"])
def profile():
    check = patient_required()
    if check:
        return check

    user = get_user_by_id(session["user_id"])

    if request.method == "POST":

        phone = request.form.get("phone", "").strip()
        age = request.form.get("age", "").strip()
        address = request.form.get("address", "").strip()

        update_profile(
            session["user_id"],
            phone,
            age,
            user["gender"] if user["gender"] else "",
            address
        )

        return redirect(url_for("profile"))

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

    patients = get_all_patients()

    raw_predictions = get_all_predictions()

    total_patients = len(
        patients
    )

    total_predictions = len(
        raw_predictions
    )

    total_benign = 0
    total_malignant = 0
    latest_confidence = 0.0

    for prediction_data in raw_predictions:

        prediction_dict = row_to_dict(
            prediction_data
        )

        binary_prediction = str(
            prediction_dict.get(
                "binary_prediction",
                ""
            )
        ).strip().lower()

        if binary_prediction == "benign":

            total_benign += 1

        elif binary_prediction == "malignant":

            total_malignant += 1

    latest_prediction = None

    if raw_predictions:

        try:

            latest_prediction = max(
                raw_predictions,
                key=lambda prediction: prediction["id"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            latest_prediction = raw_predictions[0]

    if latest_prediction is not None:

        latest_prediction = row_to_dict(
            latest_prediction
        )

        try:

            latest_confidence = float(
                latest_prediction.get(
                    "binary_confidence",
                    0
                ) or 0
            )

        except (
            TypeError,
            ValueError
        ):

            latest_confidence = 0.0

        if latest_confidence <= 1:

            latest_confidence *= 100

        latest_confidence = max(
            0.0,
            min(100.0, latest_confidence)
        )

    return render_template(
        "admin/dashboard.html",
        patient_count=total_patients,
        prediction_count=total_predictions,
        total_patients=total_patients,
        total_predictions=total_predictions,
        total_benign=total_benign,
        total_malignant=total_malignant,
        latest_confidence=round(
            latest_confidence,
            2
        )
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
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )