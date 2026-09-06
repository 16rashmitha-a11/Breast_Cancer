import os

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ============================================================
# DATABASE
# ============================================================

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "breast_cancer.db"
)

# ============================================================
# SECRET KEY
# ============================================================

SECRET_KEY = "breast-cancer-project-secret-key-2026"

# ============================================================
# MODEL PATHS
# ============================================================

BINARY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "densenet121_binary_best.keras"
)

SUBCLASS_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "densenet121_subclass_best_v2.keras"
)

# ============================================================
# UPLOAD DIRECTORIES
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

ORIGINAL_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "original"
)

GRADCAM_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "gradcam"
)

# ============================================================
# REPORT DIRECTORY
# ============================================================

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)

# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff"
}

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)