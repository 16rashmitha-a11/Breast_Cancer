import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.prediction import predict_image


app = Flask(__name__)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ---------------------------------------------------------
# CHECK FILE TYPE
# ---------------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Please select an image."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )

    if not allowed_file(file.filename):

        return render_template(
            "index.html",
            error="Only PNG, JPG and JPEG images are allowed."
        )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:

        result = predict_image(filepath)

        return render_template(
            "result.html",
            result=result,
            image_filename=filename
        )

    except Exception as e:

        print("Prediction error:", e)

        return render_template(
            "index.html",
            error=f"Prediction failed: {str(e)}"
        )


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("BREAST CANCER HISTOPATHOLOGY CLASSIFICATION")
    print("=" * 70)
    print()
    print("Starting Flask website...")
    print()
    print("Open your browser:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )