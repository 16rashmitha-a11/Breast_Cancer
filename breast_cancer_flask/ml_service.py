import os
import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    BINARY_MODEL_PATH,
    SUBCLASS_MODEL_PATH,
)

from gradcam_service import make_gradcam
from stage_estimator import estimate_stage


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma",
]


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading binary model...")

binary_model = tf.keras.models.load_model(
    BINARY_MODEL_PATH,
    compile=False,
)

print("Loading subclass model...")

subclass_model = tf.keras.models.load_model(
    SUBCLASS_MODEL_PATH,
    compile=False,
)

print("Both models loaded successfully.")

print(
    "Binary input:",
    binary_model.input_shape
)

print(
    "Binary output:",
    binary_model.output_shape
)

print(
    "Subclass input:",
    subclass_model.input_shape
)

print(
    "Subclass output:",
    subclass_model.output_shape
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = (
        tf.keras.applications.densenet.preprocess_input(
            image_array
        )
    )

    return image_array


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    print("")
    print("==========================================")
    print("STARTING IMAGE ANALYSIS")
    print("==========================================")

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    )

    print(
        "Image preprocessing completed."
    )


    # ========================================================
    # 1. BINARY CLASSIFICATION
    # ========================================================

    print(
        "Running binary model..."
    )

    binary_output = binary_model.predict(
        image,
        verbose=0
    )

    binary_values = np.asarray(
        binary_output
    ).reshape(-1)

    if binary_values.size == 0:

        raise ValueError(
            "Binary model returned an empty prediction."
        )

    binary_probability = float(
        binary_values[0]
    )

    binary_probability = float(
        np.clip(
            binary_probability,
            0.0,
            1.0
        )
    )


    if binary_probability >= 0.5:

        binary_prediction = "Malignant"

        binary_confidence = (
            binary_probability * 100.0
        )

    else:

        binary_prediction = "Benign"

        binary_confidence = (
            (1.0 - binary_probability)
            * 100.0
        )


    print(
        "Binary:",
        binary_prediction
    )

    print(
        "Binary confidence:",
        f"{binary_confidence:.2f}%"
    )


    # ========================================================
    # 2. SUBTYPE CLASSIFICATION
    # ========================================================

    print(
        "Running subclass model..."
    )

    subclass_output = subclass_model.predict(
        image,
        verbose=0
    )

    subclass_array = np.asarray(
        subclass_output
    )


    if subclass_array.ndim == 1:

        subclass_probabilities = (
            subclass_array
        )

    elif subclass_array.ndim >= 2:

        subclass_probabilities = (
            subclass_array[0]
        )

    else:

        raise ValueError(
            "Subclass model returned an invalid output."
        )


    subclass_probabilities = np.asarray(
        subclass_probabilities,
        dtype=np.float32
    ).reshape(-1)


    if (
        subclass_probabilities.size
        != len(CLASS_NAMES)
    ):

        raise ValueError(
            "Subclass model returned "
            f"{subclass_probabilities.size} classes, "
            f"but {len(CLASS_NAMES)} class names "
            "are configured."
        )


    predicted_index = int(
        np.argmax(
            subclass_probabilities
        )
    )


    subclass_prediction = (
        CLASS_NAMES[
            predicted_index
        ]
    )


    subclass_confidence = (
        float(
            subclass_probabilities[
                predicted_index
            ]
        )
        * 100.0
    )


    probabilities = {}


    for name, probability in zip(
        CLASS_NAMES,
        subclass_probabilities
    ):

        probabilities[name] = round(
            float(probability) * 100.0,
            2
        )


    print(
        "Subclass:",
        subclass_prediction
    )

    print(
        "Subclass confidence:",
        f"{subclass_confidence:.2f}%"
    )


    # ========================================================
    # 3. GRAD-CAM
    # ========================================================

    gradcam_path = None

    activation_percentage = None

    gradcam_error = None


    try:

        print(
            "Generating Grad-CAM..."
        )


        gradcam_result = make_gradcam(
            subclass_model,
            image_path,
            predicted_index
        )


        # ----------------------------------------------------
        # NEW GRAD-CAM FORMAT
        # ----------------------------------------------------

        if isinstance(
            gradcam_result,
            dict
        ):

            gradcam_path = (
                gradcam_result.get(
                    "gradcam_path"
                )
            )

            activation_percentage = (
                gradcam_result.get(
                    "activation_percentage"
                )
            )

        # ----------------------------------------------------
        # OLD GRAD-CAM FORMAT
        # ----------------------------------------------------

        else:

            gradcam_path = (
                gradcam_result
            )


        if gradcam_path:

            gradcam_path = os.path.basename(
                str(gradcam_path)
            )


            gradcam_folder = os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "uploads",
                "gradcam"
            )


            full_gradcam_path = os.path.join(
                gradcam_folder,
                gradcam_path
            )


            if os.path.exists(
                full_gradcam_path
            ):

                print(
                    "Grad-CAM created:",
                    full_gradcam_path
                )

            else:

                print(
                    "Grad-CAM file was not found:"
                )

                print(
                    full_gradcam_path
                )

                gradcam_path = None


        print(
            "Grad-CAM activation extent:",
            activation_percentage
        )


    except Exception as error:

        gradcam_error = repr(
            error
        )

        print(
            "WARNING: Grad-CAM failed."
        )

        print(
            "Grad-CAM error:",
            gradcam_error
        )

        # IMPORTANT:
        # Grad-CAM failure must NOT
        # cancel the prediction.

        gradcam_path = None

        activation_percentage = None


    # ========================================================
    # 4. EXPERIMENTAL AI STAGE ESTIMATION
    # ========================================================

    print(
        "Calculating experimental stage..."
    )


    try:

        stage_result = estimate_stage(

            binary_prediction=(
                binary_prediction
            ),

            binary_confidence=(
                binary_confidence
            ),

            subclass_confidence=(
                subclass_confidence
            ),

            activation_percentage=(
                activation_percentage
            )
        )


    except Exception as error:

        print(
            "WARNING: Stage estimation failed."
        )

        print(
            "Stage error:",
            repr(error)
        )


        stage_result = {

            "stage": "Not Available",

            "score": None,

            "confidence": None,

            "activation_percentage": (
                activation_percentage
            ),

            "notice": (
                "Experimental AI-assisted "
                "stage estimation could not "
                "be calculated."
            )
        }


    # ========================================================
    # 5. FINAL RESULT
    # ========================================================

    result = {

        "binary_prediction": (
            binary_prediction
        ),

        "binary_confidence": round(
            binary_confidence,
            2
        ),

        "subclass_prediction": (
            subclass_prediction
        ),

        "subclass_confidence": round(
            subclass_confidence,
            2
        ),

        "subclass_probabilities": (
            probabilities
        ),

        "gradcam_path": (
            gradcam_path
        ),

        "gradcam_generated": (
            gradcam_path is not None
        ),

        "gradcam_error": (
            gradcam_error
        ),

        "activation_percentage": (
            stage_result.get(
                "activation_percentage"
            )
        ),

        "stage": (
            stage_result.get(
                "stage"
            )
        ),

        "stage_confidence": (
            stage_result.get(
                "confidence"
            )
        ),

        "stage_score": (
            stage_result.get(
                "score"
            )
        ),

        "stage_notice": (
            stage_result.get(
                "notice"
            )
        )
    }


    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print("")
    print(
        "=========================================="
    )

    print(
        "PREDICTION COMPLETED SUCCESSFULLY"
    )

    print(
        "Binary:",
        result["binary_prediction"]
    )

    print(
        "Binary confidence:",
        result["binary_confidence"]
    )

    print(
        "Subclass:",
        result["subclass_prediction"]
    )

    print(
        "Subclass confidence:",
        result["subclass_confidence"]
    )

    print(
        "Grad-CAM:",
        result["gradcam_path"]
    )

    print(
        "Activation:",
        result["activation_percentage"]
    )

    print(
        "Stage:",
        result["stage"]
    )

    print(
        "Stage score:",
        result["stage_score"]
    )

    print(
        "=========================================="
    )

    return result