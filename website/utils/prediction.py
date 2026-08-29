import os
import numpy as np
import tensorflow as tf
from PIL import Image


# =========================================================
# PATH CONFIGURATION
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

WEBSITE_DIR = os.path.dirname(CURRENT_DIR)

PROJECT_DIR = os.path.dirname(WEBSITE_DIR)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)


# =========================================================
# MODEL PATHS
# =========================================================

BINARY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "densenet121_binary_best.keras"
)

BENIGN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "densenet121_benign_subclass_finetuned_best.keras"
)

MALIGNANT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "densenet121_malignant_subclass_finetuned_best.keras"
)


# =========================================================
# CLASS NAMES
# =========================================================

BENIGN_CLASSES = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma"
]


MALIGNANT_CLASSES = [
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma"
]


# =========================================================
# LOAD MODELS
# =========================================================

print()
print("=" * 70)
print("LOADING HIERARCHICAL BREAST CANCER MODELS")
print("=" * 70)

print()
print("Binary model:")
print(BINARY_MODEL_PATH)

print()
print("Benign subclass model:")
print(BENIGN_MODEL_PATH)

print()
print("Malignant subclass model:")
print(MALIGNANT_MODEL_PATH)


# Check models before loading

for model_path in [
    BINARY_MODEL_PATH,
    BENIGN_MODEL_PATH,
    MALIGNANT_MODEL_PATH
]:

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )


print()
print("Loading binary model...")

binary_model = tf.keras.models.load_model(
    BINARY_MODEL_PATH,
    compile=False
)

print("Binary model loaded.")


print()
print("Loading benign subclass model...")

benign_model = tf.keras.models.load_model(
    BENIGN_MODEL_PATH,
    compile=False
)

print("Benign model loaded.")


print()
print("Loading malignant subclass model...")

malignant_model = tf.keras.models.load_model(
    MALIGNANT_MODEL_PATH,
    compile=False
)

print("Malignant model loaded.")


print()
print("=" * 70)
print("ALL MODELS LOADED SUCCESSFULLY")
print("=" * 70)
print()


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

IMAGE_SIZE = (224, 224)


def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# BINARY PREDICTION
# =========================================================

def predict_binary(image):

    prediction = binary_model.predict(
        image,
        verbose=0
    )

    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    # Your binary model uses:
    # 0 = Benign
    # 1 = Malignant

    if probability >= 0.5:

        binary_class = "Malignant"

        binary_confidence = probability * 100

    else:

        binary_class = "Benign"

        binary_confidence = (1.0 - probability) * 100

    return (
        binary_class,
        binary_confidence,
        probability
    )


# =========================================================
# SUBCLASS PREDICTION
# =========================================================

def predict_subclass(
    image,
    binary_class
):

    if binary_class == "Benign":

        probabilities = benign_model.predict(
            image,
            verbose=0
        )[0]

        class_names = BENIGN_CLASSES

    else:

        probabilities = malignant_model.predict(
            image,
            verbose=0
        )[0]

        class_names = MALIGNANT_CLASSES

    predicted_index = int(
        np.argmax(probabilities)
    )

    subclass = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    ) * 100

    return (
        subclass,
        confidence,
        probabilities
    )


# =========================================================
# FORMAT CLASS NAME
# =========================================================

def format_class_name(class_name):

    return class_name.replace(
        "_",
        " "
    ).title()


# =========================================================
# COMPLETE HIERARCHICAL PREDICTION
# =========================================================

def predict_image(image_path):

    print()
    print("=" * 70)
    print("ANALYZING IMAGE")
    print("=" * 70)

    print("Image:", image_path)

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    image = load_image(image_path)

    # -----------------------------------------------------
    # STAGE 1
    # -----------------------------------------------------

    print()
    print("Stage 1: Binary classification...")

    (
        binary_class,
        binary_confidence,
        raw_probability
    ) = predict_binary(image)

    print(
        f"Binary result: "
        f"{binary_class} "
        f"({binary_confidence:.2f}%)"
    )

    # -----------------------------------------------------
    # STAGE 2
    # -----------------------------------------------------

    print()
    print("Stage 2: Subclass classification...")

    (
        subclass,
        subclass_confidence,
        subclass_probabilities
    ) = predict_subclass(
        image,
        binary_class
    )

    print(
        f"Subclass result: "
        f"{subclass} "
        f"({subclass_confidence:.2f}%)"
    )

    # -----------------------------------------------------
    # ALL SUBCLASS PROBABILITIES
    # -----------------------------------------------------

    if binary_class == "Benign":

        class_names = BENIGN_CLASSES

    else:

        class_names = MALIGNANT_CLASSES

    subclass_scores = []

    for name, probability in zip(
        class_names,
        subclass_probabilities
    ):

        subclass_scores.append({
            "name": format_class_name(name),
            "score": float(probability) * 100
        })

    subclass_scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    result = {

        "binary_class": binary_class,

        "binary_display": binary_class,

        "binary_confidence":
            round(
                binary_confidence,
                2
            ),

        "subclass":
            format_class_name(
                subclass
            ),

        "subclass_raw":
            subclass,

        "subclass_confidence":
            round(
                subclass_confidence,
                2
            ),

        "subclass_scores":
            subclass_scores,

        "model_name":
            "DenseNet121",

        "system":
            "Hierarchical Two-Stage Classification"

    }

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        "Binary:",
        result["binary_display"]
    )

    print(
        "Subclass:",
        result["subclass"]
    )

    print(
        "Subclass confidence:",
        result["subclass_confidence"],
        "%"
    )

    return result