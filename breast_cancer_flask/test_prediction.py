import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# =========================
# MODEL PATHS
# =========================

BINARY_MODEL_PATH = "models/densenet121_binary_best.keras"
SUBCLASS_MODEL_PATH = "models/densenet121_subclass_best_v2.keras"


# =========================
# SUBCLASS NAMES
# =========================

CLASS_NAMES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma"
]


# =========================
# LOAD MODELS
# =========================

print("Loading binary model...")
binary_model = load_model(BINARY_MODEL_PATH)

print("Loading subclass model...")
subclass_model = load_model(SUBCLASS_MODEL_PATH)

print("Models loaded successfully.")

print("Binary input:", binary_model.input_shape)
print("Binary output:", binary_model.output_shape)

print("Subclass input:", subclass_model.input_shape)
print("Subclass output:", subclass_model.output_shape)


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    # Normalize to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# =========================
# PREDICTION
# =========================

def predict_image(image_path):

    image = preprocess_image(image_path)

    # -------------------------
    # Binary prediction
    # -------------------------

    binary_output = binary_model.predict(image, verbose=0)

    binary_probability = float(binary_output[0][0])

    if binary_probability >= 0.5:
        binary_prediction = "Malignant"
        binary_confidence = binary_probability * 100
    else:
        binary_prediction = "Benign"
        binary_confidence = (1 - binary_probability) * 100


    # -------------------------
    # Subclass prediction
    # -------------------------

    subclass_output = subclass_model.predict(image, verbose=0)

    subclass_probabilities = subclass_output[0]

    predicted_index = int(np.argmax(subclass_probabilities))

    subclass_prediction = CLASS_NAMES[predicted_index]

    subclass_confidence = float(
        subclass_probabilities[predicted_index] * 100
    )


    # -------------------------
    # Display
    # -------------------------

    print()
    print("=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)

    print()
    print("BINARY CLASSIFICATION")
    print("---------------------")
    print("Prediction:", binary_prediction)
    print("Confidence:", f"{binary_confidence:.2f}%")

    print()
    print("SUBCLASS CLASSIFICATION")
    print("-----------------------")
    print("Prediction:", subclass_prediction)
    print("Confidence:", f"{subclass_confidence:.2f}%")

    print()
    print("ALL SUBCLASS PROBABILITIES")
    print("--------------------------")

    for name, probability in zip(
        CLASS_NAMES,
        subclass_probabilities
    ):
        print(
            f"{name}: {probability * 100:.2f}%"
        )

    print()
    print("=" * 50)


# =========================
# CHANGE THIS IMAGE PATH
# =========================
import os

IMAGE_PATH = input("Enter the image path: ").strip()

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image file not found.")
else:
    predict_image(IMAGE_PATH)