import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

MODEL_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models"

RESULTS_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

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

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

SEED = 42

# ============================================================
# CLASS DEFINITIONS
# ============================================================

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

ALL_CLASSES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma"
]

# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("FINAL HIERARCHICAL BREAST CANCER EVALUATION")
print("=" * 70)

# ============================================================
# CHECK MODELS
# ============================================================

print("\nChecking model files...")

for path in [
    BINARY_MODEL_PATH,
    BENIGN_MODEL_PATH,
    MALIGNANT_MODEL_PATH
]:

    print("\n", path)

    if os.path.exists(path):

        size_mb = os.path.getsize(path) / (1024 * 1024)

        print(
            f"FOUND - {size_mb:.2f} MB"
        )

    else:

        print("ERROR - MODEL NOT FOUND")
        raise FileNotFoundError(path)

# ============================================================
# LOAD MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("\nLoading binary model...")

binary_model = tf.keras.models.load_model(
    BINARY_MODEL_PATH
)

print("Binary model loaded.")

print("\nLoading benign subclass model...")

benign_model = tf.keras.models.load_model(
    BENIGN_MODEL_PATH
)

print("Benign model loaded.")

print("\nLoading malignant subclass model...")

malignant_model = tf.keras.models.load_model(
    MALIGNANT_MODEL_PATH
)

print("Malignant model loaded.")

print("\nModel shapes:")

print(
    "Binary:",
    binary_model.input_shape,
    "->",
    binary_model.output_shape
)

print(
    "Benign:",
    benign_model.input_shape,
    "->",
    benign_model.output_shape
)

print(
    "Malignant:",
    malignant_model.input_shape,
    "->",
    malignant_model.output_shape
)

# ============================================================
# FIND DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

image_paths = []
labels = []

for magnification in [
    "40X",
    "100X",
    "200X",
    "400X"
]:

    mag_path = os.path.join(
        DATASET_ROOT,
        magnification
    )

    if not os.path.isdir(mag_path):
        continue

    for class_name in ALL_CLASSES:

        class_path = os.path.join(
            mag_path,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        for root, dirs, files in os.walk(class_path):

            for file in files:

                if file.lower().endswith(
                    (
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".bmp"
                    )
                ):

                    image_paths.append(
                        os.path.join(
                            root,
                            file
                        )
                    )

                    labels.append(
                        class_name
                    )

print(
    "\nTotal images:",
    len(image_paths)
)

# ============================================================
# CHECK CLASS COUNTS
# ============================================================

print("\nClass distribution:")

for class_name in ALL_CLASSES:

    count = sum(
        1 for x in labels
        if x == class_name
    )

    print(
        f"{class_name:22s}: {count}"
    )

# ============================================================
# CREATE EXACT SAME SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL TEST SPLIT")
print("=" * 70)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(

    image_paths,
    labels,

    test_size=0.30,

    random_state=SEED,

    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(

    temp_paths,
    temp_labels,

    test_size=0.50,

    random_state=SEED,

    stratify=temp_labels
)

print("\nTraining   :", len(train_paths))
print("Validation :", len(val_paths))
print("Testing    :", len(test_paths))

# ============================================================
# LOAD TEST IMAGES
# ============================================================

def load_test_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image


print("\n" + "=" * 70)
print("PREPARING TEST DATA")
print("=" * 70)

test_images = []

for i, path in enumerate(test_paths):

    if i % 200 == 0:

        print(
            f"Loading {i}/{len(test_paths)}"
        )

    test_images.append(
        load_test_image(path)
    )

test_images = tf.stack(
    test_images
)

print(
    "\nTest tensor shape:",
    test_images.shape
)

# ============================================================
# TRUE LABELS
# ============================================================

true_labels = np.array([
    ALL_CLASSES.index(x)
    for x in test_labels
])

# ============================================================
# BINARY TRUE LABELS
# ============================================================

true_binary = np.array([

    0 if x in BENIGN_CLASSES else 1

    for x in test_labels

])

# ============================================================
# BINARY MODEL PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - BINARY PREDICTION")
print("=" * 70)

binary_probs = binary_model.predict(
    test_images,
    batch_size=BATCH_SIZE,
    verbose=1
)

binary_probs = binary_probs.reshape(-1)

# IMPORTANT:
# Binary model output was trained as:
# 0 = Benign
# 1 = Malignant

binary_predictions = (
    binary_probs >= 0.5
).astype(int)

binary_accuracy = accuracy_score(
    true_binary,
    binary_predictions
)

print(
    "\nBinary Test Accuracy:",
    f"{binary_accuracy * 100:.2f}%"
)

# ============================================================
# STAGE 2 - SUBCLASS PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - SUBCLASS PREDICTION")
print("=" * 70)

final_predictions = np.zeros(
    len(test_images),
    dtype=int
)

# ------------------------------------------------------------
# BENIGN IMAGES
# ------------------------------------------------------------

benign_indices = np.where(
    binary_predictions == 0
)[0]

print(
    "\nImages routed to BENIGN model:",
    len(benign_indices)
)

if len(benign_indices) > 0:

    benign_images = tf.gather(
        test_images,
        benign_indices
    )

    benign_probs = benign_model.predict(
        benign_images,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    benign_predictions = np.argmax(
        benign_probs,
        axis=1
    )

    for i, pred in zip(
        benign_indices,
        benign_predictions
    ):

        predicted_class = BENIGN_CLASSES[pred]

        final_predictions[i] = (
            ALL_CLASSES.index(
                predicted_class
            )
        )

# ------------------------------------------------------------
# MALIGNANT IMAGES
# ------------------------------------------------------------

malignant_indices = np.where(
    binary_predictions == 1
)[0]

print(
    "\nImages routed to MALIGNANT model:",
    len(malignant_indices)
)

if len(malignant_indices) > 0:

    malignant_images = tf.gather(
        test_images,
        malignant_indices
    )

    malignant_probs = malignant_model.predict(
        malignant_images,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    malignant_predictions = np.argmax(
        malignant_probs,
        axis=1
    )

    for i, pred in zip(
        malignant_indices,
        malignant_predictions
    ):

        predicted_class = MALIGNANT_CLASSES[pred]

        final_predictions[i] = (
            ALL_CLASSES.index(
                predicted_class
            )
        )

# ============================================================
# FINAL END-TO-END ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("FINAL HIERARCHICAL TEST RESULTS")
print("=" * 70)

overall_accuracy = accuracy_score(
    true_labels,
    final_predictions
)

precision_macro = precision_score(
    true_labels,
    final_predictions,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    true_labels,
    final_predictions,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    true_labels,
    final_predictions,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    true_labels,
    final_predictions,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    true_labels,
    final_predictions,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    true_labels,
    final_predictions,
    average="weighted",
    zero_division=0
)

print(
    "\nOverall Accuracy     :",
    f"{overall_accuracy:.4f}",
    f"({overall_accuracy * 100:.2f}%)"
)

print(
    "Macro Precision      :",
    f"{precision_macro:.4f}"
)

print(
    "Macro Recall         :",
    f"{recall_macro:.4f}"
)

print(
    "Macro F1-Score       :",
    f"{f1_macro:.4f}"
)

print(
    "Weighted Precision   :",
    f"{precision_weighted:.4f}"
)

print(
    "Weighted Recall      :",
    f"{recall_weighted:.4f}"
)

print(
    "Weighted F1-Score    :",
    f"{f1_weighted:.4f}"
)

# ============================================================
# PER CLASS RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

report = classification_report(
    true_labels,
    final_predictions,
    labels=list(range(len(ALL_CLASSES))),
    target_names=ALL_CLASSES,
    zero_division=0
)

print("\n")
print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    final_predictions,
    labels=list(range(len(ALL_CLASSES)))
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)

# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for i, class_name in enumerate(
    ALL_CLASSES
):

    total = np.sum(
        true_labels == i
    )

    correct = cm[i, i]

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:22s}: "
        f"{accuracy * 100:.2f}%"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(
    RESULTS_DIR,
    "hierarchical_final_test_results.txt"
)

with open(
    result_file,
    "w"
) as f:

    f.write(
        "FINAL HIERARCHICAL BREAST CANCER TEST RESULTS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test images: {len(test_paths)}\n\n"
    )

    f.write(
        f"Binary test accuracy: "
        f"{binary_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        f"Hierarchical overall accuracy: "
        f"{overall_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro precision: "
        f"{precision_macro:.4f}\n"
    )

    f.write(
        f"Macro recall: "
        f"{recall_macro:.4f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{f1_macro:.4f}\n"
    )

    f.write(
        f"Weighted precision: "
        f"{precision_weighted:.4f}\n"
    )

    f.write(
        f"Weighted recall: "
        f"{recall_weighted:.4f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{f1_weighted:.4f}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        report
    )

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        np.array2string(cm)
    )

    f.write(
        "\n\nPER-CLASS ACCURACY\n"
    )

    for i, class_name in enumerate(
        ALL_CLASSES
    ):

        total = np.sum(
            true_labels == i
        )

        correct = cm[i, i]

        accuracy = (
            correct / total
            if total > 0
            else 0
        )

        f.write(
            f"{class_name}: "
            f"{accuracy * 100:.2f}%\n"
        )

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    result_file
)

print("\nDO NOT TRAIN ANOTHER MODEL YET.")
print("Use these final test results to decide the next step.")