import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

MODEL_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_best_v2.keras"

RESULTS_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\results"

os.makedirs(RESULTS_DIR, exist_ok=True)

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
    "tubular_adenoma"
]

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("8-CLASS DETAILED ERROR ANALYSIS")
print("=" * 75)

print()
print("Model:")
print(MODEL_PATH)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nModel not found:\n" + MODEL_PATH
    )

if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        "\nDataset not found:\n" + DATASET_DIR
    )

# ============================================================
# SEARCH DATASET
# ============================================================

print()
print("=" * 75)
print("SEARCHING DATASET")
print("=" * 75)

magnifications = [
    "40X",
    "100X",
    "200X",
    "400X"
]

valid_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)

image_paths = []
image_labels = []

for magnification in magnifications:

    magnification_dir = os.path.join(
        DATASET_DIR,
        magnification
    )

    if not os.path.isdir(magnification_dir):

        print(
            "WARNING - missing:",
            magnification_dir
        )

        continue

    print("Found:", magnification)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = os.path.join(
            magnification_dir,
            class_name
        )

        if not os.path.isdir(class_dir):

            print(
                "WARNING - missing:",
                class_dir
            )

            continue

        for root, dirs, files in os.walk(class_dir):

            for filename in files:

                if filename.lower().endswith(
                    valid_extensions
                ):

                    path = os.path.join(
                        root,
                        filename
                    )

                    image_paths.append(path)
                    image_labels.append(class_index)

image_paths = np.array(image_paths)
image_labels = np.array(image_labels)

print()
print("Total images:", len(image_paths))

if len(image_paths) == 0:

    raise RuntimeError(
        "No images were found."
    )

# ============================================================
# CREATE SAME 70/15/15 SPLIT
# ============================================================

print()
print("=" * 75)
print("CREATING SAME DATASET SPLIT")
print("=" * 75)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(

    image_paths,
    image_labels,

    test_size=0.30,

    random_state=SEED,

    stratify=image_labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(

    temp_paths,
    temp_labels,

    test_size=0.50,

    random_state=SEED,

    stratify=temp_labels
)

print()
print("Training   :", len(train_paths))
print("Validation :", len(val_paths))
print("Testing    :", len(test_paths))

# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path, label):

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

    return image, label


# ============================================================
# TEST DATASET
# ============================================================

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(
    BATCH_SIZE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)

print()
print("Test dataset created.")

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING BEST STAGE 1 MODEL")
print("=" * 75)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("Model loaded successfully!")

print(
    "Input shape :",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)

# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 75)
print("GENERATING TEST PREDICTIONS")
print("=" * 75)

probabilities = model.predict(
    test_ds,
    verbose=1
)

predicted_labels = np.argmax(
    probabilities,
    axis=1
)

true_labels = test_labels

# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print()
print("=" * 75)
print("OVERALL RESULT")
print("=" * 75)

print()
print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(NUM_CLASSES)
)

print()
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print()

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    " " * 22,
    " ".join(
        f"{i:^5}"
        for i in range(NUM_CLASSES)
    )
)

for i in range(NUM_CLASSES):

    print(
        f"{i:<22}",
        " ".join(
            f"{value:^5}"
            for value in cm[i]
        )
    )

# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("=" * 75)
print("PER-CLASS ANALYSIS")
print("=" * 75)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4
)

print()
print(report)

# ============================================================
# MOST CONFUSED CLASS PAIRS
# ============================================================

print()
print("=" * 75)
print("MOST CONFUSED CLASS PAIRS")
print("=" * 75)

confusions = []

for actual in range(NUM_CLASSES):

    for predicted in range(NUM_CLASSES):

        if actual == predicted:
            continue

        count = cm[actual][predicted]

        if count > 0:

            confusions.append(
                (
                    count,
                    actual,
                    predicted
                )
            )

confusions.sort(
    reverse=True
)

print()

for count, actual, predicted in confusions[:15]:

    print(
        f"{CLASS_NAMES[actual]:<22}"
        f" -> "
        f"{CLASS_NAMES[predicted]:<22}"
        f": {count}"
    )

# ============================================================
# MISCLASSIFIED IMAGE LIST
# ============================================================

print()
print("=" * 75)
print("MISCLASSIFIED IMAGES")
print("=" * 75)

misclassified_indices = np.where(
    true_labels != predicted_labels
)[0]

print()
print(
    "Total misclassified:",
    len(misclassified_indices)
)

misclassified_file = os.path.join(
    RESULTS_DIR,
    "subclass_misclassified_images.txt"
)

with open(
    misclassified_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "8-CLASS MISCLASSIFIED TEST IMAGES\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        f"Total test images: "
        f"{len(test_paths)}\n"
    )

    f.write(
        f"Misclassified: "
        f"{len(misclassified_indices)}\n\n"
    )

    for index in misclassified_indices:

        actual = true_labels[index]

        predicted = predicted_labels[index]

        confidence = probabilities[index][predicted]

        path = test_paths[index]

        f.write(
            f"Actual      : "
            f"{CLASS_NAMES[actual]}\n"
        )

        f.write(
            f"Predicted   : "
            f"{CLASS_NAMES[predicted]}\n"
        )

        f.write(
            f"Confidence  : "
            f"{confidence * 100:.2f}%\n"
        )

        f.write(
            f"Image       : "
            f"{path}\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

print()
print(
    "Misclassified image list saved:"
)

print(
    misclassified_file
)

# ============================================================
# SAVE COMPLETE ANALYSIS
# ============================================================

analysis_file = os.path.join(
    RESULTS_DIR,
    "subclass_detailed_error_analysis.txt"
)

with open(
    analysis_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "DENSENET121 8-CLASS DETAILED ERROR ANALYSIS\n"
    )

    f.write(
        "============================================\n\n"
    )

    f.write(
        f"Model:\n{MODEL_PATH}\n\n"
    )

    f.write(
        f"Total images: {len(image_paths)}\n"
    )

    f.write(
        f"Training: {len(train_paths)}\n"
    )

    f.write(
        f"Validation: {len(val_paths)}\n"
    )

    f.write(
        f"Testing: {len(test_paths)}\n\n"
    )

    f.write(
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "----------------\n"
    )

    for row in cm:

        f.write(
            " ".join(
                str(x)
                for x in row
            )
            + "\n"
        )

    f.write(
        "\n\nCLASSIFICATION REPORT\n"
    )

    f.write(
        "---------------------\n"
    )

    f.write(
        report
    )

    f.write(
        "\n\nMOST CONFUSED PAIRS\n"
    )

    f.write(
        "--------------------\n"
    )

    for count, actual, predicted in confusions[:15]:

        f.write(
            f"{CLASS_NAMES[actual]}"
            f" -> "
            f"{CLASS_NAMES[predicted]}"
            f": {count}\n"
        )

# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 75)
print("ERROR ANALYSIS COMPLETED")
print("=" * 75)

print()
print("Generated files:")

print()
print(
    "1.",
    analysis_file
)

print(
    "2.",
    misclassified_file
)

print()
print("BEST CURRENT MODEL:")
print(MODEL_PATH)

print()
print(
    f"Current test accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()
print("=" * 75)