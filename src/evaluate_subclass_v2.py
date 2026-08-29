import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

MODEL_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_best_v2.keras"

RESULTS_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

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

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 75)
print("DENSENET121 - FINAL 8-CLASS EVALUATION")
print("=" * 75)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Stage 1 best model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# COLLECT DATASET
# ============================================================

print("\n" + "=" * 75)
print("LOADING DATASET")
print("=" * 75)

image_paths = []
labels = []

magnifications = ["40X", "100X", "200X", "400X"]

for magnification in magnifications:

    mag_path = os.path.join(DATASET_PATH, magnification)

    if not os.path.isdir(mag_path):
        continue

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(mag_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for root, dirs, files in os.walk(class_path):

            for file in files:

                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):

                    image_paths.append(
                        os.path.join(root, file)
                    )

                    labels.append(class_index)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("\nTotal images:", len(image_paths))

print("\nClass distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(labels == i)

    print(
        f"{class_name:20s}: {count}"
    )

# ============================================================
# SAME SPLIT AS TRAINING
# ============================================================

print("\n" + "=" * 75)
print("CREATING TEST SPLIT")
print("=" * 75)

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
# IMAGE PREPROCESSING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # DenseNet preprocessing
    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image, label


# ============================================================
# TEST DATASET
# ============================================================

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("\nTest dataset created successfully!")

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

results = model.evaluate(
    test_ds,
    verbose=1
)

print("\nKeras evaluation:")

for name, value in zip(
    model.metrics_names,
    results
):
    print(
        f"{name:15s}: {value:.4f}"
    )

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_ds,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_labels

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

precision_macro = precision_score(
    true_classes,
    predicted_classes,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    true_classes,
    predicted_classes,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    true_classes,
    predicted_classes,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    true_classes,
    predicted_classes,
    average="weighted",
    zero_division=0
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL 8-CLASS TEST RESULTS")
print("=" * 75)

print(
    f"\nOverall Accuracy       : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro Precision        : {precision_macro:.4f}"
)

print(
    f"Macro Recall           : {recall_macro:.4f}"
)

print(
    f"Macro F1-Score         : {f1_macro:.4f}"
)

print(
    f"Weighted Precision     : {precision_weighted:.4f}"
)

print(
    f"Weighted Recall        : {recall_weighted:.4f}"
)

print(
    f"Weighted F1-Score      : {f1_weighted:.4f}"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n")
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print("\nClasses:")
print(CLASS_NAMES)

print("\n")
print(cm)

# ============================================================
# PER CLASS ACCURACY
# ============================================================

print("\n")
print("=" * 75)
print("PER-CLASS ACCURACY")
print("=" * 75)

for i, class_name in enumerate(CLASS_NAMES):

    total = np.sum(
        true_classes == i
    )

    correct = cm[i, i]

    class_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:20s}: "
        f"{class_accuracy * 100:.2f}%"
    )

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)

# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(
    RESULTS_DIR,
    "subclass_v2_evaluation_results.txt"
)

with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "DENSENET121 - 8 CLASS BREAST CANCER\n"
    )

    f.write(
        "SUBCLASS FINAL TEST EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Overall Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)\n"
    )

    f.write(
        f"Macro Precision: "
        f"{precision_macro:.4f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{recall_macro:.4f}\n"
    )

    f.write(
        f"Macro F1-Score: "
        f"{f1_macro:.4f}\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{precision_weighted:.4f}\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{recall_weighted:.4f}\n"
    )

    f.write(
        f"Weighted F1-Score: "
        f"{f1_weighted:.4f}\n\n"
    )

    f.write("CONFUSION MATRIX\n")
    f.write("=" * 70 + "\n")

    f.write(
        np.array2string(cm)
    )

    f.write("\n\nCLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")

    f.write(report)

print("\n")
print("=" * 75)
print("EVALUATION COMPLETED")
print("=" * 75)

print("\nResults saved to:")
print(result_file)