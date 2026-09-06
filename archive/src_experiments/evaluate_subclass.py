import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


# ==============================================================
# DENSENET121 - 8 CLASS SUBCLASS FINAL EVALUATION
# ==============================================================

print("=" * 70)
print("DENSENET121 - 8 CLASS BREAST CANCER SUBCLASS EVALUATION")
print("=" * 70)


# ==============================================================
# PATHS
# ==============================================================

DATASET_PATH = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

MODEL_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_best.keras"

OUTPUT_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================
# CLASS NAMES
# ==============================================================

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

print("\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")


# ==============================================================
# LOAD MODEL
# ==============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ==============================================================
# COLLECT DATASET
# ==============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

image_paths = []
labels = []

magnifications = ["40X", "100X", "200X", "400X"]

for magnification in magnifications:

    mag_path = os.path.join(
        DATASET_PATH,
        magnification
    )

    if not os.path.exists(mag_path):
        continue

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            mag_path,
            class_name
        )

        if not os.path.exists(class_path):
            continue

        for root, dirs, files in os.walk(class_path):

            for file in files:

                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp")
                ):

                    image_paths.append(
                        os.path.join(root, file)
                    )

                    labels.append(class_index)


print("\nTotal images found:", len(image_paths))


# ==============================================================
# CLASS DISTRIBUTION
# ==============================================================

print("\nClass distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = sum(
        1 for label in labels
        if label == i
    )

    print(f"{class_name:20s}: {count}")


# ==============================================================
# CREATE SAME TEST SPLIT
# ==============================================================

from sklearn.model_selection import train_test_split


image_paths = np.array(image_paths)
labels = np.array(labels)


# First split:
# 70% training
# 30% temporary

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)


# Second split:
# 15% validation
# 15% testing

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)


print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print(f"Training   : {len(train_paths)}")
print(f"Validation : {len(val_paths)}")
print(f"Testing    : {len(test_paths)}")


# ==============================================================
# IMAGE LOADING FUNCTION
# ==============================================================

IMG_SIZE = (224, 224)


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

    image = image / 255.0

    return image, label


# ==============================================================
# TEST DATASET
# ==============================================================

BATCH_SIZE = 16

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


print("\nTest dataset created successfully!")


# ==============================================================
# MODEL EVALUATION
# ==============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

results = model.evaluate(
    test_ds,
    verbose=1
)

print("\nKeras evaluation:")

for metric_name, value in zip(
    model.metrics_names,
    results
):

    print(
        f"{metric_name:15s}: {value:.4f}"
    )


# ==============================================================
# PREDICTIONS
# ==============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_ds,
    verbose=1
)


predicted_labels = np.argmax(
    predictions,
    axis=1
)


true_labels = test_labels


# ==============================================================
# OVERALL METRICS
# ==============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


macro_precision = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)


# ==============================================================
# FINAL RESULTS
# ==============================================================

print("\n")
print("=" * 70)
print("FINAL 8-CLASS RESULTS")
print("=" * 70)

print(
    f"\nOverall Accuracy       : {accuracy:.4f} ({accuracy*100:.2f}%)"
)

print(
    f"Weighted Precision     : {precision:.4f}"
)

print(
    f"Weighted Recall        : {recall:.4f}"
)

print(
    f"Weighted F1-Score      : {f1:.4f}"
)

print(
    f"\nMacro Precision        : {macro_precision:.4f}"
)

print(
    f"Macro Recall           : {macro_recall:.4f}"
)

print(
    f"Macro F1-Score         : {macro_f1:.4f}"
)


# ==============================================================
# CONFUSION MATRIX
# ==============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ==============================================================
# SAVE CONFUSION MATRIX IMAGE
# ==============================================================

plt.figure(
    figsize=(12, 10)
)

plt.imshow(cm)

plt.title(
    "Densenet121 - 8 Class Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "subclass_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved:"
)

print(cm_path)


# ==============================================================
# CLASSIFICATION REPORT
# ==============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ==============================================================
# PER-CLASS ACCURACY
# ==============================================================

print("\n")
print("=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    total = np.sum(
        true_labels == i
    )

    correct = np.sum(
        (true_labels == i) &
        (predicted_labels == i)
    )

    class_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:20s}: "
        f"{class_accuracy*100:.2f}%"
    )


# ==============================================================
# SAVE RESULTS
# ==============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "subclass_evaluation_results.txt"
)

with open(
    results_path,
    "w"
) as f:

    f.write(
        "DENSENET121 - 8 CLASS BREAST CANCER SUBCLASS EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total images: {len(image_paths)}\n"
    )

    f.write(
        f"Training images: {len(train_paths)}\n"
    )

    f.write(
        f"Validation images: {len(val_paths)}\n"
    )

    f.write(
        f"Testing images: {len(test_paths)}\n\n"
    )

    f.write(
        f"Overall Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Weighted Precision: {precision:.4f}\n"
    )

    f.write(
        f"Weighted Recall: {recall:.4f}\n"
    )

    f.write(
        f"Weighted F1: {f1:.4f}\n"
    )

    f.write(
        f"Macro Precision: {macro_precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: {macro_recall:.4f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.4f}\n\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nCLASSIFICATION REPORT\n"
    )

    f.write(
        report
    )


print(
    "\nResults saved:"
)

print(results_path)


# ==============================================================
# COMPLETED
# ==============================================================

print("\n")
print("=" * 70)
print("8-CLASS EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(
    "1. results/subclass_confusion_matrix.png"
)

print(
    "2. results/subclass_evaluation_results.txt"
)

print("\n")