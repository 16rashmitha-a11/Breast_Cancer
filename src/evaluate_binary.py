import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import matplotlib.pyplot as plt


# ============================================================
# DENSENET121 BINARY CLASSIFICATION - FINAL EVALUATION
# ============================================================

print("=" * 70)
print("DENSENET121 BINARY CLASSIFICATION - FINAL EVALUATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\densenet121_binary_best.keras"

DATASET_PATH = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# FIND ALL IMAGES
# ============================================================

print("\nLoading dataset...")

all_images = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            full_path = os.path.join(root, file)

            all_images.append(full_path)


print("Total images found:", len(all_images))


# ============================================================
# CREATE BINARY LABELS
#
# SOB_B_ = Benign = 0
# SOB_M_ = Malignant = 1
# ============================================================

image_paths = []
labels = []

for path in all_images:

    filename = os.path.basename(path)

    if filename.startswith("SOB_B_"):

        image_paths.append(path)
        labels.append(0)

    elif filename.startswith("SOB_M_"):

        image_paths.append(path)
        labels.append(1)


print("\nBinary classes:")

print("Benign    :", labels.count(0))
print("Malignant :", labels.count(1))

print("Total usable images:", len(image_paths))


# ============================================================
# CONVERT TO NUMPY
# ============================================================

image_paths = np.array(image_paths)
labels = np.array(labels)


# ============================================================
# SAME RANDOM SPLIT AS TRAINING
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ============================================================

np.random.seed(SEED)

indices = np.arange(len(image_paths))

np.random.shuffle(indices)

image_paths = image_paths[indices]
labels = labels[indices]


total = len(image_paths)

train_end = int(0.70 * total)

val_end = int(0.85 * total)


train_paths = image_paths[:train_end]
train_labels = labels[:train_end]

val_paths = image_paths[train_end:val_end]
val_labels = labels[train_end:val_end]

test_paths = image_paths[val_end:]
test_labels = labels[val_end:]


print("\nDataset split:")

print("Training   :", len(train_paths))
print("Validation :", len(val_paths))
print("Testing    :", len(test_paths))


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # DenseNet121 preprocessing
    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image, label


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


print("\nTest dataset ready!")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

results = model.evaluate(
    test_dataset,
    verbose=1
)


print("\nKeras evaluation:")

for name, value in zip(model.metrics_names, results):

    print(f"{name:15s}: {value:.4f}")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_dataset,
    verbose=1
)


# Convert probabilities to 1D
probabilities = predictions.ravel()


# Binary prediction threshold
predicted_labels = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    predicted_labels
)

precision = precision_score(
    test_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    test_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    predicted_labels,
    zero_division=0
)

auc = roc_auc_score(
    test_labels,
    probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    predicted_labels
)


tn, fp, fn, tp = cm.ravel()


# Specificity
specificity = tn / (tn + fp)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy     : {accuracy:.4f}  ({accuracy * 100:.2f}%)")

print(f"Precision    : {precision:.4f}")

print(f"Recall       : {recall:.4f}")

print(f"Specificity  : {specificity:.4f}")

print(f"F1-Score     : {f1:.4f}")

print(f"AUC          : {auc:.4f}")


# ============================================================
# CONFUSION MATRIX VALUES
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n                 Predicted")
print("              Benign  Malignant")

print(
    f"Actual Benign    {tn:5d}    {fp:5d}"
)

print(
    f"Actual Malignant {fn:5d}    {tp:5d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        test_labels,
        predicted_labels,
        target_names=[
            "Benign",
            "Malignant"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "DenseNet121 - Binary Classification\nConfusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Benign", "Malignant"]
)

plt.yticks(
    [0, 1],
    ["Benign", "Malignant"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


print("\nConfusion matrix saved:")
print("confusion_matrix.png")


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    test_labels,
    probabilities
)


plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "DenseNet121 - ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.close()


print("ROC curve saved:")
print("roc_curve.png")


# ============================================================
# SAVE RESULTS TO TEXT FILE
# ============================================================

with open(
    "binary_evaluation_results.txt",
    "w"
) as f:

    f.write(
        "DENSENET121 BINARY CLASSIFICATION\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        f"Total Images      : {len(image_paths)}\n"
    )

    f.write(
        f"Training Images   : {len(train_paths)}\n"
    )

    f.write(
        f"Validation Images : {len(val_paths)}\n"
    )

    f.write(
        f"Testing Images    : {len(test_paths)}\n\n"
    )

    f.write(
        f"Accuracy     : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision    : {precision:.4f}\n"
    )

    f.write(
        f"Recall       : {recall:.4f}\n"
    )

    f.write(
        f"Specificity  : {specificity:.4f}\n"
    )

    f.write(
        f"F1-Score     : {f1:.4f}\n"
    )

    f.write(
        f"AUC          : {auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix\n"
    )

    f.write(
        str(cm)
    )


print("\nResults saved:")
print("binary_evaluation_results.txt")


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print("1. confusion_matrix.png")
print("2. roc_curve.png")
print("3. binary_evaluation_results.txt")

print("\nDone!")