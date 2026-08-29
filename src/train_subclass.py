import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 20

NUM_CLASSES = 8

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

WEIGHTS_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\weights\densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"

MODEL_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models"
RESULTS_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "densenet121_subclass_best_v2.keras"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "densenet121_subclass_final_v2.keras"
)

CLASS_NAMES_FILE = os.path.join(
    MODEL_DIR,
    "subclass_class_names_v2.txt"
)

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

CLASS_TO_ID = {
    name: i for i, name in enumerate(CLASS_NAMES)
}

# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 75)
print("DENSENET121 - IMPROVED 8-CLASS BREAST CANCER CLASSIFICATION")
print("=" * 75)

print("\nDataset:")
print(DATASET_DIR)

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"\nDataset folder not found:\n{DATASET_DIR}"
    )

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"\nDenseNet121 weights not found:\n{WEIGHTS_PATH}"
    )

print("\nLocal DenseNet weights found:")
print(WEIGHTS_PATH)

# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

image_paths = []
labels = []

magnifications = ["40X", "100X", "200X", "400X"]

valid_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)

print("\nSearching dataset...")

for magnification in magnifications:

    mag_path = os.path.join(
        DATASET_DIR,
        magnification
    )

    if not os.path.isdir(mag_path):
        print("WARNING - missing:", mag_path)
        continue

    print("Found:", magnification)

    for class_name in CLASS_NAMES:

        class_path = os.path.join(
            mag_path,
            class_name
        )

        if not os.path.isdir(class_path):
            print(
                "WARNING - missing class:",
                class_path
            )
            continue

        for root, dirs, files in os.walk(class_path):

            for file in files:

                if file.lower().endswith(valid_extensions):

                    full_path = os.path.join(
                        root,
                        file
                    )

                    image_paths.append(full_path)
                    labels.append(
                        CLASS_TO_ID[class_name]
                    )

# ============================================================
# CONVERT TO NUMPY
# ============================================================

image_paths = np.array(image_paths)
labels = np.array(labels)

print("\n" + "=" * 75)
print("DATASET SUMMARY")
print("=" * 75)

print("\nTotal images:", len(image_paths))

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(labels == i)

    print(
        f"{class_name:<22}: {count}"
    )

if len(image_paths) == 0:

    raise RuntimeError(
        "\nNo images were found. Check DATASET_DIR."
    )

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("DATASET SPLIT")
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
# ORIGINAL TRAIN DISTRIBUTION
# ============================================================

print("\nOriginal training distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(train_labels == i)

    print(
        f"{class_name:<22}: {count}"
    )

# ============================================================
# BALANCED TRAINING DATA
# ============================================================

print("\n" + "=" * 75)
print("CREATING BALANCED TRAINING DATA")
print("=" * 75)

balanced_paths = []
balanced_labels = []

class_indices = []

for class_id in range(NUM_CLASSES):

    indices = np.where(
        train_labels == class_id
    )[0]

    class_indices.append(indices)

max_count = max(
    len(indices)
    for indices in class_indices
)

print("\nLargest class size:", max_count)

for class_id, indices in enumerate(class_indices):

    current_paths = train_paths[indices]
    current_labels = train_labels[indices]

    # Oversample minority classes
    if len(current_paths) < max_count:

        sampled_paths, sampled_labels = resample(
            current_paths,
            current_labels,
            replace=True,
            n_samples=max_count,
            random_state=SEED + class_id
        )

    else:

        sampled_paths = current_paths
        sampled_labels = current_labels

    balanced_paths.extend(sampled_paths)
    balanced_labels.extend(sampled_labels)

balanced_paths = np.array(balanced_paths)
balanced_labels = np.array(balanced_labels)

# Shuffle balanced dataset

shuffle_indices = np.random.permutation(
    len(balanced_paths)
)

balanced_paths = balanced_paths[
    shuffle_indices
]

balanced_labels = balanced_labels[
    shuffle_indices
]

print("\nBalanced training distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        balanced_labels == i
    )

    print(
        f"{class_name:<22}: {count}"
    )

print(
    "\nTotal balanced training images:",
    len(balanced_paths)
)

# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation"
)

# ============================================================
# DENSENET PREPROCESSING
# ============================================================

def preprocess_image(image, label):

    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image, label


# ============================================================
# TRAIN DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        balanced_paths,
        balanced_labels
    )
)

train_ds = train_ds.shuffle(
    len(balanced_paths),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.map(
    lambda x, y: (
        augmentation(x, training=True),
        y
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

# ============================================================
# VALIDATION DATASET
# ============================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)

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

test_ds = test_ds.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(
    BATCH_SIZE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)

print("\nDatasets created successfully.")

# ============================================================
# LOAD DENSENET121
# ============================================================

print("\n" + "=" * 75)
print("LOADING LOCAL DENSENET121 WEIGHTS")
print("=" * 75)

print("\nWeights:")
print(WEIGHTS_PATH)

base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights=WEIGHTS_PATH,
    input_shape=(224, 224, 3)
)

print("\nDenseNet121 loaded successfully.")

print(
    "Base model output:",
    base_model.output.shape
)

# ============================================================
# MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image_input"
)

x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(
    0.35
)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(
    0.25
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="subclass_output"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)

# ============================================================
# STAGE 1 - FREEZE DENSENET
# ============================================================

print("\n" + "=" * 75)
print("STAGE 1 - TRAINING CLASSIFICATION HEAD")
print("=" * 75)

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ============================================================
# TRAIN STAGE 1
# ============================================================

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    verbose=1
)

# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 75)
print("STAGE 2 - FINE TUNING DENSENET121")
print("=" * 75)

base_model.trainable = True

# Freeze early layers.
# Fine-tune only the upper portion of DenseNet.

fine_tune_from = 300

for layer in base_model.layers[:fine_tune_from]:

    layer.trainable = False

for layer in base_model.layers[fine_tune_from:]:

    layer.trainable = True

# Keep BatchNormalization layers frozen.
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

trainable_count = sum(
    1 for layer in model.layers
    if layer.trainable
)

print(
    "\nTrainable model layers:",
    trainable_count
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    verbose=1
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING BEST MODEL")
print("=" * 75)

best_model = tf.keras.models.load_model(
    BEST_MODEL
)

print("\nBest model loaded:")
print(BEST_MODEL)

# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST - 8 CLASS CLASSIFICATION")
print("=" * 75)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\n")
print("=" * 75)
print("FINAL 8-CLASS RESULTS")
print("=" * 75)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

best_model.save(
    FINAL_MODEL
)

print("\nFinal model saved:")
print(FINAL_MODEL)

# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_FILE,
    "w"
) as f:

    for class_name in CLASS_NAMES:

        f.write(
            class_name + "\n"
        )

print(
    "\nClass names saved:"
)

print(
    CLASS_NAMES_FILE
)

# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

acc = (
    history1.history["accuracy"]
    + history2.history["accuracy"]
)

val_acc = (
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)

loss = (
    history1.history["loss"]
    + history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    + history2.history["val_loss"]
)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    acc,
    label="Training Accuracy"
)

plt.plot(
    val_acc,
    label="Validation Accuracy"
)

plt.axvline(
    x=HEAD_EPOCHS - 1,
    linestyle="--",
    label="Fine-tuning starts"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "DenseNet121 8-Class Training Accuracy"
)

plt.legend()

plt.grid(True)

accuracy_graph = os.path.join(
    RESULTS_DIR,
    "subclass_training_accuracy_v2.png"
)

plt.savefig(
    accuracy_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.axvline(
    x=HEAD_EPOCHS - 1,
    linestyle="--",
    label="Fine-tuning starts"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "DenseNet121 8-Class Training Loss"
)

plt.legend()

plt.grid(True)

loss_graph = os.path.join(
    RESULTS_DIR,
    "subclass_training_loss_v2.png"
)

plt.savefig(
    loss_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

INFO_FILE = os.path.join(
    RESULTS_DIR,
    "subclass_training_results_v2.txt"
)

with open(
    INFO_FILE,
    "w"
) as f:

    f.write(
        "DENSENET121 8-CLASS BREAST CANCER SUBCLASSIFICATION\n"
    )

    f.write(
        "====================================================\n\n"
    )

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
        f"Test loss: {test_loss:.6f}\n"
    )

    f.write(
        f"Test accuracy: {test_accuracy:.6f}\n"
    )

    f.write(
        f"Test accuracy percentage: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classes:\n"
    )

    for name in CLASS_NAMES:

        f.write(
            f"- {name}\n"
        )

# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 75)
print("8-CLASS TRAINING COMPLETED")
print("=" * 75)

print("\nBest model:")
print(BEST_MODEL)

print("\nFinal model:")
print(FINAL_MODEL)

print("\nTraining accuracy graph:")
print(accuracy_graph)

print("\nTraining loss graph:")
print(loss_graph)

print("\nTraining information:")
print(INFO_FILE)

print("\n")
print("=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    "\nDo NOT evaluate the old model."
)

print(
    "The new model is:"
)

print(
    "models/densenet121_subclass_best_v2.keras"
)

print(
    "\nNext step: detailed evaluation with confusion matrix,"
)

print(
    "precision, recall, F1-score and per-class results."
)

print("=" * 75)