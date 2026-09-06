import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

BASE_MODEL_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_best_v2.keras"

OUTPUT_MODEL_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_finetuned_v2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

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

# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("DENSENET121 - CONTROLLED FINE-TUNING V2")
print("=" * 75)

print("\nStarting from existing 80.79% model:")
print(BASE_MODEL_PATH)

if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(
        "\nERROR: Existing Stage 1 model was not found:\n"
        + BASE_MODEL_PATH
    )

print("\nExisting model found successfully.")

# ================================================================
# STEP 1 - COLLECT ALL IMAGE PATHS
# ================================================================

print("\n" + "=" * 75)
print("LOADING DATASET")
print("=" * 75)

image_paths = []
labels = []

for magnification in ["40X", "100X", "200X", "400X"]:

    magnification_path = os.path.join(
        DATASET_DIR,
        magnification
    )

    if not os.path.isdir(magnification_path):
        print("WARNING - missing:", magnification_path)
        continue

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            magnification_path,
            class_name
        )

        if not os.path.isdir(class_path):
            print("WARNING - missing:", class_path)
            continue

        for root, dirs, files in os.walk(class_path):

            for filename in files:

                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):

                    image_paths.append(
                        os.path.join(root, filename)
                    )

                    labels.append(class_index)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("\nTotal images:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No images found. Check DATASET_DIR.")

print("\nOriginal class distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(labels == i)

    print(
        f"{class_name:22s}: {count}"
    )

# ================================================================
# STEP 2 - STRATIFIED SPLIT
# ================================================================

print("\n" + "=" * 75)
print("CREATING STRATIFIED DATA SPLIT")
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

# ================================================================
# STEP 3 - CLASS WEIGHTS
# ================================================================

print("\n" + "=" * 75)
print("CALCULATING MODERATE CLASS WEIGHTS")
print("=" * 75)

classes = np.arange(NUM_CLASSES)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

class_weights = {
    i: float(weights[i])
    for i in range(NUM_CLASSES)
}

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:22s}: {class_weights[i]:.4f}"
    )

# ================================================================
# STEP 4 - IMAGE LOADING FUNCTION
# ================================================================

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

    return image, label


# ================================================================
# STEP 5 - TRAINING AUGMENTATION
# ================================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="training_augmentation"
)


def augment_image(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    return image, label


# ================================================================
# STEP 6 - CREATE DATASETS
# ================================================================

print("\n" + "=" * 75)
print("CREATING TF.DATA DATASETS")
print("=" * 75)

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

train_ds = (
    train_ds
    .shuffle(
        len(train_paths),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Datasets created successfully.")

# ================================================================
# STEP 7 - LOAD EXISTING BEST MODEL
# ================================================================

print("\n" + "=" * 75)
print("LOADING EXISTING 80.79% MODEL")
print("=" * 75)

model = tf.keras.models.load_model(
    BASE_MODEL_PATH
)

print("\nModel loaded successfully.")

print(
    "Input shape :",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)

# ================================================================
# STEP 8 - FIND DENSENET BASE MODEL
# ================================================================

print("\n" + "=" * 75)
print("FINDING DENSENET121 BASE MODEL")
print("=" * 75)

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        if (
            "densenet" in layer.name.lower()
            or "densenet121" in layer.name.lower()
        ):

            base_model = layer
            break

if base_model is None:

    print(
        "\nCould not find nested DenseNet model."
    )

    print(
        "\nTop-level layers:"
    )

    for layer in model.layers:
        print(
            layer.name,
            "->",
            type(layer)
        )

    raise RuntimeError(
        "\nDenseNet base model could not be identified."
    )

print(
    "\nDenseNet base model found:",
    base_model.name
)

print(
    "Total DenseNet layers:",
    len(base_model.layers)
)

# ================================================================
# STEP 9 - FREEZE MOST OF DENSENET
# ================================================================

print("\n" + "=" * 75)
print("CONTROLLED FINE-TUNING")
print("=" * 75)

# First freeze everything

base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

# ------------------------------------------------
# Unfreeze only the final convolutional section.
# This is deliberately conservative.
# ------------------------------------------------

UNFREEZE_LAST = 30

for layer in base_model.layers[-UNFREEZE_LAST:]:

    # Keep BatchNormalization frozen.
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

    else:
        layer.trainable = True

trainable_count = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)

print(
    "\nTrainable DenseNet layers:",
    trainable_count
)

# ================================================================
# STEP 10 - COMPILE
# ================================================================

print("\n" + "=" * 75)
print("COMPILING MODEL")
print("=" * 75)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

print("Model compiled successfully.")

# ================================================================
# STEP 11 - CALLBACKS
# ================================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    OUTPUT_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=4,
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

# ================================================================
# STEP 12 - FINE-TUNING
# ================================================================

print("\n" + "=" * 75)
print("STARTING CONTROLLED FINE-TUNING")
print("=" * 75)

print("\nIMPORTANT:")
print("The original 80.79% model will NOT be overwritten.")
print(
    "New model:",
    OUTPUT_MODEL_PATH
)

history = model.fit(
    train_ds,
    validation_data=val_ds,

    epochs=10,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

# ================================================================
# STEP 13 - LOAD BEST FINE-TUNED MODEL
# ================================================================

print("\n" + "=" * 75)
print("LOADING BEST FINE-TUNED MODEL")
print("=" * 75)

best_model = tf.keras.models.load_model(
    OUTPUT_MODEL_PATH
)

print(
    "\nBest fine-tuned model loaded successfully."
)

# ================================================================
# STEP 14 - FINAL TEST
# ================================================================

print("\n" + "=" * 75)
print("FINAL TEST - FINE-TUNED V2")
print("=" * 75)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\n" + "=" * 75)
print("FINAL FINE-TUNED RESULTS")
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

# ================================================================
# STEP 15 - SAVE INFORMATION
# ================================================================

info_path = (
    r"C:\Users\16ras\OneDrive\Documents\Desktop"
    r"\BreastCancerProject\models"
    r"\finetuning_v2_info.txt"
)

with open(
    info_path,
    "w"
) as f:

    f.write(
        "DENSENET121 CONTROLLED FINE-TUNING V2\n"
    )

    f.write(
        "=====================================\n\n"
    )

    f.write(
        "Baseline model:\n"
    )

    f.write(
        BASE_MODEL_PATH + "\n\n"
    )

    f.write(
        "Fine-tuned model:\n"
    )

    f.write(
        OUTPUT_MODEL_PATH + "\n\n"
    )

    f.write(
        "Test loss: "
        + str(test_loss)
        + "\n"
    )

    f.write(
        "Test accuracy: "
        + str(test_accuracy)
        + "\n"
    )

    f.write(
        "Test accuracy percentage: "
        + str(test_accuracy * 100)
        + "%\n"
    )

    f.write(
        "\nUnfrozen final DenseNet layers: "
        + str(UNFREEZE_LAST)
        + "\n"
    )

print(
    "\nInformation saved:"
)

print(info_path)

# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 75)
print("FINE-TUNING V2 COMPLETED")
print("=" * 75)

print(
    "\nIMPORTANT:"
)

print(
    "Original 80.79% model remains SAFE:"
)

print(
    BASE_MODEL_PATH
)

print(
    "\nNew experimental model:"
)

print(
    OUTPUT_MODEL_PATH
)

print("\nDo NOT delete either model.")