import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria"

WEIGHTS_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\weights\densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 8

SEED = 42

MODEL_PATH = "densenet121_binary_best.keras"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 75)
print("DENSENET121 - IMPROVED BINARY BREAST CANCER CLASSIFICATION")
print("=" * 75)


# ============================================================
# CHECK WEIGHTS
# ============================================================

if not os.path.exists(WEIGHTS_PATH):

    raise FileNotFoundError(
        "\nDenseNet121 weights not found!\n\n"
        "Expected:\n"
        + WEIGHTS_PATH
    )

print("\nDenseNet weights found:")
print(WEIGHTS_PATH)


# ============================================================
# COLLECT DATA
# ============================================================

image_paths = []
labels = []

magnifications = [
    "40X",
    "100X",
    "200X",
    "400X"
]

for magnification in magnifications:

    magnification_path = os.path.join(
        DATASET_PATH,
        magnification
    )

    for class_name, label in [
        ("benign", 0),
        ("malignant", 1)
    ]:

        class_path = os.path.join(
            magnification_path,
            class_name
        )

        if not os.path.exists(class_path):

            print(
                "WARNING - missing:",
                class_path
            )

            continue

        for filename in os.listdir(class_path):

            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):

                image_paths.append(
                    os.path.join(
                        class_path,
                        filename
                    )
                )

                labels.append(label)


print("\nDataset statistics")
print("-" * 50)

print(
    "Total images:",
    len(image_paths)
)

print(
    "Benign:",
    labels.count(0)
)

print(
    "Malignant:",
    labels.count(1)
)


# ============================================================
# SHUFFLE
# ============================================================

data = list(
    zip(
        image_paths,
        labels
    )
)

random.shuffle(data)

image_paths, labels = zip(*data)

image_paths = list(image_paths)
labels = list(labels)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

total = len(image_paths)

train_end = int(total * 0.70)

val_end = int(total * 0.85)


train_paths = image_paths[:train_end]
train_labels = labels[:train_end]

val_paths = image_paths[
    train_end:val_end
]

val_labels = labels[
    train_end:val_end
]

test_paths = image_paths[
    val_end:
]

test_labels = labels[
    val_end:
]


print("\nDataset split")
print("-" * 50)

print(
    "Training:",
    len(train_paths)
)

print(
    "Validation:",
    len(val_paths)
)

print(
    "Testing:",
    len(test_paths)
)


# ============================================================
# IMAGE LOADER
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

    image = preprocess_input(
        image
    )

    return (
        image,
        tf.cast(
            label,
            tf.float32
        )
    )


# ============================================================
# DATASETS
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)


train_ds = (
    train_ds
    .shuffle(1000)
    .map(
        load_image,
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


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.10
    ),

])


# ============================================================
# DENSENET121
# ============================================================

print("\nLoading DenseNet121...")

base_model = DenseNet121(

    include_top=False,

    weights=WEIGHTS_PATH,

    input_shape=(
        224,
        224,
        3
    )

)


# ============================================================
# FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# CLASSIFICATION MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)


x = augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(
    0.4
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# STAGE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]

)


print("\n")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 1 - TRANSFER LEARNING")
print("=" * 75)


history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=HEAD_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping
    ]

)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 2 - DENSENET121 FINE TUNING")
print("=" * 75)


# Unfreeze DenseNet

base_model.trainable = True


# Freeze earlier layers
# Fine-tune only the last ~40 layers

for layer in base_model.layers[:-40]:

    layer.trainable = False


# Keep BatchNormalization layers frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COMPILE AGAIN WITH VERY SMALL LR
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]

)


# ============================================================
# FINE TUNING
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping
    ]

)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TEST")
print("=" * 75)


results = model.evaluate(
    test_ds,
    verbose=1
)


print("\n")
print("Test Loss     :", results[0])
print("Test Accuracy :", results[1])
print("Test AUC      :", results[2])


print("\nBest model saved at:")

print(
    os.path.abspath(
        MODEL_PATH
    )
)


print("\n")
print("=" * 75)
print("DENSENET121 TRAINING COMPLETED")
print("=" * 75)