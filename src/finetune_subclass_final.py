import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

# ==============================================================
# SETTINGS
# ==============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

BASE_MODEL_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_best_v2.keras"

OUTPUT_MODEL = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_finetuned_v2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

EPOCHS = 8

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


# ==============================================================
# HEADER
# ==============================================================

print("=" * 75)
print("CONTROLLED FINE-TUNING - 8 CLASS DENSENET121")
print("=" * 75)

print()
print("Base model:")
print(BASE_MODEL_PATH)

print()

if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(
        "Base model not found:\n" + BASE_MODEL_PATH
    )


# ==============================================================
# FIND DATASET
# ==============================================================

print("=" * 75)
print("SEARCHING DATASET")
print("=" * 75)

image_paths = []
labels = []

magnifications = ["40X", "100X", "200X", "400X"]

for magnification in magnifications:

    mag_path = os.path.join(
        DATASET_PATH,
        magnification
    )

    if not os.path.isdir(mag_path):
        print("Missing:", mag_path)
        continue

    print("Found:", magnification)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            mag_path,
            class_name
        )

        if not os.path.isdir(class_path):
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


image_paths = np.array(image_paths)
labels = np.array(labels)

print()
print("Total images:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No images found.")


# ==============================================================
# DATA DISTRIBUTION
# ==============================================================

print()
print("=" * 75)
print("DATASET DISTRIBUTION")
print("=" * 75)

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(labels == i)

    print(
        f"{class_name:<22}: {count}"
    )


# ==============================================================
# SAME 70/15/15 STYLE SPLIT
# ==============================================================
#
# IMPORTANT:
# We use stratification so every class appears in
# train / validation / test.
#
# =============================================================

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

print()
print("=" * 75)
print("DATASET SPLIT")
print("=" * 75)

print("Training   :", len(train_paths))
print("Validation :", len(val_paths))
print("Testing    :", len(test_paths))


# ==============================================================
# DATA LOADING FUNCTION
# ==============================================================

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

    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image, label


# ==============================================================
# AUGMENTATION
# ==============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.05
        ),

        tf.keras.layers.RandomContrast(
            0.05
        )
    ],
    name="safe_augmentation"
)


def augment_image(image, label):

    image = augmentation(
        image,
        training=True
    )

    return image, label


# ==============================================================
# DATASET CREATION
# ==============================================================

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


print()
print("Datasets created successfully.")


# ==============================================================
# LOAD BEST 80.79% MODEL
# ==============================================================

print()
print("=" * 75)
print("LOADING CURRENT BEST MODEL")
print("=" * 75)

model = tf.keras.models.load_model(
    BASE_MODEL_PATH
)

print("Model loaded successfully.")

print(
    "Input shape :",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)


# ==============================================================
# FREEZE EVERYTHING FIRST
# ==============================================================

for layer in model.layers:

    layer.trainable = False


# ==============================================================
# FIND DENSENET BASE
# ==============================================================

densenet_base = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        if "densenet" in layer.name.lower():

            densenet_base = layer
            break


if densenet_base is None:

    print()
    print("DenseNet nested model not found.")
    print("Using top-level model layers.")

else:

    print()
    print(
        "DenseNet base found:",
        densenet_base.name
    )

    # ----------------------------------------------------------
    # Unfreeze ONLY the last small portion of DenseNet
    # ----------------------------------------------------------

    for layer in densenet_base.layers:

        layer.trainable = False


    trainable_count = 0

    for layer in reversed(
        densenet_base.layers
    ):

        if trainable_count >= 20:
            break

        # Keep BatchNorm frozen
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

        else:
            layer.trainable = True
            trainable_count += 1


# ==============================================================
# COUNT TRAINABLE PARAMETERS
# ==============================================================

trainable_params = np.sum(
    [
        np.prod(v.shape)
        for v in model.trainable_variables
    ]
)

print()
print(
    "Trainable parameters:",
    trainable_params
)


# ==============================================================
# COMPILE
# ==============================================================

print()
print("=" * 75)
print("COMPILING MODEL")
print("=" * 75)

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-6
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ==============================================================
# CALLBACKS
# ==============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    OUTPUT_MODEL,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=2,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=1,

    min_lr=1e-8,

    verbose=1
)


# ==============================================================
# TRAIN
# ==============================================================

print()
print("=" * 75)
print("CONTROLLED FINE-TUNING")
print("=" * 75)

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],

    verbose=1
)


# ==============================================================
# LOAD BEST FINE-TUNED MODEL
# ==============================================================

print()
print("=" * 75)
print("LOADING BEST FINE-TUNED MODEL")
print("=" * 75)

best_model = tf.keras.models.load_model(
    OUTPUT_MODEL
)


# ==============================================================
# VALIDATION RESULT
# ==============================================================

print()
print("=" * 75)
print("FINAL VALIDATION RESULT")
print("=" * 75)

val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print()
print(
    f"Validation accuracy: {val_accuracy:.4f}"
)

print(
    f"Validation accuracy: {val_accuracy * 100:.2f}%"
)


# ==============================================================
# SAFETY CHECK
# ==============================================================

print()
print("=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print()
print("Previous best test accuracy:")
print("80.79%")

print()
print("New validation accuracy:")
print(
    f"{val_accuracy * 100:.2f}%"
)

print()
print("New model saved at:")
print(OUTPUT_MODEL)

print()
print("=" * 75)
print("FINE-TUNING EXPERIMENT COMPLETED")
print("=" * 75)