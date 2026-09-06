import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Fine-tuning settings
EPOCHS = 15
LEARNING_RATE = 1e-5

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

STAGE1_MODEL = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_best_v2.keras"

OUTPUT_MODEL = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\densenet121_subclass_finetuned_best.keras"

OUTPUT_INFO = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models\finetuning_info.txt"

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
# PRINT HEADER
# ============================================================

print("=" * 75)
print("DENSENET121 - STAGE 2 FINE-TUNING")
print("=" * 75)

print()
print("Stage 1 model:")
print(STAGE1_MODEL)

print()
print("Output model:")
print(OUTPUT_MODEL)

# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(STAGE1_MODEL):
    raise FileNotFoundError(
        "\nStage 1 model was not found:\n" + STAGE1_MODEL
    )

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        "\nDataset was not found:\n" + DATASET_DIR
    )

# ============================================================
# SEARCH DATASET
# ============================================================

print()
print("=" * 75)
print("SEARCHING DATASET")
print("=" * 75)

magnifications = ["40X", "100X", "200X", "400X"]

image_paths = []
image_labels = []

valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")

for magnification in magnifications:

    mag_dir = os.path.join(DATASET_DIR, magnification)

    if not os.path.isdir(mag_dir):
        print("WARNING - missing:", mag_dir)
        continue

    print("Found:", magnification)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = os.path.join(mag_dir, class_name)

        if not os.path.isdir(class_dir):
            print("WARNING - missing class:", class_dir)
            continue

        for root, dirs, files in os.walk(class_dir):

            for filename in files:

                if filename.lower().endswith(valid_extensions):

                    full_path = os.path.join(root, filename)

                    image_paths.append(full_path)
                    image_labels.append(class_index)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

image_paths = np.array(image_paths)
image_labels = np.array(image_labels)

print()
print("Total images:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No images were found.")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("Class distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(image_labels == i)

    print(f"{class_name:<22}: {count}")

# ============================================================
# SAME 70 / 15 / 15 SPLIT
# ============================================================

print()
print("=" * 75)
print("CREATING DATASET SPLIT")
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
# IMAGE LOADING FUNCTION
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

    image = tf.cast(image, tf.float32)

    # DenseNet preprocessing
    image = tf.keras.applications.densenet.preprocess_input(
        image
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
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            0.05,
            0.05
        )
    ],
    name="fine_tuning_augmentation"
)

# ============================================================
# CREATE DATASETS
# ============================================================

def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            lambda x, y: (
                augmentation(x, training=True),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = create_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_dataset(
    val_paths,
    val_labels,
    training=False
)

test_ds = create_dataset(
    test_paths,
    test_labels,
    training=False
)

print()
print("Datasets created successfully.")

# ============================================================
# CLASS WEIGHTS
# ============================================================

print()
print("=" * 75)
print("CALCULATING CLASS WEIGHTS")
print("=" * 75)

classes = np.unique(train_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<22}: "
        f"{class_weights[i]:.4f}"
    )

# ============================================================
# LOAD STAGE 1 MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING STAGE 1 MODEL")
print("=" * 75)

print()
print(STAGE1_MODEL)

model = tf.keras.models.load_model(
    STAGE1_MODEL
)

print()
print("Model loaded successfully!")

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# FIND DENSENET121 BASE MODEL
# ============================================================

print()
print("=" * 75)
print("FINDING DENSENET121")
print("=" * 75)

base_model = None

# Search for the nested DenseNet model by layer name.
for layer in model.layers:

    print(
        "Layer:",
        layer.name,
        "Type:",
        type(layer).__name__
    )

    if layer.name.lower() == "densenet121":
        base_model = layer
        break


# If not found at top level, search recursively
if base_model is None:

    for layer in model.layers:

        if hasattr(layer, "layers"):

            for sub_layer in layer.layers:

                if sub_layer.name.lower() == "densenet121":

                    base_model = sub_layer
                    break

        if base_model is not None:
            break


if base_model is None:

    print()
    print("ERROR: DenseNet121 base model was not found.")
    print()
    print("Available model layers:")

    for layer in model.layers:

        print(
            layer.name,
            "->",
            type(layer).__name__
        )

    raise RuntimeError(
        "Could not find DenseNet121 inside the trained model."
    )


print()
print("DenseNet121 found successfully!")
print("Name:", base_model.name)
print("Output:", base_model.output_shape)

# ============================================================
# FREEZE EVERYTHING FIRST
# ============================================================

for layer in base_model.layers:
    layer.trainable = False

# ============================================================
# UNFREEZE LAST DENSENET LAYERS
# ============================================================

print()
print("=" * 75)
print("UNFREEZING LAST DENSENET LAYERS")
print("=" * 75)

# Keep most of DenseNet frozen.
# Fine-tune only the final convolutional block.

trainable_count = 0

for layer in base_model.layers:

    if layer.name.startswith("conv5_"):

        layer.trainable = True
        trainable_count += 1

print()
print("Trainable DenseNet layers:", trainable_count)

# ============================================================
# KEEP BATCH NORMALIZATION FROZEN
# ============================================================

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False

# ============================================================
# COMPILE MODEL
# ============================================================

print()
print("=" * 75)
print("COMPILING FINE-TUNING MODEL")
print("=" * 75)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

print()
print("Learning rate:", LEARNING_RATE)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    OUTPUT_MODEL,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
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
# STAGE 2 TRAINING
# ============================================================

print()
print("=" * 75)
print("STAGE 2 - FINE-TUNING DENSENET121")
print("=" * 75)

print()
print("Starting fine-tuning...")
print()
print("IMPORTANT:")
print("Your Stage 1 model will NOT be overwritten.")
print()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# ============================================================
# LOAD BEST FINE-TUNED MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING BEST FINE-TUNED MODEL")
print("=" * 75)

if os.path.exists(OUTPUT_MODEL):

    model = tf.keras.models.load_model(
        OUTPUT_MODEL
    )

    print()
    print("Best fine-tuned model loaded.")

else:

    print()
    print("WARNING: checkpoint was not created.")
    print("Using current model.")

# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 75)
print("FINAL TEST - FINE-TUNED MODEL")
print("=" * 75)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print()
print("=" * 75)
print("FINE-TUNED TEST RESULTS")
print("=" * 75)

print()
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ============================================================
# SAVE INFORMATION
# ============================================================

with open(
    OUTPUT_INFO,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "DENSENET121 8-CLASS FINE-TUNING RESULTS\n"
    )

    f.write(
        "=========================================\n\n"
    )

    f.write(
        f"Stage 1 model:\n{STAGE1_MODEL}\n\n"
    )

    f.write(
        f"Fine-tuned model:\n{OUTPUT_MODEL}\n\n"
    )

    f.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    f.write(
        f"Test Accuracy: {test_accuracy:.6f}\n"
    )

    f.write(
        f"Test Accuracy (%): "
        f"{test_accuracy * 100:.2f}\n"
    )

    f.write(
        f"\nLearning Rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        "\nClasses:\n"
    )

    for class_name in CLASS_NAMES:

        f.write(
            f"- {class_name}\n"
        )

# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 75)
print("FINE-TUNING COMPLETED")
print("=" * 75)

print()
print("Original Stage 1 model:")
print(STAGE1_MODEL)

print()
print("New fine-tuned model:")
print(OUTPUT_MODEL)

print()
print("Fine-tuning information:")
print(OUTPUT_INFO)

print()
print("=" * 75)
print("IMPORTANT")
print("=" * 75)

print()
print("Your original 80.79% Stage 1 model is SAFE.")
print()
print("Do NOT delete:")
print("densenet121_subclass_best_v2.keras")
print()
print("The new model is:")
print("densenet121_subclass_finetuned_best.keras")
print()
print("=" * 75)