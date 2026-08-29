import os
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse"

WEIGHTS_PATH = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\weights\densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"

MODEL_DIR = r"C:\Users\16ras\OneDrive\Documents\Desktop\BreastCancerProject\models"

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

ALL_CLASSES = BENIGN_CLASSES + MALIGNANT_CLASSES

# ============================================================
# FIND ALL IMAGES
# ============================================================

print("=" * 70)
print("FINAL HIERARCHICAL BREAST CANCER CLASSIFICATION")
print("=" * 70)

print("\nSearching dataset...")

image_paths = []
labels = []

for magnification in ["40X", "100X", "200X", "400X"]:

    mag_path = os.path.join(DATASET_ROOT, magnification)

    if not os.path.isdir(mag_path):
        continue

    for class_name in ALL_CLASSES:

        class_path = os.path.join(mag_path, class_name)

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

                    labels.append(class_name)


print("\nTotal images:", len(image_paths))

# ============================================================
# VERIFY DATASET
# ============================================================

print("\nClass distribution:")

for class_name in ALL_CLASSES:

    count = sum(1 for x in labels if x == class_name)

    print(f"{class_name:22s}: {count}")

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

    image = tf.cast(image, tf.float32)

    # DenseNet preprocessing
    image = tf.keras.applications.densenet.preprocess_input(
        image
    )

    return image, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="augmentation")


def augment_image(image, label):

    image = augmentation(
        image,
        training=True
    )

    return image, label


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

paths_train, paths_temp, labels_train, labels_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

paths_val, paths_test, labels_val, labels_test = train_test_split(
    paths_temp,
    labels_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=labels_temp
)

print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print("Training   :", len(paths_train))
print("Validation :", len(paths_val))
print("Testing    :", len(paths_test))


# ============================================================
# CREATE MODEL FUNCTION
# ============================================================

def create_model(num_classes):

    print("\nLoading DenseNet121...")

    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=WEIGHTS_PATH,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    return model, base_model


# ============================================================
# DATASET CREATION FUNCTION
# ============================================================

def create_dataset(paths, labels, class_names, training=False):

    class_to_id = {
        name: i
        for i, name in enumerate(class_names)
    }

    numeric_labels = [
        class_to_id[x]
        for x in labels
    ]

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, numeric_labels)
    )

    if training:

        ds = ds.shuffle(
            len(paths),
            seed=SEED
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        ds = ds.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


# ============================================================
# TRAIN ONE FOUR-CLASS MODEL
# ============================================================

def train_four_class_model(
    name,
    class_names,
    train_paths,
    train_labels,
    val_paths,
    val_labels
):

    print("\n")
    print("=" * 70)
    print(f"TRAINING {name.upper()} MODEL")
    print("=" * 70)

    # Keep only required classes
    train_indices = [
        i for i, x in enumerate(train_labels)
        if x in class_names
    ]

    val_indices = [
        i for i, x in enumerate(val_labels)
        if x in class_names
    ]

    selected_train_paths = [
        train_paths[i]
        for i in train_indices
    ]

    selected_train_labels = [
        train_labels[i]
        for i in train_indices
    ]

    selected_val_paths = [
        val_paths[i]
        for i in val_indices
    ]

    selected_val_labels = [
        val_labels[i]
        for i in val_indices
    ]

    print("\nTraining images:", len(selected_train_paths))
    print("Validation images:", len(selected_val_paths))

    print("\nClass distribution:")

    for class_name in class_names:

        count = sum(
            1
            for x in selected_train_labels
            if x == class_name
        )

        print(
            f"{class_name:22s}: {count}"
        )

    # Create datasets

    train_ds = create_dataset(
        selected_train_paths,
        selected_train_labels,
        class_names,
        training=True
    )

    val_ds = create_dataset(
        selected_val_paths,
        selected_val_labels,
        class_names,
        training=False
    )

    # Create model

    model, base_model = create_model(
        len(class_names)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"densenet121_{name}_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nStage 1 training...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # STAGE 2 FINE-TUNING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(f"FINE-TUNING {name.upper()} MODEL")
    print("=" * 70)

    # Load best stage-1 model
    model = tf.keras.models.load_model(
        model_path
    )

    # Find DenseNet base
    base_model = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ) and "densenet" in layer.name.lower():

            base_model = layer
            break

    if base_model is None:

        print(
            "DenseNet base model not found."
        )

        return model_path

    # Freeze everything first
    base_model.trainable = True

    # Freeze early layers
    for layer in base_model.layers[:-30]:

        layer.trainable = False

    # Keep BatchNorm frozen
    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    print(
        "Trainable DenseNet layers:",
        sum(
            1
            for layer in base_model.layers
            if layer.trainable
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_path = os.path.join(
        MODEL_DIR,
        f"densenet121_{name}_finetuned_best.keras"
    )

    callbacks_ft = [

        tf.keras.callbacks.ModelCheckpoint(
            fine_tune_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=6,
        callbacks=callbacks_ft
    )

    return fine_tune_path


# ============================================================
# TRAIN BENIGN MODEL
# ============================================================

benign_model_path = train_four_class_model(
    "benign_subclass",
    BENIGN_CLASSES,
    paths_train,
    labels_train,
    paths_val,
    labels_val
)


# ============================================================
# TRAIN MALIGNANT MODEL
# ============================================================

malignant_model_path = train_four_class_model(
    "malignant_subclass",
    MALIGNANT_CLASSES,
    paths_train,
    labels_train,
    paths_val,
    labels_val
)


# ============================================================
# SAVE INFORMATION
# ============================================================

info_path = os.path.join(
    MODEL_DIR,
    "hierarchical_model_info.txt"
)

with open(
    info_path,
    "w"
) as f:

    f.write(
        "HIERARCHICAL BREAST CANCER CLASSIFICATION\n\n"
    )

    f.write(
        "Benign classes:\n"
    )

    for x in BENIGN_CLASSES:
        f.write(x + "\n")

    f.write(
        "\nMalignant classes:\n"
    )

    for x in MALIGNANT_CLASSES:
        f.write(x + "\n")

    f.write(
        "\nBenign model:\n"
    )

    f.write(
        benign_model_path + "\n"
    )

    f.write(
        "\nMalignant model:\n"
    )

    f.write(
        malignant_model_path + "\n"
    )
print("\n")
print("=" * 70)
print("HIERARCHICAL TRAINING COMPLETED")
print("=" * 70)

print("\nBenign model:")
print(benign_model_path)

print("\nMalignant model:")
print(malignant_model_path)

print("\nInformation:")
print(info_path)