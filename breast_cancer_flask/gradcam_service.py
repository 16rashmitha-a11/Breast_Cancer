import os
import uuid

import numpy as np
import tensorflow as tf

from PIL import Image

import matplotlib
matplotlib.use("Agg")

from matplotlib import colormaps


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "uploads",
    "gradcam",
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True,
)


# ============================================================
# DENSENET121 TARGET LAYER
# ============================================================

TARGET_LAYER_NAME = "conv5_block16_concat"


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    image_array = (
        tf.keras.applications.densenet.preprocess_input(
            image_array
        )
    )

    return image_array


# ============================================================
# FIND DENSENET BASE MODEL
# ============================================================

def get_densenet_base(model):

    # First: normal expected name.
    try:
        base_model = model.get_layer(
            "densenet121"
        )

        print(
            "DenseNet121 base model:",
            base_model.name,
        )

        return base_model

    except Exception:
        pass

    # Backup: find a nested model containing DenseNet.
    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model,
        ):

            if "densenet" in layer.name.lower():

                print(
                    "DenseNet base model found:",
                    layer.name,
                )

                return layer

    # Backup: look through all layers by name.
    for layer in model.layers:

        if "densenet" in layer.name.lower():

            print(
                "DenseNet-like layer found:",
                layer.name,
            )

            return layer

    raise ValueError(
        "DenseNet121 base model could not be found."
    )


# ============================================================
# FIND TARGET CONVOLUTIONAL LAYER
# ============================================================

def get_target_layer(base_model):

    try:
        target_layer = base_model.get_layer(
            TARGET_LAYER_NAME
        )

        print(
            "Grad-CAM target layer:",
            target_layer.name,
        )

        print(
            "Target output:",
            target_layer.output.shape,
        )

        return target_layer

    except Exception as error:

        print(
            "Preferred Grad-CAM layer not found:",
            repr(error),
        )

    # Backup: use the last 4-D layer.
    for layer in reversed(
        base_model.layers
    ):

        try:

            output_shape = layer.output.shape

            if (
                len(output_shape) == 4
                and output_shape[-1] is not None
            ):

                print(
                    "Automatic Grad-CAM layer:",
                    layer.name,
                )

                print(
                    "Target output:",
                    layer.output.shape,
                )

                return layer

        except Exception:
            continue

    raise ValueError(
        "No suitable convolutional layer was found for Grad-CAM."
    )


# ============================================================
# APPLY OUTER MODEL LAYERS
# ============================================================

def apply_layers_after_base(
    model,
    base_model,
    tensor,
):

    base_index = None

    for index, layer in enumerate(
        model.layers
    ):

        if layer is base_model:
            base_index = index
            break

        if layer.name == base_model.name:
            base_index = index
            break

    if base_index is None:

        raise ValueError(
            "Could not find DenseNet base model inside outer model."
        )

    following_layers = model.layers[
        base_index + 1:
    ]

    x = tensor

    for layer in following_layers:

        # Skip InputLayer safely.
        if isinstance(
            layer,
            tf.keras.layers.InputLayer,
        ):
            continue

        try:

            x = layer(
                x,
                training=False,
            )

        except TypeError:

            x = layer(x)

    return x


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def make_gradcam(
    model,
    image_path,
    predicted_class_index,
):

    print("\n" + "=" * 70)
    print("GENERATING GRAD-CAM")
    print("IMAGE:", image_path)
    print("CLASS INDEX:", predicted_class_index)
    print("=" * 70)

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image_array = preprocess_image(
        image_path
    )

    base_model = get_densenet_base(
        model
    )

    target_layer = get_target_layer(
        base_model
    )

    # --------------------------------------------------------
    # FEATURE MODEL
    #
    # This model starts at DenseNet's real input and ends at
    # the selected convolutional feature map.
    # --------------------------------------------------------

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=target_layer.output,
    )

    # --------------------------------------------------------
    # GRADIENT CALCULATION
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs = feature_model(
            image_array,
            training=False,
        )

        tape.watch(
            conv_outputs
        )

        # Continue through the layers after DenseNet so that
        # the gradient is calculated against the SAME subclass
        # prediction produced by the actual model.
        predictions = apply_layers_after_base(
            model,
            base_model,
            conv_outputs,
        )

        predictions = tf.convert_to_tensor(
            predictions
        )

        if (
            len(predictions.shape) != 2
            or predictions.shape[-1] is None
        ):

            raise ValueError(
                "Unexpected model output shape for Grad-CAM: "
                f"{predictions.shape}"
            )

        class_index = int(
            predicted_class_index
        )

        if class_index < 0 or class_index >= int(
            predictions.shape[-1]
        ):

            raise ValueError(
                f"Invalid Grad-CAM class index: {class_index}"
            )

        class_score = predictions[
            :,
            class_index,
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs,
    )

    if gradients is None:

        raise ValueError(
            "Gradients are None. Grad-CAM cannot be generated."
        )

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING OF GRADIENTS
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # --------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1,
    )

    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    max_value = float(
        tf.reduce_max(
            heatmap
        ).numpy()
    )

    if max_value <= 1e-10:

        raise ValueError(
            "Grad-CAM heatmap contains no positive activation."
        )

    heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_size = (
        original_image.width,
        original_image.height,
    )

    # --------------------------------------------------------
    # RESIZE HEATMAP
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original_size,
        Image.Resampling.BILINEAR,
    )

    heatmap = (
        np.asarray(
            heatmap_image,
            dtype=np.float32,
        )
        / 255.0
    )

    # --------------------------------------------------------
    # COLOR MAP
    # --------------------------------------------------------

    jet = colormaps["jet"]

    colored_heatmap = jet(
        heatmap
    )

    colored_heatmap = (
        colored_heatmap[:, :, :3] * 255
    ).astype(
        np.uint8
    )

    heatmap_rgb = Image.fromarray(
        colored_heatmap
    ).convert("RGB")

    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    original_array = np.asarray(
        original_image,
        dtype=np.float32,
    )

    heatmap_array = np.asarray(
        heatmap_rgb,
        dtype=np.float32,
    )

    alpha = (
        0.15
        + (heatmap ** 0.8) * 0.60
    )

    alpha = np.expand_dims(
        alpha,
        axis=-1,
    )

    overlay_array = (
        original_array * (1.0 - alpha)
        + heatmap_array * alpha
    )

    overlay_array = np.clip(
        overlay_array,
        0,
        255,
    ).astype(
        np.uint8
    )

    result_image = Image.fromarray(
        overlay_array
    ).convert("RGB")

    # --------------------------------------------------------
    # UNIQUE FILENAMES
    #
    # UUID is CRITICAL for multiple uploaded images.
    # No Grad-CAM file can overwrite another image's result.
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    heatmap_filename = (
        f"heatmap_{unique_id}.png"
    )

    overlay_filename = (
        f"gradcam_{unique_id}.png"
    )

    heatmap_path = os.path.join(
        GRADCAM_DIR,
        heatmap_filename,
    )

    overlay_path = os.path.join(
        GRADCAM_DIR,
        overlay_filename,
    )

    heatmap_rgb.save(
        heatmap_path,
        format="PNG",
    )

    result_image.save(
        overlay_path,
        format="PNG",
    )

    # --------------------------------------------------------
    # VERIFY BOTH FILES
    # --------------------------------------------------------

    if not os.path.exists(
        overlay_path
    ):

        raise IOError(
            "Grad-CAM overlay was not created."
        )

    print(
        "Heatmap saved:",
        heatmap_path,
    )

    print(
        "Grad-CAM overlay saved:",
        overlay_path,
    )

    print(
        "Grad-CAM generation completed successfully."
    )

    print("=" * 70)

    # Return ONLY the filename because app.py stores the filename
    # and the /gradcam-image/<filename> route serves the file.
    return overlay_filename