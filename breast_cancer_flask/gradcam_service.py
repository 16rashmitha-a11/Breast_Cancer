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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "uploads",
    "gradcam",
)

os.makedirs(GRADCAM_DIR, exist_ok=True)


# ============================================================
# DENSENET121 TARGET LAYER
# ============================================================
#
# We use the last convolutional feature map of DenseNet121.
#
# conv5_block16_concat = 7 x 7 x 1024
#
# The important difference from the old code is that after
# this feature map we continue through DenseNet's remaining
# BN/ReLU layers AND then through the actual classifier.
#
# ============================================================

TARGET_LAYER_NAME = "conv5_block16_concat"


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Preprocess the image exactly as DenseNet121 expects.

    Input:
        image_path

    Output:
        Tensor/array with shape (1, 224, 224, 3)
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = (
        tf.keras.applications.densenet.preprocess_input(
            image_array
        )
    )

    return image_array


# ============================================================
# FIND DENSENET121 BASE MODEL
# ============================================================

def get_densenet_base(model):
    """
    Find the nested DenseNet121 model inside the classifier.
    """

    # --------------------------------------------------------
    # First try the expected name.
    # --------------------------------------------------------

    try:
        base_model = model.get_layer("densenet121")

        print(
            "DenseNet121 base model found:",
            base_model.name
        )

        return base_model

    except Exception:
        pass


    # --------------------------------------------------------
    # Search nested models.
    # --------------------------------------------------------

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if "densenet" in layer.name.lower():

                print(
                    "DenseNet base model found:",
                    layer.name
                )

                return layer


    # --------------------------------------------------------
    # Search by name.
    # --------------------------------------------------------

    for layer in model.layers:

        if "densenet" in layer.name.lower():

            print(
                "DenseNet-like layer found:",
                layer.name
            )

            return layer


    raise ValueError(
        "DenseNet121 base model could not be found "
        "inside the subclass model."
    )


# ============================================================
# FIND TARGET LAYER
# ============================================================

def get_target_layer(base_model):
    """
    Find the last convolutional feature map.

    Preferred:
        conv5_block16_concat

    If unavailable, automatically search backwards for a
    suitable 4-D convolutional feature layer.
    """

    # --------------------------------------------------------
    # Preferred layer.
    # --------------------------------------------------------

    try:

        target_layer = base_model.get_layer(
            TARGET_LAYER_NAME
        )

        print(
            "Grad-CAM target layer:",
            target_layer.name
        )

        print(
            "Grad-CAM target output:",
            target_layer.output.shape
        )

        return target_layer

    except Exception as error:

        print(
            "Preferred Grad-CAM layer not found:",
            repr(error)
        )


    # --------------------------------------------------------
    # Fallback.
    # --------------------------------------------------------

    for layer in reversed(base_model.layers):

        try:

            output = layer.output
            shape = output.shape

            if (
                len(shape) == 4
                and shape[-1] is not None
            ):

                print(
                    "Fallback Grad-CAM layer:",
                    layer.name
                )

                print(
                    "Fallback output:",
                    shape
                )

                return layer

        except Exception:
            continue


    raise ValueError(
        "No suitable convolutional layer was found "
        "for Grad-CAM."
    )


# ============================================================
# FIND LAYER INDEX
# ============================================================

def get_layer_index(model, target_layer):
    """
    Find target layer index inside a model.
    """

    for index, layer in enumerate(model.layers):

        if layer is target_layer:
            return index

        if layer.name == target_layer.name:
            return index

    return None


# ============================================================
# APPLY LAYERS AFTER TARGET INSIDE DENSENET
# ============================================================

def apply_layers_after_target(
    base_model,
    target_layer,
    tensor
):
    """
    Continue the DenseNet121 forward pass after the Grad-CAM
    target layer.

    This is important because conv5_block16_concat is NOT the
    final output of DenseNet121.

    DenseNet continues through its final BN and ReLU.
    """

    target_index = get_layer_index(
        base_model,
        target_layer
    )

    if target_index is None:

        raise ValueError(
            "Could not find Grad-CAM target layer "
            "inside DenseNet121."
        )

    x = tensor

    following_layers = base_model.layers[
        target_index + 1:
    ]

    print(
        "Layers after Grad-CAM target:",
        len(following_layers)
    )

    for layer in following_layers:

        # ----------------------------------------------------
        # Ignore InputLayer if encountered.
        # ----------------------------------------------------

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):
            continue

        try:

            x = layer(
                x,
                training=False
            )

        except TypeError:

            x = layer(x)

    return x


# ============================================================
# APPLY OUTER CLASSIFIER
# ============================================================

def apply_outer_classifier(
    model,
    base_model,
    tensor
):
    """
    Continue from DenseNet121 output through the actual
    classifier layers of the trained model.

    This ensures that the Grad-CAM gradient corresponds to
    the same classifier prediction produced by the real model.
    """

    base_index = get_layer_index(
        model,
        base_model
    )

    if base_index is None:

        raise ValueError(
            "Could not find DenseNet121 base model "
            "inside the subclass model."
        )

    x = tensor

    following_layers = model.layers[
        base_index + 1:
    ]

    print(
        "Outer classifier layers:",
        len(following_layers)
    )

    for layer in following_layers:

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):
            continue

        try:

            x = layer(
                x,
                training=False
            )

        except TypeError:

            x = layer(x)

    return x


# ============================================================
# BUILD GRADIENT MODEL
# ============================================================

def build_gradcam_path(
    model,
    base_model,
    target_layer
):
    """
    Build the complete differentiable path:

        DenseNet input
            ↓
        target convolution
            ↓
        remaining DenseNet layers
            ↓
        classifier
            ↓
        8-class output

    This fixes the incomplete gradient path from the previous
    implementation.
    """

    base_input = base_model.input

    # --------------------------------------------------------
    # Get feature map.
    # --------------------------------------------------------

    feature_model = tf.keras.Model(
        inputs=base_input,
        outputs=target_layer.output,
        name="gradcam_feature_model"
    )

    return feature_model


# ============================================================
# CALCULATE GRADIENTS
# ============================================================

def calculate_gradcam(
    model,
    base_model,
    target_layer,
    image_array,
    predicted_class_index
):
    """
    Calculate Grad-CAM for the exact predicted subclass.
    """

    feature_model = build_gradcam_path(
        model,
        base_model,
        target_layer
    )

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # Forward pass to target feature map.
        # ----------------------------------------------------

        conv_outputs = feature_model(
            image_array,
            training=False
        )

        tape.watch(conv_outputs)

        # ----------------------------------------------------
        # Continue through the rest of DenseNet.
        # ----------------------------------------------------

        x = apply_layers_after_target(
            base_model,
            target_layer,
            conv_outputs
        )

        # ----------------------------------------------------
        # Continue through the actual classifier.
        # ----------------------------------------------------

        predictions = apply_outer_classifier(
            model,
            base_model,
            x
        )

        predictions = tf.convert_to_tensor(
            predictions
        )

        # ----------------------------------------------------
        # Validate model output.
        # ----------------------------------------------------

        if len(predictions.shape) != 2:

            raise ValueError(
                "Unexpected subclass model output shape: "
                f"{predictions.shape}"
            )

        number_of_classes = int(
            predictions.shape[-1]
        )

        class_index = int(
            predicted_class_index
        )

        if (
            class_index < 0
            or class_index >= number_of_classes
        ):

            raise ValueError(
                "Invalid Grad-CAM class index: "
                f"{class_index}"
            )

        # ----------------------------------------------------
        # Score for EXACT predicted subclass.
        # ----------------------------------------------------

        class_score = predictions[
            0,
            class_index
        ]

    # --------------------------------------------------------
    # Gradient of predicted class with respect to feature map.
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients are None. "
            "Grad-CAM cannot be generated."
        )

    return (
        conv_outputs,
        gradients,
        predictions
    )


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    conv_outputs,
    gradients
):
    """
    Convert convolutional features and gradients into a
    normalized Grad-CAM heatmap.
    """

    # --------------------------------------------------------
    # Remove batch dimension.
    # --------------------------------------------------------

    conv_outputs = conv_outputs[0]
    gradients = gradients[0]

    # --------------------------------------------------------
    # Standard Grad-CAM:
    #
    # weight of each feature channel =
    # mean gradient over spatial dimensions
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    # --------------------------------------------------------
    # Weighted feature maps.
    # --------------------------------------------------------

    weighted_features = (
        conv_outputs
        * pooled_gradients
    )

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    # --------------------------------------------------------
    # Positive evidence.
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # Clean invalid values.
    # --------------------------------------------------------

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Normalize.
    #
    # IMPORTANT:
    # Do NOT fail just because the maximum is extremely small.
    # Instead, use the strongest absolute response as a
    # fallback. This prevents unnecessary "Grad-CAM unavailable"
    # results.
    # --------------------------------------------------------

    max_value = float(
        np.max(heatmap)
    )

    if max_value > 1e-12:

        heatmap = (
            heatmap
            / max_value
        )

    else:

        # ----------------------------------------------------
        # Fallback to absolute gradient-weighted activation.
        # ----------------------------------------------------

        absolute_features = np.abs(
            conv_outputs.numpy()
        )

        absolute_gradients = np.abs(
            gradients.numpy()
        )

        fallback = np.sum(
            absolute_features
            * absolute_gradients,
            axis=-1
        )

        fallback = np.nan_to_num(
            fallback,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        fallback_max = float(
            np.max(fallback)
        )

        if fallback_max > 1e-12:

            heatmap = (
                fallback
                / fallback_max
            )

        else:

            # ------------------------------------------------
            # If the model produces virtually no gradient,
            # return a valid zero heatmap rather than crashing.
            # ------------------------------------------------

            heatmap = np.zeros_like(
                heatmap,
                dtype=np.float32
            )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # SUPPRESS WEAK / DIFFUSE ACTIVATION
    #
    # The DenseNet target feature map is only 7x7. After it is
    # resized to the original image, very small activations can
    # visually spread across a large area. Keep the stronger
    # Grad-CAM responses and suppress the weak background.
    #
    # This changes ONLY the visualization of the Grad-CAM.
    # It does NOT change the model prediction.
    # --------------------------------------------------------

    positive_values = heatmap[heatmap > 0]

    if positive_values.size > 0:
        activation_threshold = float(
            np.percentile(
                positive_values,
                75.0
            )
        )

        if activation_threshold > 1e-12:
            heatmap = np.where(
                heatmap >= activation_threshold,
                heatmap,
                0.0
            )

            remaining_max = float(
                np.max(heatmap)
            )

            if remaining_max > 1e-12:
                heatmap = (
                    heatmap
                    / remaining_max
                )

    # Make strong responses visually more distinct without
    # changing their locations.
    heatmap = np.power(
        np.clip(heatmap, 0.0, 1.0),
        1.5
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    return heatmap


# ============================================================
# RESIZE HEATMAP
# ============================================================

def resize_heatmap(
    heatmap,
    original_size
):
    """
    Resize the 7x7/feature-map heatmap to the exact original
    image size.
    """

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255.0
        )
    )

    heatmap_image = heatmap_image.resize(
        original_size,
        Image.Resampling.BILINEAR
    )

    heatmap = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        )
        / 255.0
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    return heatmap


# ============================================================
# CREATE COLORED HEATMAP
# ============================================================

def colorize_heatmap(
    heatmap
):
    """
    Convert grayscale heatmap into a standard Jet Grad-CAM
    visualization.
    """

    jet = colormaps["jet"]

    colored_heatmap = jet(
        heatmap
    )

    colored_heatmap = (
        colored_heatmap[:, :, :3]
        * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        colored_heatmap
    ).convert("RGB")


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap_rgb,
    heatmap
):
    """
    Overlay Grad-CAM on the original histopathology image.

    The original image remains visible while high-attention
    regions become strongly colored.
    """

    original_array = np.asarray(
        original_image,
        dtype=np.float32
    )

    heatmap_array = np.asarray(
        heatmap_rgb,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Adaptive transparency.
    #
    # Weak areas remain mostly original.
    # Strong areas become clearly visible.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Use heatmap strength as the actual visibility mask.
    #
    # IMPORTANT:
    # There is intentionally NO minimum alpha here.
    # A zero/weak Grad-CAM response must leave the original
    # histopathology image unchanged.
    # --------------------------------------------------------

    alpha = (
        np.power(
            np.clip(heatmap, 0.0, 1.0),
            1.0
        ) * 0.65
    )

    # Completely suppress tiny residual responses.
    alpha = np.where(
        heatmap < 0.10,
        0.0,
        alpha
    )

    alpha = np.clip(
        alpha,
        0.0,
        0.65
    )

    alpha = np.expand_dims(
        alpha,
        axis=-1
    )

    overlay_array = (
        original_array
        * (1.0 - alpha)
        +
        heatmap_array
        * alpha
    )

    overlay_array = np.clip(
        overlay_array,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay_array
    ).convert("RGB")


# ============================================================
# SAVE IMAGES
# ============================================================

def save_gradcam_images(
    heatmap_rgb,
    overlay_image
):
    """
    Save both the pure heatmap and the overlay.
    """

    unique_id = uuid.uuid4().hex

    heatmap_filename = (
        f"heatmap_{unique_id}.png"
    )

    overlay_filename = (
        f"gradcam_{unique_id}.png"
    )

    heatmap_path = os.path.join(
        GRADCAM_DIR,
        heatmap_filename
    )

    overlay_path = os.path.join(
        GRADCAM_DIR,
        overlay_filename
    )

    heatmap_rgb.save(
        heatmap_path,
        format="PNG"
    )

    overlay_image.save(
        overlay_path,
        format="PNG"
    )

    # --------------------------------------------------------
    # Verify.
    # --------------------------------------------------------

    if not os.path.isfile(
        heatmap_path
    ):

        raise IOError(
            "Grad-CAM heatmap file was not created."
        )

    if not os.path.isfile(
        overlay_path
    ):

        raise IOError(
            "Grad-CAM overlay file was not created."
        )

    print(
        "Heatmap saved:",
        heatmap_path
    )

    print(
        "Grad-CAM overlay saved:",
        overlay_path
    )

    return (
        heatmap_filename,
        overlay_filename
    )


# ============================================================
# MAIN GRAD-CAM FUNCTION
# ============================================================

def make_gradcam(
    model,
    image_path,
    predicted_class_index
):
    """
    Generate Grad-CAM for the predicted subclass.

    Parameters
    ----------
    model:
        Your trained 8-class DenseNet121 subclass model.

    image_path:
        Original uploaded image.

    predicted_class_index:
        Index of the subclass predicted by the model.

    Returns
    -------
    str
        Filename of the Grad-CAM overlay.
    """

    print()
    print("=" * 70)
    print("GENERATING GRAD-CAM")
    print("IMAGE:", image_path)
    print(
        "PREDICTED CLASS INDEX:",
        predicted_class_index
    )
    print("=" * 70)

    # ========================================================
    # 1. VALIDATE IMAGE
    # ========================================================

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    # ========================================================
    # 2. PREPROCESS
    # ========================================================

    image_array = preprocess_image(
        image_path
    )

    # ========================================================
    # 3. FIND DENSENET
    # ========================================================

    base_model = get_densenet_base(
        model
    )

    # ========================================================
    # 4. FIND LAST CONVOLUTIONAL FEATURE MAP
    # ========================================================

    target_layer = get_target_layer(
        base_model
    )

    # ========================================================
    # 5. CALCULATE GRADIENT
    # ========================================================

    (
        conv_outputs,
        gradients,
        predictions
    ) = calculate_gradcam(
        model,
        base_model,
        target_layer,
        image_array,
        predicted_class_index
    )

    # ========================================================
    # 6. PRINT MODEL SCORE USED FOR GRAD-CAM
    # ========================================================

    class_index = int(
        predicted_class_index
    )

    model_score = float(
        predictions[0, class_index].numpy()
    )

    print(
        "Grad-CAM class score:",
        f"{model_score:.6f}"
    )

    print(
        "Feature map shape:",
        conv_outputs.shape
    )

    print(
        "Gradient shape:",
        gradients.shape
    )

    # ========================================================
    # 7. CREATE HEATMAP
    # ========================================================

    heatmap = create_heatmap(
        conv_outputs,
        gradients
    )

    print(
        "Heatmap shape:",
        heatmap.shape
    )

    print(
        "Heatmap minimum:",
        float(np.min(heatmap))
    )

    print(
        "Heatmap maximum:",
        float(np.max(heatmap))
    )
        # ========================================================
    # 7A. CALCULATE GRAD-CAM ACTIVATION EXTENT
    # ========================================================
    #
    # This measures the percentage of the feature-map
    # heatmap with strong activation.
    #
    # IMPORTANT:
    # This is NOT tumor size and NOT clinical tumor extent.
    # It is only an image-based Grad-CAM measurement.
    #

    activation_threshold = 0.50

    activated_pixels = np.sum(
        heatmap >= activation_threshold
    )

    total_pixels = heatmap.size

    if total_pixels > 0:
        activation_percentage = (
            float(activated_pixels)
            / float(total_pixels)
            * 100.0
        )
    else:
        activation_percentage = 0.0

    activation_percentage = round(
        activation_percentage,
        2
    )

    print(
        "Grad-CAM activation extent:",
        f"{activation_percentage:.2f}%"
    )

    # ========================================================
    # 8. LOAD ORIGINAL IMAGE
    # ========================================================

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_size = (
        original_image.width,
        original_image.height
    )

    # ========================================================
    # 9. RESIZE HEATMAP
    # ========================================================

    heatmap = resize_heatmap(
        heatmap,
        original_size
    )

    # ========================================================
    # 10. COLOR HEATMAP
    # ========================================================

    heatmap_rgb = colorize_heatmap(
        heatmap
    )

    # ========================================================
    # 11. CREATE OVERLAY
    # ========================================================

    overlay_image = create_overlay(
        original_image,
        heatmap_rgb,
        heatmap
    )

    # ========================================================
    # 12. SAVE
    # ========================================================

    (
        heatmap_filename,
        overlay_filename
    ) = save_gradcam_images(
        heatmap_rgb,
        overlay_image
    )

# ========================================================
    # 13. COMPLETE
    # ========================================================

    print(
        "Grad-CAM generation completed successfully."
    )
    print(
        "Returned file:",
        overlay_filename
    )

    print("=" * 70)

    # IMPORTANT:
    # ml_service.py expects the filename.
    return {
    "gradcam_path": overlay_filename,
    "activation_percentage": activation_percentage,
}