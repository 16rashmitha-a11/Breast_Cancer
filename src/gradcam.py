import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# ============================================================
# DENSENET121 BINARY CLASSIFICATION - GRAD-CAM
# ============================================================

print("=" * 70)
print("DENSENET121 BINARY CLASSIFICATION - GRAD-CAM")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

MODEL_PATH = "densenet121_binary_best.keras"

DATASET_PATH = (
    r"C:\Users\16ras\Downloads\breast"
    r"\BreakHis - Breast Cancer Histopathological Database"
    r"\dataset_cancer_v1"
    r"\dataset_cancer_v1"
    r"\classificacao_binaria"
)

OUTPUT_PATH = "gradcam_result.png"

IMG_SIZE = (224, 224)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ------------------------------------------------------------
# FIND AN IMAGE
# ------------------------------------------------------------

print("\nSearching for images...")

image_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*.png"),
    recursive=True
)

print("Total images found:", len(image_files))

if len(image_files) == 0:
    raise Exception("No PNG images found.")

# Select first image
img_path = image_files[0]

print("\nSelected image:")
print(img_path)

# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

img = image.load_img(img_path, target_size=IMG_SIZE)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

prediction = model.predict(img_array, verbose=0)[0][0]

if prediction >= 0.5:
    predicted_class = "Malignant"
    confidence = prediction * 100
else:
    predicted_class = "Benign"
    confidence = (1 - prediction) * 100

print("\nPrediction:")
print("Class      :", predicted_class)
print("Confidence :", f"{confidence:.2f}%")

# ------------------------------------------------------------
# FIND DENSENET121 BASE MODEL
# ------------------------------------------------------------

print("\nFinding DenseNet121 base model...")

base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "densenet" in layer.name.lower():
            base_model = layer
            break

if base_model is None:
    raise Exception("DenseNet121 base model not found.")

print("Base model:", base_model.name)

# ------------------------------------------------------------
# FIND LAST CONVOLUTIONAL LAYER
# ------------------------------------------------------------

last_conv_layer = None

for layer in reversed(base_model.layers):
    try:
        output_shape = layer.output_shape

        if len(output_shape) == 4:
            last_conv_layer = layer
            break

    except:
        continue

if last_conv_layer is None:
    raise Exception("Could not find convolutional layer.")

print("Using layer:", last_conv_layer.name)

# ------------------------------------------------------------
# CREATE FEATURE MODEL
# ------------------------------------------------------------

feature_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

# ------------------------------------------------------------
# CREATE CLASSIFICATION HEAD
# ------------------------------------------------------------

print("\nPreparing classification head...")

# The layers after DenseNet121 form the classification head.
base_index = model.layers.index(base_model)

head_layers = model.layers[base_index + 1:]

print("Classification layers:")

for layer in head_layers:
    print(" -", layer.name)

# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------

print("\nGenerating Grad-CAM...")

with tf.GradientTape() as tape:

    conv_outputs, base_output = feature_model(img_array)

    x = base_output

    # Pass DenseNet output through the remaining classification layers
    for layer in head_layers:
        x = layer(x)

    score = x[:, 0]

gradients = tape.gradient(score, conv_outputs)

# Average gradients over spatial dimensions
pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

# Weight feature maps
heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

# ReLU
heatmap = tf.maximum(heatmap, 0)

# Normalize
max_value = tf.reduce_max(heatmap)

if max_value != 0:
    heatmap /= max_value

heatmap = heatmap.numpy()

# ------------------------------------------------------------
# CREATE GRAD-CAM IMAGE
# ------------------------------------------------------------

original_img = image.load_img(img_path)

original_array = image.img_to_array(original_img).astype("uint8")

heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    (original_array.shape[0], original_array.shape[1])
).numpy()

heatmap_resized = np.squeeze(heatmap_resized)

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

# Original image
plt.subplot(1, 2, 1)

plt.imshow(original_array)

plt.title(
    f"Original Image\n"
    f"Prediction: {predicted_class} ({confidence:.2f}%)"
)

plt.axis("off")

# Grad-CAM
plt.subplot(1, 2, 2)

plt.imshow(original_array)

plt.imshow(
    heatmap_resized,
    cmap="jet",
    alpha=0.45
)

plt.title("Grad-CAM Visualization")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# COMPLETED
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GRAD-CAM COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nPrediction :", predicted_class)
print("Confidence :", f"{confidence:.2f}%")

print("\nGrad-CAM saved at:")
print(os.path.abspath(OUTPUT_PATH))