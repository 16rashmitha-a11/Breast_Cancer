import tensorflow as tf
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

efficientnet_path = os.path.join(
    BASE_DIR,
    "models",
    "efficientnet_final.keras"
)

xception_path = os.path.join(
    BASE_DIR,
    "models",
    "xception_final.keras"
)


print("=" * 60)
print("TESTING ADDITIONAL MODELS")
print("=" * 60)


print("\nLoading EfficientNetB0...")

efficientnet = tf.keras.models.load_model(
    efficientnet_path,
    compile=False
)

print("EfficientNetB0 loaded successfully.")

print("Input:", efficientnet.input_shape)
print("Output:", efficientnet.output_shape)


print("\nLoading Xception...")

xception = tf.keras.models.load_model(
    xception_path,
    compile=False
)

print("Xception loaded successfully.")

print("Input:", xception.input_shape)
print("Output:", xception.output_shape)


print("\n" + "=" * 60)
print("ALL MODELS LOADED SUCCESSFULLY")
print("=" * 60)