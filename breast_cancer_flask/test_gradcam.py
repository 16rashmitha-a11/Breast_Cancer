import os

from ml_service import (
    predict_image
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "test_images",
    "test1.png"
)


if not os.path.exists(
    IMAGE_PATH
):

    raise FileNotFoundError(
        "Test image not found:\n"
        + IMAGE_PATH
    )


print("=" * 70)
print("BREAST CANCER MODEL + GRAD-CAM TEST")
print("=" * 70)

print(
    "\nImage:",
    IMAGE_PATH
)

print(
    "\nRunning prediction..."
)

result = predict_image(
    IMAGE_PATH
)


print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    "\nBinary Prediction:",
    result["binary_prediction"]
)

print(
    "Binary Confidence:",
    result["binary_confidence"],
    "%"
)

print(
    "\nSubclass Prediction:",
    result["subclass_prediction"]
)

print(
    "Subclass Confidence:",
    result["subclass_confidence"],
    "%"
)

print(
    "\nAll Subclass Probabilities:"
)

for name, probability in (
    result[
        "subclass_probabilities"
    ].items()
):

    print(
        f"{name}: {probability}%"
    )


print(
    "\nGrad-CAM Generated:",
    result["gradcam_generated"]
)

print(
    "Grad-CAM Path:",
    result["gradcam_path"]
)

print("\n" + "=" * 70)

if result["gradcam_generated"]:

    print(
        "SUCCESS - Grad-CAM generated!"
    )

else:

    print(
        "Grad-CAM was NOT generated."
    )

print("=" * 70)