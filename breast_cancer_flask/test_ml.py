from ml_service import predict_image


IMAGE_PATH = r"C:\Users\16ras\Downloads\breast\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_multiclasse\400X\tubular_adenoma\SOB_B_TA-14-3411F-400-010.png"


result = predict_image(
    IMAGE_PATH
)

print("\n")
print("=" * 60)
print("BREAST CANCER PREDICTION")
print("=" * 60)

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

print("\nAll subclass probabilities:")

for name, probability in result[
    "subclass_probabilities"
].items():

    print(
        f"{name}: {probability}%"
    )

print("=" * 60)