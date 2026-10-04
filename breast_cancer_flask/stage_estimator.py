"""
Experimental AI-Assisted Cancer Stage Estimator

IMPORTANT:
This is NOT clinical TNM staging.

The BreakHis histopathology dataset does not contain
clinical Stage I-IV labels, tumor size, lymph-node status,
or metastasis information.

Therefore this module provides only an experimental
image-based AI score using existing model outputs and
Grad-CAM activation extent.
"""


def _safe_percentage(value):
    """
    Convert a value into a safe percentage between 0 and 100.
    """

    if value is None:
        return 0.0

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(100.0, value))


def calculate_activation_extent(heatmap):
    """
    Calculate the percentage of the heatmap with strong
    activation.

    This is an image-analysis measurement only.
    It does NOT represent actual tumor size.
    """

    if heatmap is None:
        return 0.0

    try:
        import numpy as np

        heatmap = np.asarray(
            heatmap,
            dtype=float
        )

        if heatmap.size == 0:
            return 0.0

        heatmap = np.nan_to_num(
            heatmap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        heatmap = np.clip(
            heatmap,
            0.0,
            1.0
        )

        # Strong Grad-CAM activation threshold.
        threshold = 0.50

        activated_pixels = np.sum(
            heatmap >= threshold
        )

        total_pixels = heatmap.size

        if total_pixels == 0:
            return 0.0

        percentage = (
            activated_pixels
            / total_pixels
            * 100.0
        )

        return round(
            float(percentage),
            2
        )

    except Exception:
        return 0.0


def calculate_stage_score(
    binary_prediction,
    binary_confidence,
    subclass_confidence,
    activation_percentage
):
    """
    Calculate an experimental AI stage score.

    The score combines:

        50% -> binary confidence
        25% -> subclass confidence
        25% -> Grad-CAM activation extent

    This is a project-defined experimental score,
    NOT a medical staging rule.
    """

    binary_confidence = _safe_percentage(
        binary_confidence
    )

    subclass_confidence = _safe_percentage(
        subclass_confidence
    )

    activation_percentage = _safe_percentage(
        activation_percentage
    )

    # ---------------------------------------------------------
    # BENIGN RESULT
    # ---------------------------------------------------------

    # Benign findings do not receive a cancer stage.
    if str(binary_prediction).lower() != "malignant":

        return {
            "stage": "Not Applicable",
            "score": None,
            "confidence": None
        }

    # ---------------------------------------------------------
    # MALIGNANT RESULT
    # ---------------------------------------------------------

    score = (
        binary_confidence * 0.50
        +
        subclass_confidence * 0.25
        +
        activation_percentage * 0.25
    )

    score = max(
        0.0,
        min(100.0, score)
    )

    # ---------------------------------------------------------
    # EXPERIMENTAL PROJECT-DEFINED BANDS
    # ---------------------------------------------------------

    if score < 25:

        stage = "Stage I"

    elif score < 50:

        stage = "Stage II"

    elif score < 75:

        stage = "Stage III"

    else:

        stage = "Stage IV"

    # This is an internal consistency measure based
    # on the two model confidence values.
    confidence = (
        binary_confidence
        +
        subclass_confidence
    ) / 2.0

    return {
        "stage": stage,
        "score": round(
            score,
            2
        ),
        "confidence": round(
            confidence,
            2
        )
    }


def estimate_stage(
    binary_prediction,
    binary_confidence,
    subclass_confidence,
    activation_percentage
):
    """
    Main function used by ml_service.py.
    """

    result = calculate_stage_score(
        binary_prediction=binary_prediction,
        binary_confidence=binary_confidence,
        subclass_confidence=subclass_confidence,
        activation_percentage=activation_percentage
    )

    result["notice"] = (
        "Experimental AI-assisted stage estimation. "
        "This result is derived from image-based AI "
        "features and is not a clinically validated "
        "TNM cancer stage."
    )

    result["activation_percentage"] = round(
        _safe_percentage(
            activation_percentage
        ),
        2
    )
    return result