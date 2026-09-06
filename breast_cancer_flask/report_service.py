import os
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import inch

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether
)


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

ORIGINAL_DIR = os.path.join(
    UPLOAD_DIR,
    "original"
)

GRADCAM_DIR = os.path.join(
    UPLOAD_DIR,
    "gradcam"
)


os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    ORIGINAL_DIR,
    exist_ok=True
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# =========================================================
# ROBUST IMAGE PATH FINDER
# =========================================================

def find_image_path(path, image_type=None):

    if not path:
        return None

    path = str(path).strip()

    # -----------------------------------------------------
    # Remove URL-style prefix
    # -----------------------------------------------------

    path = path.replace(
        "\\",
        "/"
    )

    if "/uploaded-image/" in path:

        path = path.split(
            "/uploaded-image/",
            1
        )[1]

    if "/gradcam-image/" in path:

        path = path.split(
            "/gradcam-image/",
            1
        )[1]

    filename = os.path.basename(
        path
    )

    # -----------------------------------------------------
    # Candidate paths
    # -----------------------------------------------------

    candidates = []

    # Original exact path
    candidates.append(path)

    # Absolute version
    candidates.append(
        os.path.abspath(path)
    )

    # Project-relative version
    candidates.append(
        os.path.join(
            BASE_DIR,
            path
        )
    )

    # Original folder
    candidates.append(
        os.path.join(
            ORIGINAL_DIR,
            filename
        )
    )

    # Grad-CAM folder
    candidates.append(
        os.path.join(
            GRADCAM_DIR,
            filename
        )
    )

    # Upload root
    candidates.append(
        os.path.join(
            UPLOAD_DIR,
            filename
        )
    )

    # -----------------------------------------------------
    # Search candidates
    # -----------------------------------------------------

    checked = set()

    for candidate in candidates:

        if not candidate:
            continue

        candidate = os.path.normpath(
            candidate
        )

        if candidate in checked:
            continue

        checked.add(candidate)

        if os.path.isfile(candidate):

            print(
                "IMAGE FOUND:",
                candidate
            )

            return candidate

    # -----------------------------------------------------
    # If image_type is known, search recursively
    # -----------------------------------------------------

    search_dirs = []

    if image_type == "original":

        search_dirs.append(
            ORIGINAL_DIR
        )

    elif image_type == "gradcam":

        search_dirs.append(
            GRADCAM_DIR
        )

    else:

        search_dirs.extend([
            ORIGINAL_DIR,
            GRADCAM_DIR
        ])

    for search_dir in search_dirs:

        if not os.path.exists(search_dir):
            continue

        for root, dirs, files in os.walk(
            search_dir
        ):

            for file in files:

                if file.lower() == filename.lower():

                    found = os.path.join(
                        root,
                        file
                    )

                    print(
                        "IMAGE FOUND BY SEARCH:",
                        found
                    )

                    return found

    print(
        "IMAGE NOT FOUND:",
        path
    )

    return None


# =========================================================
# GENERATE REPORT
# =========================================================

def generate_report(prediction):

    # -----------------------------------------------------
    # Convert sqlite Row / dict
    # -----------------------------------------------------

    try:

        data = dict(prediction)

    except Exception:

        data = prediction

    # -----------------------------------------------------
    # Prediction ID
    # -----------------------------------------------------

    prediction_id = data.get(
        "id",
        "unknown"
    )

    # -----------------------------------------------------
    # Report filename
    # -----------------------------------------------------

    report_filename = (
        "breast_cancer_report_"
        + str(prediction_id)
        + ".pdf"
    )

    report_path = os.path.join(
        REPORT_DIR,
        report_filename
    )

    print()
    print(
        "=========================================="
    )
    print(
        "GENERATING BREASTCARE AI REPORT"
    )
    print(
        "Prediction ID:",
        prediction_id
    )
    print(
        "=========================================="
    )

    # =====================================================
    # DOCUMENT
    # =====================================================

    doc = SimpleDocTemplate(

        report_path,

        pagesize=A4,

        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # =====================================================
    # STYLES
    # =====================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=26,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.grey,
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=15,
        leading=19,
        spaceBefore=10,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=14
    )

    center_style = ParagraphStyle(
        "Center",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.grey,
        alignment=TA_LEFT
    )

    # =====================================================
    # STORY
    # =====================================================

    story = []

    # =====================================================
    # HEADER
    # =====================================================

    story.append(
        Paragraph(
            "BreastCare AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Smart Breast Cancer Detection and Diagnosis",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "<b>AI Prediction Report</b>",
            center_style
        )
    )

    story.append(
        Spacer(1, 18)
    )

    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    patient_name = data.get(
        "patient_name",
        data.get(
            "name",
            "N/A"
        )
    )

    patient_email = data.get(
        "patient_email",
        data.get(
            "email",
            "N/A"
        )
    )

    original_filename = data.get(
        "original_filename",
        "N/A"
    )

    created_at = data.get(
        "created_at",
        "N/A"
    )

    patient_table = Table(

        [
            [
                Paragraph(
                    "<b>Patient Name</b>",
                    normal_style
                ),
                Paragraph(
                    str(patient_name),
                    normal_style
                )
            ],

            [
                Paragraph(
                    "<b>Email</b>",
                    normal_style
                ),
                Paragraph(
                    str(patient_email),
                    normal_style
                )
            ],

            [
                Paragraph(
                    "<b>Image</b>",
                    normal_style
                ),
                Paragraph(
                    str(original_filename),
                    normal_style
                )
            ],

            [
                Paragraph(
                    "<b>Date</b>",
                    normal_style
                ),
                Paragraph(
                    str(created_at),
                    normal_style
                )
            ]
        ],

        colWidths=[
            1.7 * inch,
            4.8 * inch
        ]
    )

    patient_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.lightgrey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.whitesmoke
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        patient_table
    )

    story.append(
        Spacer(1, 20)
    )

    # =====================================================
    # PREDICTION RESULTS
    # =====================================================

    story.append(
        Paragraph(
            "Prediction Results",
            heading_style
        )
    )

    binary_prediction = data.get(
        "binary_prediction",
        "N/A"
    )

    binary_confidence = data.get(
        "binary_confidence",
        0
    )

    subclass_prediction = data.get(
        "subclass_prediction",
        "N/A"
    )

    subclass_confidence = data.get(
        "subclass_confidence",
        0
    )

    subclass_display = str(
        subclass_prediction
    ).replace(
        "_",
        " "
    ).title()

    result_table = Table(

        [
            [
                Paragraph(
                    "<b>Binary Classification</b>",
                    normal_style
                ),

                Paragraph(
                    str(binary_prediction).title(),
                    normal_style
                ),

                Paragraph(
                    f"<b>{float(binary_confidence):.2f}%</b>",
                    normal_style
                )
            ],

            [
                Paragraph(
                    "<b>Subclass Classification</b>",
                    normal_style
                ),

                Paragraph(
                    f"<b>{subclass_display}</b>",
                    normal_style
                ),

                Paragraph(
                    f"<b>{float(subclass_confidence):.2f}%</b>",
                    normal_style
                )
            ]
        ],

        colWidths=[
            2.4 * inch,
            2.4 * inch,
            1.7 * inch
        ]
    )

    result_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.lightgrey
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, 1),
                colors.HexColor(
                    "#fff1f7"
                )
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (2, 0),
                (2, -1),
                "CENTER"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        result_table
    )

    story.append(
        Spacer(1, 20)
    )

    # =====================================================
    # SUBCLASS PROBABILITIES
    # =====================================================

    story.append(
        Paragraph(
            "Subclass Probability Distribution",
            heading_style
        )
    )

    probabilities = data.get(
        "subclass_probabilities",
        {}
    )

    if isinstance(
        probabilities,
        str
    ):

        try:

            probabilities = json.loads(
                probabilities
            )

        except Exception:

            probabilities = {}

    probability_rows = [

        [
            Paragraph(
                "<b>Subclass</b>",
                normal_style
            ),

            Paragraph(
                "<b>Probability</b>",
                normal_style
            )
        ]
    ]

    for name, probability in probabilities.items():

        try:

            value = float(
                probability
            )

        except Exception:

            value = 0

        display_name = str(
            name
        ).replace(
            "_",
            " "
        ).title()

        probability_rows.append(

            [
                Paragraph(
                    f"<b>{display_name}</b>",
                    normal_style
                ),

                Paragraph(
                    f"{value:.2f}%",
                    normal_style
                )
            ]
        )

    probability_table = Table(

        probability_rows,

        colWidths=[
            4.8 * inch,
            1.7 * inch
        ]
    )

    probability_table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.lightgrey
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.whitesmoke
            ),
            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "RIGHT"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        probability_table
    )

    # =====================================================
    # ORIGINAL IMAGE
    # =====================================================

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Original Histopathology Image",
            heading_style
        )
    )

    original_db_path = data.get(
        "image_path"
    )

    print(
        "DATABASE ORIGINAL PATH:",
        original_db_path
    )

    original_image_path = find_image_path(
        original_db_path,
        "original"
    )

    if original_image_path:

        try:

            original_image = Image(
                original_image_path
            )

            original_image._restrictSize(
                6.2 * inch,
                4.5 * inch
            )

            story.append(
                original_image
            )

            story.append(
                Spacer(1, 10)
            )

        except Exception as error:

            print(
                "ORIGINAL IMAGE ERROR:",
                repr(error)
            )

            story.append(
                Paragraph(
                    "Original image could not be embedded.",
                    normal_style
                )
            )

    else:

        story.append(
            Paragraph(
                "Original image is not available.",
                normal_style
            )
        )

    # =====================================================
    # GRAD-CAM IMAGE
    # =====================================================

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Grad-CAM AI Attention Visualization",
            heading_style
        )
    )

    gradcam_db_path = data.get(
        "gradcam_path"
    )

    print(
        "DATABASE GRAD-CAM PATH:",
        gradcam_db_path
    )

    gradcam_image_path = find_image_path(
        gradcam_db_path,
        "gradcam"
    )

    if gradcam_image_path:

        try:

            gradcam_image = Image(
                gradcam_image_path
            )

            gradcam_image._restrictSize(
                6.2 * inch,
                4.5 * inch
            )

            story.append(
                gradcam_image
            )

            story.append(
                Spacer(1, 8)
            )

            story.append(
                Paragraph(
                    "Grad-CAM highlights regions "
                    "that contributed to the AI model's "
                    "prediction. It is an explainability "
                    "visualization and not a medical diagnosis.",
                    disclaimer_style
                )
            )

        except Exception as error:

            print(
                "GRAD-CAM IMAGE ERROR:",
                repr(error)
            )

            story.append(
                Paragraph(
                    "Grad-CAM image could not be embedded.",
                    normal_style
                )
            )

    else:

        story.append(
            Paragraph(
                "Grad-CAM visualization is not available "
                "for this prediction.",
                normal_style
            )
        )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    story.append(
        Spacer(1, 25)
    )

    story.append(
        Paragraph(
            "<b>Important:</b> This report is generated by "
            "an artificial intelligence research system. "
            "The results are intended for research and "
            "educational purposes only and must not be "
            "considered a medical diagnosis. A qualified "
            "healthcare professional should make the final "
            "clinical assessment.",
            disclaimer_style
        )
    )

    # =====================================================
    # BUILD
    # =====================================================

    doc.build(
        story
    )

    print(
        "=========================================="
    )

    print(
        "PDF REPORT GENERATED:"
    )

    print(
        report_path
    )

    print(
        "=========================================="
    )

    return report_path