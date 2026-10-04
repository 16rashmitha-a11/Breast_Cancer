document.addEventListener("DOMContentLoaded", function () {

    /* =========================================================
       MOBILE SIDEBAR
    ========================================================= */

    const mobileMenu = document.getElementById("mobileMenu");
    const sidebar = document.getElementById("sidebar");

    if (mobileMenu && sidebar) {

        mobileMenu.addEventListener("click", function () {

            sidebar.classList.toggle("mobile-open");

        });

    }


    /* =========================================================
       CLOSE MOBILE SIDEBAR WHEN LINK IS CLICKED
    ========================================================= */

    if (sidebar) {

        const navLinks = sidebar.querySelectorAll("a");

        navLinks.forEach(function (link) {

            link.addEventListener("click", function () {

                sidebar.classList.remove("mobile-open");

            });

        });

    }


    /* =========================================================
       CONFIDENCE BARS
    ========================================================= */

    const confidenceBars =
        document.querySelectorAll(".confidence-fill");

    confidenceBars.forEach(function (bar) {

        let value =
            parseFloat(
                bar.getAttribute("data-confidence")
            );

        if (isNaN(value)) {
            value = 0;
        }

        /*
         * If backend provides decimal:
         * 0.95 -> 95
         *
         * If backend already provides:
         * 95 -> 95
         */

        if (value > 0 && value <= 1) {
            value = value * 100;
        }

        value = Math.max(
            0,
            Math.min(100, value)
        );

        setTimeout(function () {

            bar.style.width = value + "%";

        }, 100);

    });


    /* =========================================================
       FILE UPLOAD PREVIEW
    ========================================================= */

    const imageInput =
        document.getElementById("imageInput");

    const imagePreview =
        document.getElementById("imagePreview");

    const fileName =
        document.getElementById("fileName");

    const fileCount =
        document.getElementById("fileCount");

    const selectedImageSection =
        document.getElementById("selectedImageSection");


    if (imageInput) {

        imageInput.addEventListener(
            "change",
            function () {

                const files = imageInput.files;

                if (!files || files.length === 0) {

                    if (fileName) {
                        fileName.textContent =
                            "No image selected";
                    }

                    if (fileCount) {
                        fileCount.textContent = "0";
                    }

                    if (imagePreview) {

                        imagePreview.innerHTML = `
                            <div class="preview-placeholder">
                                Image preview will appear here
                            </div>
                        `;

                    }

                    return;
                }


                if (fileName) {

                    fileName.textContent =
                        files[0].name;

                }


                if (fileCount) {

                    fileCount.textContent =
                        files.length;

                }


                if (selectedImageSection) {

                    selectedImageSection.classList.add(
                        "has-selection"
                    );

                }


                if (imagePreview) {

                    imagePreview.innerHTML = "";

                    const firstFile = files[0];

                    if (
                        firstFile.type.startsWith("image/")
                    ) {

                        const reader =
                            new FileReader();

                        reader.onload =
                            function (event) {

                                const img =
                                    document.createElement("img");

                                img.src =
                                    event.target.result;

                                img.alt =
                                    "Selected histopathology image";

                                imagePreview.appendChild(img);

                            };

                        reader.readAsDataURL(firstFile);

                    }

                }

            }
        );

    }


    /* =========================================================
       UPLOAD AREA DRAG EFFECT
    ========================================================= */

    const uploadArea =
        document.getElementById("uploadArea");

    if (uploadArea && imageInput) {

        ["dragenter", "dragover"].forEach(
            function (eventName) {

                uploadArea.addEventListener(
                    eventName,
                    function (event) {

                        event.preventDefault();

                        uploadArea.classList.add(
                            "drag-active"
                        );

                    }
                );

            }
        );


        ["dragleave", "drop"].forEach(
            function (eventName) {

                uploadArea.addEventListener(
                    eventName,
                    function (event) {

                        event.preventDefault();

                        uploadArea.classList.remove(
                            "drag-active"
                        );

                    }
                );

            }
        );


        uploadArea.addEventListener(
            "drop",
            function (event) {

                const droppedFiles =
                    event.dataTransfer.files;

                if (
                    droppedFiles &&
                    droppedFiles.length > 0
                ) {

                    imageInput.files =
                        droppedFiles;

                    imageInput.dispatchEvent(
                        new Event("change")
                    );

                }

            }
        );

    }


    /* =========================================================
       FORM SUBMIT STATE
    ========================================================= */

    const predictionForm =
        document.getElementById("predictionForm");

    if (predictionForm) {

        predictionForm.addEventListener(
            "submit",
            function () {

                const submitButton =
                    predictionForm.querySelector(
                        'button[type="submit"]'
                    );

                if (submitButton) {

                    submitButton.disabled = true;

                    submitButton.textContent =
                        "Analyzing...";

                }

            }
        );

    }


    /* =========================================================
       AUTO HIDE FLASH MESSAGES
    ========================================================= */

    const flashes =
        document.querySelectorAll(".flash");

    flashes.forEach(function (flash) {

        setTimeout(function () {

            flash.classList.add("flash-hide");

        }, 5000);

    });

});