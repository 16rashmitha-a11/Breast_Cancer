function previewImage(event) {

    const input = event.target;

    const previewContainer =
        document.getElementById(
            "previewContainer"
        );

    const preview =
        document.getElementById(
            "previewImage"
        );

    if (!input.files || !input.files[0]) {
        previewContainer.style.display = "none";
        return;
    }

    const file = input.files[0];

    const reader = new FileReader();

    reader.onload = function (e) {

        preview.src = e.target.result;

        previewContainer.style.display =
            "block";

    };

    reader.readAsDataURL(file);
}