
from pathlib import Path
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

DETECTOR_PATH = MODEL_DIR / "best.pt"
CLASSIFIER_PATH = MODEL_DIR / "best_model.pt"

# Use GPU if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detector settings
DETECTION_CONFIDENCE = 0.01
DETECTION_IOU = 0.7
DETECTION_IMAGE_SIZE = 640
MAX_DETECTIONS = 1000


# Load parking-slot detector
if not DETECTOR_PATH.exists():
    raise FileNotFoundError(f"Detector not found: {DETECTOR_PATH}")

print("Loading parking-slot detector...")
detector = YOLO(str(DETECTOR_PATH))


# Load occupancy classifier
if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")

print("Loading occupancy classifier...")

classifier = models.mobilenet_v3_small(weights=None)
classifier.classifier[3] = nn.Linear(
    classifier.classifier[3].in_features, 2
)

checkpoint = torch.load(
    str(CLASSIFIER_PATH),
    map_location=DEVICE,
    weights_only=False
)

state_dict = (
    checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
    else checkpoint
)

classifier.load_state_dict(state_dict)
classifier = classifier.to(DEVICE)
classifier.eval()

print(f"Both models loaded successfully on {DEVICE}.")
print("Detector classes:", detector.names)


# Same preprocessing used during classifier training
classifier_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def classify_crop(crop_image):
    tensor = classifier_transform(crop_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(classifier(tensor), dim=1)[0]

    class_id = int(probabilities.argmax().item())
    confidence = float(probabilities[class_id].item())

    status = "Empty" if class_id == 0 else "Occupied"
    return status, confidence


def annotate_parking_image(image_path):
    results = detector.predict(
        source=str(image_path),
        imgsz=DETECTION_IMAGE_SIZE,
        conf=DETECTION_CONFIDENCE,
        iou=DETECTION_IOU,
        max_det=MAX_DETECTIONS,
        augment=True,
        verbose=False
    )

    result = results[0]

    with Image.open(image_path) as source:
        output_image = source.convert("RGB")

    draw = ImageDraw.Draw(output_image)

    total_slots = 0
    empty_slots = 0
    occupied_slots = 0
    slot_details = []

    boxes = result.boxes

    print(f"Uploaded image size: {result.orig_shape}")
    print(f"YOLO detections: {len(boxes)}")
    print(
        "Detection confidences:",
        [round(float(value), 3) for value in boxes.conf.tolist()]
    )

    for box in boxes:
        x1, y1, x2, y2 = [
            int(value) for value in box.xyxy[0].tolist()
        ]

        # Keep box coordinates within the image.
        x1 = max(0, min(x1, output_image.width - 1))
        y1 = max(0, min(y1, output_image.height - 1))
        x2 = max(0, min(x2, output_image.width))
        y2 = max(0, min(y2, output_image.height))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = output_image.crop((x1, y1, x2, y2))

        if crop.width == 0 or crop.height == 0:
            continue

        status, confidence = classify_crop(crop)

        total_slots += 1

        if status == "Empty":
            empty_slots += 1
            color = (0, 200, 83)       # Green
        else:
            occupied_slots += 1
            color = (255, 59, 48)      # Red

        # Draw colored rectangles only, with no text labels.
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3
        )

        slot_details.append({
            "number": total_slots,
            "status": status,
            "confidence": round(confidence * 100, 2)
        })

    occupancy_percentage = (
        round(occupied_slots / total_slots * 100, 2)
        if total_slots > 0
        else 0
    )

    print(f"Slots classified: {total_slots}")
    print(f"Empty: {empty_slots} | Occupied: {occupied_slots}")

    return (
        output_image,
        total_slots,
        empty_slots,
        occupied_slots,
        occupancy_percentage,
        slot_details
    )


@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    image_url = None
    result_url = None
    total_slots = None
    empty_slots = None
    occupied_slots = None
    occupancy_percentage = None
    slot_details = []

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            message = "Please select a parking-lot image."

        elif not allowed_file(file.filename):
            message = "Upload a JPG, JPEG, PNG or WebP image."

        else:
            unique_id = uuid.uuid4().hex
            safe_name = secure_filename(file.filename)
            saved_name = f"{unique_id}_{safe_name}"
            image_path = UPLOAD_DIR / saved_name

            try:
                file.save(image_path)

                (
                    annotated_image,
                    total_slots,
                    empty_slots,
                    occupied_slots,
                    occupancy_percentage,
                    slot_details
                ) = annotate_parking_image(image_path)

                result_name = f"result_{unique_id}.jpg"
                result_path = RESULT_DIR / result_name

                annotated_image.save(
                    result_path,
                    format="JPEG",
                    quality=95
                )

                image_url = f"/static/uploads/{saved_name}"
                result_url = f"/static/results/{result_name}"

                message = "Parking analysis completed successfully."

            except Exception:
                app.logger.exception("Parking analysis failed")
                message = (
                    "Analysis failed. Check the VS Code terminal "
                    "for the error details."
                )

    return render_template(
        "index.html",
        message=message,
        image_url=image_url,
        result_url=result_url,
        total_slots=total_slots,
        empty_slots=empty_slots,
        occupied_slots=occupied_slots,
        occupancy_percentage=occupancy_percentage,
        slot_details=slot_details
    )


@app.errorhandler(413)
def file_too_large(error):
    return "Image too large. Maximum upload size is 15 MB.", 413


if __name__ == "__main__":
    app.run(debug=True)



