import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, jsonify, render_template, request

try:
    import mediapipe as mp
except ImportError:
    mp = None

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# =====================================================
# KONFIGURASI DASAR
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_gender_age_model.pth"

IMAGE_SIZE = 128
AGE_MAX = 116.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENDER_LABELS = ["Laki-laki", "Perempuan"]
FACE_MARGIN = 30
MAX_UPLOAD_SIZE = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
FACE_DETECTION_CONFIDENCE = 0.55
QUALITY_GATE_ENABLED = True
MIN_FACE_AREA_RATIO = 0.018
MIN_FACE_BRIGHTNESS = 45.0
MAX_FACE_BRIGHTNESS = 225.0
MIN_FACE_SHARPNESS = 28.0
AGE_CALIBRATION_SCALE = 1.0
AGE_CALIBRATION_OFFSET = 0.0


# =====================================================
# ARSITEKTUR MODEL 1: SimpleCNN
# Harus sama dengan model yang dipakai saat training di notebook.
# =====================================================
class SimpleCNN(nn.Module):
    def __init__(self, age_max: float = 116.0):
        super().__init__()
        self.age_max = age_max

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.gender_head = nn.Linear(128, 2)
        self.age_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)
        gender_logits = self.gender_head(x)
        age_years = torch.sigmoid(self.age_head(x)).squeeze(1) * self.age_max
        return gender_logits, age_years


# =====================================================
# ARSITEKTUR MODEL 2: ResNet18 Multi-Task
# Di app.py tidak memakai pretrained supaya tidak download ulang.
# Bobot pretrained/final sudah tersimpan di file .pth hasil training.
# =====================================================
class ResNet18MultiTask(nn.Module):
    def __init__(self, pretrained: bool = False, age_max: float = 116.0):
        super().__init__()
        self.age_max = age_max

        # Pakai weights=None agar tidak mencoba download dari internet.
        backbone = models.resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
        )
        self.gender_head = nn.Linear(256, 2)
        self.age_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        gender_logits = self.gender_head(x)
        age_years = torch.sigmoid(self.age_head(x)).squeeze(1) * self.age_max
        return gender_logits, age_years


# =====================================================
# LOAD MODEL FINAL
# =====================================================
def load_final_model():
    global IMAGE_SIZE, AGE_MAX, GENDER_LABELS

    if not MODEL_PATH.exists():
        return None, f"Model belum ditemukan: {MODEL_PATH}"

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # Format yang disarankan dari notebook:
        # torch.save({
        #     "model_name": best_model_name,
        #     "model_state_dict": final_model.state_dict(),
        #     "image_size": IMAGE_SIZE,
        #     "age_max": AGE_MAX,
        #     "gender_labels": ["Laki-laki", "Perempuan"],
        # }, MODEL_PATH)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_name = checkpoint.get("model_name", "ResNet18")
            state_dict = checkpoint["model_state_dict"]
            IMAGE_SIZE = int(checkpoint.get("image_size", IMAGE_SIZE))
            AGE_MAX = float(checkpoint.get("age_max", AGE_MAX))
            GENDER_LABELS = checkpoint.get("gender_labels", GENDER_LABELS)
        else:
            # Fallback kalau file .pth hanya berisi state_dict.
            model_name = "ResNet18"
            state_dict = checkpoint

        if str(model_name).lower().startswith("simple"):
            model = SimpleCNN(age_max=AGE_MAX)
        else:
            model = ResNet18MultiTask(pretrained=False, age_max=AGE_MAX)

        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()

        return model, None

    except Exception as exc:
        return None, f"Gagal load model: {exc}"


MODEL, MODEL_ERROR = load_final_model()


# Transform harus dibuat setelah load model, karena IMAGE_SIZE bisa berubah dari checkpoint.
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =====================================================
# FACE DETECTOR DAN FLASK APP
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_detector_lock = threading.Lock()
face_detector_name = "OpenCV Haar Cascade"
face_detector_error = None
mp_face_detector = None

try:
    has_mediapipe_face_detection = (
        mp is not None
        and hasattr(mp, "solutions")
        and hasattr(mp.solutions, "face_detection")
    )

    if has_mediapipe_face_detection:
        mp_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
        )
        face_detector_name = "MediaPipe Face Detection"
    elif mp is None:
        face_detector_error = "Package mediapipe belum terpasang."
    else:
        face_detector_error = "API mediapipe.solutions.face_detection tidak tersedia."
except Exception as exc:
    face_detector_error = f"Gagal inisialisasi MediaPipe: {exc}"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
app.config["TEMPLATES_AUTO_RELOAD"] = True


# =====================================================
# FUNGSI BANTUAN
# =====================================================
def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def add_margin_to_box(x, y, w, h, frame_shape, margin=FACE_MARGIN):
    """Membuat crop wajah berbentuk persegi dengan margin adaptif."""
    height, width = frame_shape[:2]
    face_size = max(int(w), int(h))
    adaptive_margin = max(int(margin), int(face_size * 0.28))
    side = min(face_size + adaptive_margin * 2, width, height)

    center_x = int(x + w / 2)
    center_y = int(y + h / 2)

    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))

    x1 = max(0, min(x1, width - side))
    y1 = max(0, min(y1, height - side))
    x2 = int(x1 + side)
    y2 = int(y1 + side)

    return int(x1), int(y1), x2, y2


def clamp_face_box(x, y, w, h, frame_shape):
    height, width = frame_shape[:2]

    x1 = max(int(x), 0)
    y1 = max(int(y), 0)
    x2 = min(int(x + w), width)
    y2 = min(int(y + h), height)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2 - x1, y2 - y1


def calibrate_age(age):
    corrected_age = age * AGE_CALIBRATION_SCALE + AGE_CALIBRATION_OFFSET
    return max(0.0, min(float(corrected_age), AGE_MAX))


def predict_pil_image(pil_img: Image.Image):
    """Prediksi gender dan usia dari satu gambar PIL."""
    if MODEL is None:
        raise RuntimeError(MODEL_ERROR or "Model belum siap.")

    img = pil_img.convert("RGB")
    augmented_images = [img, ImageOps.mirror(img)]
    x = torch.stack([transform(aug_img) for aug_img in augmented_images]).to(DEVICE)

    with torch.inference_mode():
        gender_logits, age_pred = MODEL(x)
        prob = torch.softmax(gender_logits, dim=1).mean(dim=0)
        gender_idx = int(torch.argmax(prob).item())
        confidence = float(prob[gender_idx].item())
        age = calibrate_age(float(age_pred.mean().item()))

    return {
        "gender": GENDER_LABELS[gender_idx],
        "gender_index": gender_idx,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "age": round(age, 1),
        "age_range": format_age_range(age),
    }


def evaluate_face_quality(frame_bgr, crop_bgr, box):
    """Validasi kualitas wajah sebelum prediksi."""
    x, y, w, h = [int(v) for v in box]
    frame_height, frame_width = frame_bgr.shape[:2]
    frame_area = max(frame_width * frame_height, 1)
    face_area_ratio = (w * h) / frame_area

    gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_crop))
    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

    issues = []
    if face_area_ratio < MIN_FACE_AREA_RATIO:
        issues.append("Dekatkan wajah")
    if brightness < MIN_FACE_BRIGHTNESS:
        issues.append("Pencahayaan kurang")
    elif brightness > MAX_FACE_BRIGHTNESS:
        issues.append("Pencahayaan terlalu terang")
    if sharpness < MIN_FACE_SHARPNESS:
        issues.append("Wajah terlalu blur")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "message": ", ".join(issues) if issues else "Kualitas wajah baik",
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "face_area_ratio": round(face_area_ratio, 4),
    }


def detect_faces_mediapipe_bgr(frame_bgr):
    """Deteksi wajah dengan MediaPipe dan kembalikan box format OpenCV."""
    if mp_face_detector is None:
        return []

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    with face_detector_lock:
        detections = mp_face_detector.process(frame_rgb).detections

    if not detections:
        return []

    height, width = frame_bgr.shape[:2]
    faces = []

    for detection in detections:
        relative_box = detection.location_data.relative_bounding_box
        x = relative_box.xmin * width
        y = relative_box.ymin * height
        w = relative_box.width * width
        h = relative_box.height * height
        box = clamp_face_box(x, y, w, h, frame_bgr.shape)

        if box is not None:
            faces.append(box)

    return faces


def detect_faces_haar_bgr(frame_bgr):
    """Deteksi wajah fallback dari frame BGR OpenCV."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return [tuple(int(v) for v in face) for face in faces]


def detect_faces_bgr(frame_bgr):
    """Deteksi wajah dari frame BGR memakai MediaPipe, fallback ke Haar Cascade."""
    faces = detect_faces_mediapipe_bgr(frame_bgr)

    if faces:
        return faces

    return detect_faces_haar_bgr(frame_bgr)


def format_age_range(age, span=4):
    low = max(0, int(round(age - span)))
    high = int(round(age + span))
    return f"{low}-{high}"


def frame_to_base64(frame_bgr):
    """Konversi frame BGR menjadi string base64 JPEG untuk HTML."""
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Gagal encode gambar.")
    return base64.b64encode(buffer).decode("utf-8")


def draw_prediction(frame_bgr, box, result):
    """Menggambar bounding box dan label prediksi pada frame."""
    x, y, w, h = [int(v) for v in box]
    quality_ok = result.get("quality_ok", True)
    if quality_ok:
        labels = [
            f"Gender: {result['gender']} ({result['confidence_percent']:.1f}%)",
            f"Estimasi usia: {result['age_range']} tahun",
        ]
        color = (16, 185, 129)
    else:
        labels = [
            "Input belum layak",
            result.get("quality_message", "Perbaiki posisi wajah"),
        ]
        color = (0, 149, 217)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    frame_height, frame_width = frame_bgr.shape[:2]
    label_width = max(cv2.getTextSize(label, font, font_scale, thickness)[0][0] for label in labels) + 18
    label_width = min(label_width, frame_width - 4)
    label_height = 50
    label_x = max(0, min(x, frame_width - label_width - 4))
    label_y = y - label_height - 6 if y > label_height + 8 else y + 6
    label_y = max(0, min(label_y, frame_height - label_height - 4))

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 3)
    cv2.rectangle(frame_bgr, (label_x, label_y), (label_x + label_width, label_y + label_height), color, -1)

    for index, label in enumerate(labels):
        cv2.putText(
            frame_bgr,
            label,
            (label_x + 8, label_y + 19 + index * 19),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return frame_bgr


def predict_faces_in_frame(frame_bgr, fallback_to_full_image=False):
    faces = detect_faces_bgr(frame_bgr)
    predictions = []

    sorted_faces = sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)
    for face in sorted_faces:
        x, y, w, h = [int(v) for v in face]
        x1, y1, x2, y2 = add_margin_to_box(x, y, w, h, frame_bgr.shape)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        quality = evaluate_face_quality(frame_bgr, crop, (x, y, w, h))
        if QUALITY_GATE_ENABLED and not quality["ok"]:
            predictions.append({
                "box": {"x": x, "y": y, "width": w, "height": h},
                "source": "face",
                "quality_ok": False,
                "quality": quality,
                "quality_message": quality["message"],
            })
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result = predict_pil_image(crop_pil)
        result["box"] = {"x": x, "y": y, "width": w, "height": h}
        result["source"] = "face"
        result["quality_ok"] = True
        result["quality"] = quality
        predictions.append(result)

    if not predictions and fallback_to_full_image:
        full_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = predict_pil_image(full_pil)
        height, width = frame_bgr.shape[:2]
        result["box"] = {"x": 0, "y": 0, "width": width, "height": height}
        result["source"] = "full_image"
        result["quality_ok"] = True
        predictions.append(result)

    return predictions


def read_image_file(file_storage):
    pil_img = Image.open(file_storage.stream)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return pil_img


def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def error_response(message, status_code):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": message}), status_code
    return render_template("error.html", message=message), status_code


# =====================================================
# RESPONSE HEADERS
# =====================================================
@app.after_request
def add_no_cache_headers(response):
    if request.endpoint in {"index", "predict_upload"} or request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def index():
    model_status = "Model siap digunakan" if MODEL is not None else f"Model belum siap: {MODEL_ERROR}"

    return render_template(
        "index.html",
        model_ready=MODEL is not None,
        model_status=model_status,
        device=str(DEVICE),
        image_size=IMAGE_SIZE,
        max_upload_mb=MAX_UPLOAD_SIZE // (1024 * 1024),
        face_detector=face_detector_name,
        face_detector_error=face_detector_error,
    )


@app.route("/status")
def status():
    return jsonify({
        "model_ready": MODEL is not None,
        "model_path": str(MODEL_PATH),
        "model_error": MODEL_ERROR,
        "device": str(DEVICE),
        "image_size": IMAGE_SIZE,
        "age_max": AGE_MAX,
        "gender_labels": GENDER_LABELS,
        "face_detector": face_detector_name,
        "face_detector_error": face_detector_error,
        "face_detection_confidence": FACE_DETECTION_CONFIDENCE,
        "quality_gate_enabled": QUALITY_GATE_ENABLED,
        "quality_thresholds": {
            "min_face_area_ratio": MIN_FACE_AREA_RATIO,
            "min_face_brightness": MIN_FACE_BRIGHTNESS,
            "max_face_brightness": MAX_FACE_BRIGHTNESS,
            "min_face_sharpness": MIN_FACE_SHARPNESS,
        },
        "age_calibration": {
            "scale": AGE_CALIBRATION_SCALE,
            "offset": AGE_CALIBRATION_OFFSET,
        },
    })


@app.route("/predict", methods=["POST"])
def predict_upload():
    if "image" not in request.files:
        return error_response("File gambar belum dipilih.", 400)

    file = request.files["image"]
    if not allowed_file(file.filename):
        return error_response("Format gambar harus JPG, JPEG, PNG, atau WEBP.", 400)

    try:
        pil_img = read_image_file(file)
    except Exception as exc:
        return error_response(f"Gagal membaca gambar: {exc}", 400)

    frame_bgr = pil_to_bgr(pil_img)

    try:
        predictions = predict_faces_in_frame(frame_bgr, fallback_to_full_image=True)
    except Exception as exc:
        return error_response(f"Gagal prediksi: {exc}", 500)

    annotated_frame = frame_bgr.copy()
    for prediction in predictions:
        box = prediction["box"]
        draw_prediction(
            annotated_frame,
            (box["x"], box["y"], box["width"], box["height"]),
            prediction,
        )

    encoded = frame_to_base64(annotated_frame)
    face_count = sum(1 for prediction in predictions if prediction["source"] == "face")

    return render_template(
        "result.html",
        predictions=predictions,
        image_data=encoded,
        face_count=face_count,
        used_full_image=face_count == 0,
    )


@app.route("/api/predict-frame", methods=["POST"])
def predict_frame():
    if "frame" not in request.files:
        return jsonify({"ok": False, "error": "Frame kamera belum terkirim."}), 400

    file = request.files["frame"]

    try:
        pil_img = read_image_file(file)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Gagal membaca frame: {exc}"}), 400

    frame_bgr = pil_to_bgr(pil_img)

    try:
        predictions = predict_faces_in_frame(frame_bgr, fallback_to_full_image=False)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Gagal prediksi: {exc}"}), 500

    height, width = frame_bgr.shape[:2]

    return jsonify({
        "ok": True,
        "faces": predictions,
        "face_count": len(predictions),
        "image_width": width,
        "image_height": height,
        "model_ready": MODEL is not None,
        "face_detector": face_detector_name,
    })


@app.errorhandler(413)
def request_entity_too_large(_exc):
    return error_response(f"Ukuran file maksimal {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.", 413)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
