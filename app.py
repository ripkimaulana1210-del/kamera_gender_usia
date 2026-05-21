import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, jsonify, redirect, render_template, request, url_for

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
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
UPLOADED_MODEL_PATH = MODEL_DIR / "uploaded_gender_age_model.pth"

IMAGE_SIZE = 128
AGE_MAX = 116.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENDER_LABELS = ["Laki-laki", "Perempuan"]
FACE_MARGIN = 30
MAX_FACE_PREDICTIONS = 5
MAX_UPLOAD_SIZE = 8 * 1024 * 1024
MAX_MODEL_UPLOAD_SIZE = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MODEL_EXTENSIONS = {"pth", "pt"}
FACE_DETECTION_CONFIDENCE = 0.55
MIN_MEDIAPIPE_FACE_SCORE = 0.55
QUALITY_GATE_ENABLED = True
MIN_FACE_AREA_RATIO = 0.018
MIN_FACE_BRIGHTNESS = 45.0
MAX_FACE_BRIGHTNESS = 225.0
MIN_FACE_SHARPNESS = 28.0
MIN_REAL_FACE_SKIN_RATIO = 0.035
MAX_LINE_ART_WHITE_RATIO = 0.42
MIN_LINE_ART_EDGE_DENSITY = 0.018
MAX_LINE_ART_SATURATION = 42.0
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
# LOAD MODEL DARI UPLOAD
# =====================================================
def load_checkpoint(model_path: Path):
    try:
        return torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=DEVICE)


def build_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model_from_file(model_path: Path):
    if not model_path.exists():
        return None, None, f"Model belum ditemukan: {model_path}"

    try:
        checkpoint = load_checkpoint(model_path)

        # Format yang disarankan dari notebook:
        # torch.save({
        #     "model_name": best_model_name,
        #     "model_state_dict": final_model.state_dict(),
        #     "image_size": IMAGE_SIZE,
        #     "age_max": AGE_MAX,
        #     "gender_labels": ["Laki-laki", "Perempuan"],
        # }, path_model)

        image_size = IMAGE_SIZE
        age_max = AGE_MAX
        gender_labels = list(GENDER_LABELS)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_name = checkpoint.get("model_name", "ResNet18")
            state_dict = checkpoint["model_state_dict"]
            image_size = int(checkpoint.get("image_size", image_size))
            age_max = float(checkpoint.get("age_max", age_max))
            gender_labels = checkpoint.get("gender_labels", gender_labels)
        else:
            # Fallback kalau file .pth hanya berisi state_dict.
            model_name = "ResNet18"
            state_dict = checkpoint

        if not isinstance(gender_labels, (list, tuple)) or len(gender_labels) < 2:
            gender_labels = ["Laki-laki", "Perempuan"]
        gender_labels = [str(gender_labels[0]), str(gender_labels[1])]

        if str(model_name).lower().startswith("simple"):
            model = SimpleCNN(age_max=age_max)
        else:
            model = ResNet18MultiTask(pretrained=False, age_max=age_max)

        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()

        metadata = {
            "model_name": str(model_name),
            "image_size": image_size,
            "age_max": age_max,
            "gender_labels": gender_labels,
        }
        return model, metadata, None

    except Exception as exc:
        return None, None, f"Gagal load model: {exc}"


MODEL_LOCK = threading.Lock()
MODEL = None
MODEL_ERROR = "Upload file model .pth atau .pt terlebih dahulu."
MODEL_NAME = None
MODEL_SOURCE = None
transform = build_transform(IMAGE_SIZE)


def activate_model(model, metadata, source_name):
    global IMAGE_SIZE, AGE_MAX, GENDER_LABELS, transform
    global MODEL, MODEL_ERROR, MODEL_NAME, MODEL_SOURCE

    with MODEL_LOCK:
        IMAGE_SIZE = int(metadata["image_size"])
        AGE_MAX = float(metadata["age_max"])
        GENDER_LABELS = list(metadata["gender_labels"])
        transform = build_transform(IMAGE_SIZE)
        MODEL = model
        MODEL_ERROR = None
        MODEL_NAME = metadata["model_name"]
        MODEL_SOURCE = source_name


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
mp_face_detectors = []

try:
    has_mediapipe_face_detection = (
        mp is not None
        and hasattr(mp, "solutions")
        and hasattr(mp.solutions, "face_detection")
    )

    if has_mediapipe_face_detection:
        mp_face_detectors = [
            mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            ),
            mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            ),
        ]
        mp_face_detector = mp_face_detectors[0]
        face_detector_name = "MediaPipe Multi-Face + OpenCV Haar"
    elif mp is None:
        face_detector_error = "Package mediapipe belum terpasang."
    else:
        face_detector_error = "API mediapipe.solutions.face_detection tidak tersedia."
except Exception as exc:
    face_detector_error = f"Gagal inisialisasi MediaPipe: {exc}"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MODEL_UPLOAD_SIZE
app.config["TEMPLATES_AUTO_RELOAD"] = True


# =====================================================
# FUNGSI BANTUAN
# =====================================================
def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def allowed_model_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_MODEL_EXTENSIONS


def get_uploaded_size(file_storage):
    try:
        current_position = file_storage.stream.tell()
        file_storage.stream.seek(0, 2)
        size = file_storage.stream.tell()
        file_storage.stream.seek(current_position)
        return size
    except (AttributeError, OSError):
        return None


def clean_filename(filename):
    return Path(str(filename).replace("\\", "/")).name


def get_model_status():
    with MODEL_LOCK:
        model_ready = MODEL is not None
        model_error = MODEL_ERROR
        model_name = MODEL_NAME
        model_source = MODEL_SOURCE
        image_size = IMAGE_SIZE
        age_max = AGE_MAX
        gender_labels = list(GENDER_LABELS)

    if model_ready:
        shown_name = model_name or "Model"
        source_text = f" dari {model_source}" if model_source else ""
        model_status = f"{shown_name} siap digunakan{source_text}"
    else:
        model_status = f"Model belum siap: {model_error}"

    return {
        "model_ready": model_ready,
        "model_error": model_error,
        "model_name": model_name,
        "model_source": model_source,
        "model_status": model_status,
        "image_size": image_size,
        "age_max": age_max,
        "gender_labels": gender_labels,
    }


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


def box_area(box):
    return max(int(box[2]), 0) * max(int(box[3]), 0)


def box_iou(box_a, box_b):
    ax, ay, aw, ah = [int(v) for v in box_a]
    bx, by, bw, bh = [int(v) for v in box_b]

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = box_area(box_a) + box_area(box_b) - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def merge_face_boxes(face_boxes, iou_threshold=0.35):
    merged = []

    for box in sorted(face_boxes, key=box_area, reverse=True):
        if all(box_iou(box, selected) < iou_threshold for selected in merged):
            merged.append(tuple(int(v) for v in box))

    return merged


def calibrate_age(age, age_max=None):
    max_age = AGE_MAX if age_max is None else age_max
    corrected_age = age * AGE_CALIBRATION_SCALE + AGE_CALIBRATION_OFFSET
    return max(0.0, min(float(corrected_age), max_age))


def predict_pil_image(pil_img: Image.Image):
    """Prediksi gender dan usia dari satu gambar PIL."""
    with MODEL_LOCK:
        model = MODEL
        model_error = MODEL_ERROR
        active_transform = transform
        gender_labels = list(GENDER_LABELS)
        age_max = AGE_MAX

    if model is None:
        raise RuntimeError(model_error or "Model belum siap.")

    img = pil_img.convert("RGB")
    x = active_transform(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        gender_logits, age_pred = model(x)
        prob = torch.softmax(gender_logits, dim=1)[0]
        gender_idx = int(torch.argmax(prob).item())
        confidence = float(prob[gender_idx].item())
        age = calibrate_age(float(age_pred[0].item()), age_max=age_max)

    return {
        "gender": gender_labels[gender_idx],
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
    hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ycrcb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)

    brightness = float(np.mean(gray_crop))
    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    saturation = float(np.mean(hsv_crop[:, :, 1]))
    white_ratio = float(np.mean((gray_crop > 235) & (hsv_crop[:, :, 1] < 45)))
    edges = cv2.Canny(gray_crop, 80, 160)
    edge_density = float(np.mean(edges > 0))

    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_crop)
    skin_mask = (
        (y_channel > 40)
        & (cr_channel >= 133)
        & (cr_channel <= 180)
        & (cb_channel >= 75)
        & (cb_channel <= 135)
    )
    skin_ratio = float(np.mean(skin_mask))
    looks_like_line_art = (
        white_ratio > MAX_LINE_ART_WHITE_RATIO
        and edge_density > MIN_LINE_ART_EDGE_DENSITY
        and saturation < MAX_LINE_ART_SATURATION
    )

    issues = []
    if face_area_ratio < MIN_FACE_AREA_RATIO:
        issues.append("Dekatkan wajah")
    if brightness < MIN_FACE_BRIGHTNESS:
        issues.append("Pencahayaan kurang")
    elif brightness > MAX_FACE_BRIGHTNESS:
        issues.append("Pencahayaan terlalu terang")
    if sharpness < MIN_FACE_SHARPNESS:
        issues.append("Wajah terlalu blur")
    if looks_like_line_art:
        issues.append("Gunakan foto wajah manusia, bukan gambar/coretan")
    elif skin_ratio < MIN_REAL_FACE_SKIN_RATIO and saturation < MAX_LINE_ART_SATURATION:
        issues.append("Wajah manusia belum cukup jelas")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "message": ", ".join(issues) if issues else "Kualitas wajah baik",
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "face_area_ratio": round(face_area_ratio, 4),
        "saturation": round(saturation, 1),
        "white_ratio": round(white_ratio, 4),
        "edge_density": round(edge_density, 4),
        "skin_ratio": round(skin_ratio, 4),
    }


def detect_faces_mediapipe_bgr(frame_bgr):
    """Deteksi wajah dengan MediaPipe dan kembalikan box format OpenCV."""
    if not mp_face_detectors:
        return []

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    height, width = frame_bgr.shape[:2]
    faces = []

    with face_detector_lock:
        for detector in mp_face_detectors:
            detections = detector.process(frame_rgb).detections

            if not detections:
                continue

            for detection in detections:
                score = float(detection.score[0]) if detection.score else 0.0
                if score < MIN_MEDIAPIPE_FACE_SCORE:
                    continue

                relative_box = detection.location_data.relative_bounding_box
                x = relative_box.xmin * width
                y = relative_box.ymin * height
                w = relative_box.width * width
                h = relative_box.height * height
                box = clamp_face_box(x, y, w, h, frame_bgr.shape)

                if box is not None:
                    faces.append(box)

    return merge_face_boxes(faces)


def detect_faces_haar_bgr(frame_bgr):
    """Deteksi wajah fallback dari frame BGR OpenCV."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = min(frame_bgr.shape[:2])
    min_face_size = max(36, int(min_side * 0.055))
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_face_size, min_face_size),
    )
    return [tuple(int(v) for v in face) for face in faces]


def detect_faces_bgr(frame_bgr):
    """Deteksi wajah manusia dari frame BGR."""
    faces = []
    faces.extend(detect_faces_mediapipe_bgr(frame_bgr))
    faces.extend(detect_faces_haar_bgr(frame_bgr))
    return merge_face_boxes(faces)


def format_age_range(age, max_width=5):
    center = int(round(age))
    low = max(0, center - (max_width // 2))
    high = low + max_width
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


def predict_faces_in_frame(frame_bgr):
    faces = detect_faces_bgr(frame_bgr)
    predictions = []
    rejected_predictions = []

    sorted_faces = sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)
    for face in sorted_faces:
        if len(predictions) >= MAX_FACE_PREDICTIONS:
            break

        x, y, w, h = [int(v) for v in face]
        x1, y1, x2, y2 = add_margin_to_box(x, y, w, h, frame_bgr.shape)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        quality = evaluate_face_quality(frame_bgr, crop, (x, y, w, h))
        if QUALITY_GATE_ENABLED and not quality["ok"]:
            rejected_predictions.append({
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

    if predictions:
        return predictions

    return rejected_predictions[:MAX_FACE_PREDICTIONS]


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
    if request.endpoint in {"index", "predict_upload", "upload_model"} or request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def index():
    runtime = get_model_status()

    return render_template(
        "index.html",
        model_ready=runtime["model_ready"],
        model_status=runtime["model_status"],
        device=str(DEVICE),
        image_size=runtime["image_size"],
        max_upload_mb=MAX_UPLOAD_SIZE // (1024 * 1024),
        max_model_upload_mb=MAX_MODEL_UPLOAD_SIZE // (1024 * 1024),
        face_detector=face_detector_name,
        face_detector_error=face_detector_error,
    )


@app.route("/status")
def status():
    runtime = get_model_status()

    return jsonify({
        "model_ready": runtime["model_ready"],
        "model_name": runtime["model_name"],
        "model_source": runtime["model_source"],
        "uploaded_model_path": str(UPLOADED_MODEL_PATH),
        "model_error": runtime["model_error"],
        "device": str(DEVICE),
        "image_size": runtime["image_size"],
        "age_max": runtime["age_max"],
        "gender_labels": runtime["gender_labels"],
        "face_detector": face_detector_name,
        "face_detector_error": face_detector_error,
        "face_detection_confidence": FACE_DETECTION_CONFIDENCE,
        "max_face_predictions": MAX_FACE_PREDICTIONS,
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


@app.route("/model/upload", methods=["POST"])
def upload_model():
    if "model_file" not in request.files:
        return error_response("File model belum dipilih.", 400)

    file = request.files["model_file"]
    original_filename = clean_filename(file.filename)

    if not original_filename:
        return error_response("File model belum dipilih.", 400)

    if not allowed_model_file(original_filename):
        return error_response("Format model harus .pth atau .pt.", 400)

    uploaded_size = get_uploaded_size(file)
    if uploaded_size is not None and uploaded_size > MAX_MODEL_UPLOAD_SIZE:
        max_mb = MAX_MODEL_UPLOAD_SIZE // (1024 * 1024)
        return error_response(f"Ukuran file model maksimal {max_mb} MB.", 413)

    temp_path = MODEL_DIR / "_uploaded_model_tmp.pth"

    try:
        file.save(temp_path)
        loaded_model, metadata, model_error = load_model_from_file(temp_path)

        if model_error is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
            return error_response(model_error, 400)

        temp_path.replace(UPLOADED_MODEL_PATH)
        activate_model(loaded_model, metadata, original_filename)
    except Exception as exc:
        return error_response(f"Gagal upload model: {exc}", 500)

    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict_upload():
    runtime = get_model_status()
    if not runtime["model_ready"]:
        return error_response(runtime["model_status"], 400)

    if "image" not in request.files:
        return error_response("File gambar belum dipilih.", 400)

    file = request.files["image"]
    if not allowed_file(file.filename):
        return error_response("Format gambar harus JPG, JPEG, PNG, atau WEBP.", 400)

    uploaded_size = get_uploaded_size(file)
    if uploaded_size is not None and uploaded_size > MAX_UPLOAD_SIZE:
        max_mb = MAX_UPLOAD_SIZE // (1024 * 1024)
        return error_response(f"Ukuran file gambar maksimal {max_mb} MB.", 413)

    try:
        pil_img = read_image_file(file)
    except Exception as exc:
        return error_response(f"Gagal membaca gambar: {exc}", 400)

    frame_bgr = pil_to_bgr(pil_img)

    try:
        predictions = predict_faces_in_frame(frame_bgr)
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
    face_count = sum(
        1
        for prediction in predictions
        if prediction["source"] == "face" and prediction.get("quality_ok", True)
    )

    return render_template(
        "result.html",
        predictions=predictions,
        image_data=encoded,
        face_count=face_count,
        no_face_detected=len(predictions) == 0,
        only_rejected_faces=len(predictions) > 0 and face_count == 0,
    )


@app.route("/api/predict-frame", methods=["POST"])
def predict_frame():
    runtime = get_model_status()
    if not runtime["model_ready"]:
        return jsonify({"ok": False, "error": runtime["model_status"], "model_ready": False}), 400

    if "frame" not in request.files:
        return jsonify({"ok": False, "error": "Frame kamera belum terkirim."}), 400

    file = request.files["frame"]
    uploaded_size = get_uploaded_size(file)
    if uploaded_size is not None and uploaded_size > MAX_UPLOAD_SIZE:
        max_mb = MAX_UPLOAD_SIZE // (1024 * 1024)
        return jsonify({"ok": False, "error": f"Ukuran frame maksimal {max_mb} MB."}), 413

    try:
        pil_img = read_image_file(file)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Gagal membaca frame: {exc}"}), 400

    frame_bgr = pil_to_bgr(pil_img)

    try:
        predictions = predict_faces_in_frame(frame_bgr)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Gagal prediksi: {exc}"}), 500

    height, width = frame_bgr.shape[:2]

    return jsonify({
        "ok": True,
        "faces": predictions,
        "face_count": len(predictions),
        "image_width": width,
        "image_height": height,
        "model_ready": True,
        "face_detector": face_detector_name,
    })


@app.errorhandler(413)
def request_entity_too_large(_exc):
    return error_response(f"Ukuran request maksimal {MAX_MODEL_UPLOAD_SIZE // (1024 * 1024)} MB.", 413)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
