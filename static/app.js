const video = document.getElementById("cameraVideo");
const overlay = document.getElementById("overlayCanvas");
const cameraStage = document.querySelector(".camera-stage");
const cameraEmpty = document.getElementById("cameraEmpty");
const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");
const cameraMode = document.getElementById("cameraMode");
const cameraStatus = document.getElementById("cameraStatus");
const faceCounter = document.getElementById("faceCounter");
const liveResults = document.getElementById("liveResults");
const modelInput = document.getElementById("modelInput");
const modelFileName = document.getElementById("modelFileName");
const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const refreshStatus = document.getElementById("refreshStatus");
const statusJson = document.getElementById("statusJson");
const faceDetectorName = document.getElementById("faceDetectorName");
const modelStatusChip = document.getElementById("modelStatusChip");
const modelStatusText = document.getElementById("modelStatusText");
const modelPanelHint = document.getElementById("modelPanelHint");
const modelMeterTitle = document.getElementById("modelMeterTitle");
const modelMeterSub = document.getElementById("modelMeterSub");
const modelMeter = document.querySelector(".model-meter");
const liveSignal = document.getElementById("liveSignal");

const captureCanvas = document.createElement("canvas");
const SMOOTHING_WINDOW = 5;
let modelReady = document.body.classList.contains("model-loaded");
let stream = null;
let predictInterval = null;
let isPredicting = false;
let lastResponse = null;
let faceHistory = [];

function setCameraState(message) {
    cameraStatus.textContent = message;
}

function setLiveSignal(message) {
    if (liveSignal) {
        liveSignal.textContent = message;
    }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function updateResults(faces) {
    faceCounter.textContent = `${faces.length} wajah`;

    if (!faces.length) {
        liveResults.classList.add("empty");
        liveResults.textContent = "Wajah belum terdeteksi.";
        setLiveSignal(stream ? "Scanning" : "Standby");
        return;
    }

    liveResults.classList.remove("empty");
    setLiveSignal(`${faces.length} wajah terdeteksi`);
    liveResults.innerHTML = faces.map((face) => `
        <article class="result-item ${face.quality_ok === false ? "warning" : ""}">
            ${face.quality_ok === false ? `
                <div>
                    <strong>Input belum layak</strong>
                    <span>${escapeHtml(face.quality_message || "Perbaiki posisi wajah")}</span>
                </div>
                <span class="confidence warning">Cek kualitas</span>
            ` : `
                <div>
                    <strong>Gender: ${escapeHtml(face.gender)}</strong>
                    <span>Estimasi usia: ${escapeHtml(face.age_range)} tahun</span>
                </div>
                <span class="confidence confidence-stack">
                    <span>Confidence Gender ${Number(face.confidence_percent).toFixed(1)}%</span>
                    <span class="confidence-track"><span style="width: ${Math.max(0, Math.min(100, Number(face.confidence_percent))).toFixed(1)}%"></span></span>
                </span>
            `}
        </article>
    `).join("");
}

function formatAgeRange(age, span = 4) {
    const low = Math.max(0, Math.round(age - span));
    const high = Math.round(age + span);
    return `${low}-${high}`;
}

function smoothFaces(faces) {
    if (!faces.length) {
        faceHistory = [];
        return [];
    }

    const nextHistory = [];

    const smoothedFaces = faces.map((face, index) => {
        if (face.quality_ok === false || !Number.isFinite(Number(face.age))) {
            nextHistory[index] = [];
            return face;
        }

        const previous = faceHistory[index] || [];
        const history = [...previous, Number(face.age)].slice(-SMOOTHING_WINDOW);
        nextHistory[index] = history;

        const smoothedAge = history.reduce((sum, age) => sum + age, 0) / history.length;

        return {
            ...face,
            age: Number(smoothedAge.toFixed(1)),
            age_range: formatAgeRange(smoothedAge),
            smoothing_window: history.length,
        };
    });

    faceHistory = nextHistory;
    return smoothedFaces;
}

function sizeOverlay() {
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    overlay.width = Math.max(1, Math.round(rect.width * dpr));
    overlay.height = Math.max(1, Math.round(rect.height * dpr));
    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;
    return { width: rect.width, height: rect.height, dpr };
}

function drawOverlay(response) {
    const ctx = overlay.getContext("2d");
    const { width, height, dpr } = sizeOverlay();

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    if (!response || !response.faces || !response.faces.length) {
        return;
    }

    const scaleX = width / response.image_width;
    const scaleY = height / response.image_height;

    response.faces.forEach((face) => {
        const box = face.box;
        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const boxWidth = box.width * scaleX;
        const boxHeight = box.height * scaleY;
        const isQualityWarning = face.quality_ok === false;
        const genderLabel = isQualityWarning
            ? "Input belum layak"
            : `Gender: ${face.gender} (${Number(face.confidence_percent).toFixed(1)}%)`;
        const ageLabel = isQualityWarning
            ? (face.quality_message || "Perbaiki posisi wajah")
            : `Estimasi usia: ${face.age_range} tahun`;
        const overlayColor = isQualityWarning ? "#f97316" : "#10b981";

        ctx.lineWidth = 3;
        ctx.strokeStyle = overlayColor;
        ctx.strokeRect(x, y, boxWidth, boxHeight);

        ctx.font = "700 14px system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
        const labelWidth = Math.min(
            Math.max(ctx.measureText(genderLabel).width, ctx.measureText(ageLabel).width) + 18,
            width - 12
        );
        const labelHeight = 46;
        const labelX = Math.min(Math.max(6, x), Math.max(6, width - labelWidth - 6));
        const labelY = y > labelHeight + 8 ? y - labelHeight - 6 : y + 8;

        ctx.fillStyle = overlayColor;
        ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(genderLabel, labelX + 9, labelY + 18, labelWidth - 16);
        ctx.fillText(ageLabel, labelX + 9, labelY + 36, labelWidth - 16);
    });
}

async function startCamera() {
    if (!modelReady) {
        setCameraState("Upload model terlebih dahulu.");
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraState("Browser tidak mendukung kamera.");
        return;
    }

    stopCamera();

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: cameraMode.value },
                width: { ideal: 640 },
                height: { ideal: 480 },
            },
            audio: false,
        });

        video.srcObject = stream;
        video.style.transform = "none";
        overlay.style.transform = "none";
        await video.play();

        cameraStage.classList.add("active");
        document.body.classList.add("camera-running");
        cameraEmpty.hidden = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        setCameraState("Kamera aktif");

        predictInterval = window.setInterval(predictCurrentFrame, 900);
        window.setTimeout(predictCurrentFrame, 350);
    } catch (error) {
        setCameraState(`Kamera gagal aktif: ${error.message}`);
        cameraStage.classList.remove("active");
        cameraEmpty.hidden = false;
    }
}

function stopCamera() {
    if (predictInterval) {
        window.clearInterval(predictInterval);
        predictInterval = null;
    }

    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }

    video.srcObject = null;
    lastResponse = null;
    faceHistory = [];
    drawOverlay(null);
    updateResults([]);
    cameraStage.classList.remove("active");
    document.body.classList.remove("camera-running");
    cameraEmpty.hidden = false;
    startButton.disabled = false;
    startButton.disabled = !modelReady;
    stopButton.disabled = true;
    setCameraState("Kamera mati");
    setLiveSignal("Standby");
}

async function predictCurrentFrame() {
    if (!stream || isPredicting || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return;
    }

    isPredicting = true;

    try {
        captureCanvas.width = video.videoWidth || 640;
        captureCanvas.height = video.videoHeight || 480;
        const ctx = captureCanvas.getContext("2d");
        ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

        const blob = await new Promise((resolve) => {
            captureCanvas.toBlob(resolve, "image/jpeg", 0.82);
        });

        if (!blob) {
            throw new Error("Frame kamera tidak bisa diproses.");
        }

        const formData = new FormData();
        formData.append("frame", blob, "camera-frame.jpg");

        const response = await fetch("/api/predict-frame", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok || !data.ok) {
            throw new Error(data.error || "Prediksi gagal.");
        }

        const smoothedFaces = smoothFaces(data.faces);
        const smoothedData = { ...data, faces: smoothedFaces, face_count: smoothedFaces.length };

        lastResponse = smoothedData;
        drawOverlay(smoothedData);
        updateResults(smoothedFaces);
        setCameraState(smoothedFaces.length ? "Prediksi stabil berjalan" : "Mencari wajah");
    } catch (error) {
        setCameraState(error.message);
        setLiveSignal("Perlu perhatian");
    } finally {
        isPredicting = false;
    }
}

function setModelUi(data) {
    modelReady = Boolean(data.model_ready);
    document.body.classList.toggle("model-loaded", modelReady);
    document.body.classList.toggle("model-empty", !modelReady);

    if (modelStatusChip) {
        modelStatusChip.classList.toggle("ready", modelReady);
        modelStatusChip.classList.toggle("error", !modelReady);
        modelStatusChip.textContent = modelReady ? "Model siap" : "Model belum siap";
    }

    const statusText = modelReady
        ? `${data.model_name || "Model"} siap digunakan${data.model_source ? ` dari ${data.model_source}` : ""}`
        : `Model belum siap: ${data.model_error || "Upload file model .pth atau .pt terlebih dahulu."}`;

    if (modelStatusText) {
        modelStatusText.textContent = statusText;
    }

    if (modelPanelHint) {
        modelPanelHint.textContent = modelReady ? "Model aktif di sesi ini." : "Menunggu file model.";
    }

    if (modelMeter) {
        modelMeter.classList.toggle("ready", modelReady);
        modelMeter.classList.toggle("pending", !modelReady);
        const meterCore = modelMeter.querySelector(".meter-core span");
        if (meterCore) {
            meterCore.textContent = modelReady ? "ON" : "OFF";
        }
    }

    if (modelMeterTitle) {
        modelMeterTitle.textContent = modelReady ? "Inference aktif" : "Inference nonaktif";
    }

    if (modelMeterSub) {
        modelMeterSub.textContent = statusText;
    }

    if (!stream) {
        startButton.disabled = !modelReady;
    }
}

async function loadStatus() {
    try {
        const response = await fetch("/status");
        const data = await response.json();
        statusJson.textContent = JSON.stringify(data, null, 2);
        setModelUi(data);

        if (faceDetectorName && data.face_detector) {
            faceDetectorName.textContent = data.face_detector;
        }
    } catch (error) {
        statusJson.textContent = JSON.stringify({ error: error.message }, null, 2);
    }
}

function configureFileDrop(input, label, labelText, fallbackText) {
    if (!input || !label || !labelText) {
        return;
    }

    function updateName() {
        const file = input.files[0];
        labelText.textContent = file ? file.name : fallbackText;
    }

    label.addEventListener("dragover", (event) => {
        event.preventDefault();
        label.classList.add("drag-active");
    });

    label.addEventListener("dragleave", () => {
        label.classList.remove("drag-active");
    });

    label.addEventListener("drop", (event) => {
        event.preventDefault();
        label.classList.remove("drag-active");

        if (event.dataTransfer.files.length) {
            input.files = event.dataTransfer.files;
            updateName();
        }
    });

    input.addEventListener("change", updateName);
}

function bindSubmitState(formSelector, busyText) {
    const form = document.querySelector(formSelector);
    if (!form) {
        return;
    }

    form.addEventListener("submit", () => {
        const button = form.querySelector("button[type='submit']");
        if (button) {
            button.textContent = busyText;
            button.disabled = true;
        }
    });
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
refreshStatus.addEventListener("click", loadStatus);

cameraMode.addEventListener("change", () => {
    if (stream) {
        startCamera();
    }
});

configureFileDrop(
    imageInput,
    document.querySelector("label[for='imageInput']"),
    fileName,
    "JPG, PNG, WEBP"
);
configureFileDrop(
    modelInput,
    document.querySelector("label[for='modelInput']"),
    modelFileName,
    "PTH atau PT"
);
bindSubmitState("form[action$='/model/upload']", "Memuat Model");
bindSubmitState("form[action$='/predict']", "Memproses Gambar");

window.addEventListener("resize", () => drawOverlay(lastResponse));
