let selectedFiles = [];
let latestPayload = null;

const els = {
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("fileInput"),
    fileHint: document.getElementById("fileHint"),
    singleBtn: document.getElementById("singleBtn"),
    batchBtn: document.getElementById("batchBtn"),
    exportBtn: document.getElementById("exportBtn"),
    status: document.getElementById("status"),
    threshold: document.getElementById("threshold"),
    thresholdValue: document.getElementById("thresholdValue"),
    refreshHistoryBtn: document.getElementById("refreshHistoryBtn"),
};

function setStatus(message, isError = false) {
    els.status.textContent = message;
    els.status.style.color = isError ? "#ff9bb7" : "#9eaed1";
}

function threshold() {
    return Number(els.threshold.value);
}

function updateThresholdLabel() {
    els.thresholdValue.textContent = threshold().toFixed(2);
}

function setFiles(files) {
    selectedFiles = Array.from(files || []);
    if (!selectedFiles.length) {
        els.fileHint.textContent = "No files selected";
        els.singleBtn.disabled = true;
        els.batchBtn.disabled = true;
        return;
    }
    els.fileHint.textContent = selectedFiles.length === 1
        ? `${selectedFiles[0].name} selected`
        : `${selectedFiles.length} files selected`;
    els.singleBtn.disabled = selectedFiles.length !== 1;
    els.batchBtn.disabled = selectedFiles.length < 2;
}

function updateBars(probabilities) {
    const good = Number(probabilities?.Good || 0);
    const defect = Number(probabilities?.Defect || 0);
    document.getElementById("goodBar").style.width = `${(good * 100).toFixed(1)}%`;
    document.getElementById("defectBar").style.width = `${(defect * 100).toFixed(1)}%`;
    document.getElementById("goodText").textContent = `${(good * 100).toFixed(1)}%`;
    document.getElementById("defectText").textContent = `${(defect * 100).toFixed(1)}%`;
}

function setText(id, value) {
    document.getElementById(id).textContent = value;
}

function renderSingleResult(data) {
    latestPayload = data;
    els.exportBtn.disabled = false;

    setText("resClass", `${data.class_name}${data.predicted_defect ? " (alert)" : ""}`);
    setText("resConfidence", `${(Number(data.confidence) * 100).toFixed(2)}%`);
    setText("resEntropy", Number(data.uncertainty_entropy).toFixed(3));
    setText("resMargin", Number(data.confidence_margin).toFixed(3));
    setText("resRiskBand", String(data.risk.risk_band).toUpperCase());
    setText("resDrift", String(data.drift.drift_state).replaceAll("_", " "));

    updateBars(data.probabilities);

    setText("qSharpness", Number(data.quality.sharpness).toFixed(3));
    setText("qBrightness", Number(data.quality.brightness).toFixed(3));
    setText("qContrast", Number(data.quality.contrast).toFixed(3));
    setText("qSaturation", Number(data.quality.saturation).toFixed(3));
    setText("qScore", Number(data.quality.quality_score).toFixed(3));
    setText("qLatency", `${Number(data.inference_ms).toFixed(2)} ms`);
}

async function analyzeSingle() {
    if (selectedFiles.length !== 1) return;
    const formData = new FormData();
    formData.append("file", selectedFiles[0]);
    setStatus("Running single-image inference...");
    try {
        const response = await fetch(`/predict?decision_threshold=${threshold().toFixed(2)}`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Inference failed");
        renderSingleResult(data);
        setStatus(`Inference complete. Risk: ${data.risk.risk_band}.`);
        await refreshHistory();
    } catch (error) {
        setStatus(error.message, true);
    }
}

async function analyzeBatch() {
    if (selectedFiles.length < 2) return;
    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));
    setStatus("Running batch inference...");
    try {
        const response = await fetch(`/predict/batch?decision_threshold=${threshold().toFixed(2)}`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Batch failed");

        latestPayload = data;
        els.exportBtn.disabled = false;
        setStatus(`Batch complete: ${data.summary.processed}/${data.summary.total_files} processed.`);

        setText("resClass", `${data.summary.processed} items`);
        setText("resConfidence", `${(Number(data.summary.avg_confidence) * 100).toFixed(2)}% avg`);
        setText("resEntropy", "-");
        setText("resMargin", "-");
        setText("resRiskBand", `${data.summary.critical_risk_count} critical`);
        setText("resDrift", "batch mode");
        setText("qSharpness", "-");
        setText("qBrightness", "-");
        setText("qContrast", "-");
        setText("qSaturation", "-");
        setText("qScore", "-");
        setText("qLatency", `${Number(data.summary.avg_latency_ms).toFixed(2)} ms`);
        updateBars({ Good: 1 - Number(data.summary.defect_rate), Defect: Number(data.summary.defect_rate) });

        await refreshHistory();
    } catch (error) {
        setStatus(error.message, true);
    }
}

function exportJson() {
    if (!latestPayload) return;
    const blob = new Blob([JSON.stringify(latestPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `inspectai-result-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function drawHistory(history) {
    const canvas = document.getElementById("historyCanvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b1321";
    ctx.fillRect(0, 0, w, h);

    if (!history.length) {
        ctx.fillStyle = "#9eaed1";
        ctx.font = "15px Manrope";
        ctx.fillText("No history yet", 24, 36);
        return;
    }

    const values = history.map((x) => Number(x.confidence || 0));
    const pad = 28;
    const gh = h - pad * 2;
    const gw = w - pad * 2;
    const stepX = gw / Math.max(values.length - 1, 1);

    ctx.strokeStyle = "#2a426f";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i += 1) {
        const y = pad + (gh / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(w - pad, y);
        ctx.stroke();
    }

    ctx.strokeStyle = "#3dd2ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((value, index) => {
        const x = pad + stepX * index;
        const y = pad + gh - (value * gh);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "#58ffcb";
    values.forEach((value, index) => {
        const x = pad + stepX * index;
        const y = pad + gh - (value * gh);
        ctx.beginPath();
        ctx.arc(x, y, 2.8, 0, Math.PI * 2);
        ctx.fill();
    });
}

async function refreshHistory() {
    try {
        const health = await fetch("/health").then((r) => r.json());
        setText("kpiMode", String(health.mode || "-").toUpperCase());

        const payload = await fetch("/analytics/history?limit=50").then((r) => r.json());
        const summary = payload.summary || {};
        setText("kpiDefectRate", `${(Number(summary.defect_rate || 0) * 100).toFixed(1)}%`);
        setText("kpiLatency", `${Number(summary.avg_latency_ms || 0).toFixed(1)} ms`);
        setText("kpiRisk", String(summary.risk_distribution?.critical || 0));
        document.getElementById("historyMeta").textContent =
            `Samples: ${summary.total || 0} | Avg confidence: ${(Number(summary.avg_confidence || 0) * 100).toFixed(1)}%`;
        drawHistory(payload.history || []);
    } catch (_error) {
        // non-blocking
    }
}

function bindEvents() {
    els.dropzone.addEventListener("click", () => els.fileInput.click());
    els.fileInput.addEventListener("change", (event) => setFiles(event.target.files));
    els.dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        els.dropzone.classList.add("drag");
    });
    els.dropzone.addEventListener("dragleave", () => els.dropzone.classList.remove("drag"));
    els.dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        els.dropzone.classList.remove("drag");
        setFiles(event.dataTransfer.files);
    });

    els.threshold.addEventListener("input", updateThresholdLabel);
    els.singleBtn.addEventListener("click", analyzeSingle);
    els.batchBtn.addEventListener("click", analyzeBatch);
    els.exportBtn.addEventListener("click", exportJson);
    els.refreshHistoryBtn.addEventListener("click", refreshHistory);
}

function init() {
    updateThresholdLabel();
    bindEvents();
    refreshHistory();
}

init();
