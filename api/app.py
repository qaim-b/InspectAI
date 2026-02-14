"""
InspectAI FastAPI application
Industrial defect intelligence with advanced inference analytics.
"""

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
import io
import math
import statistics
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

import sys

# Add src to import path
sys.path.append(str(Path(__file__).parent.parent / "src"))


APP_VERSION = "2.0.0"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
INFERENCE_HISTORY: Deque[Dict[str, Any]] = deque(maxlen=300)


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    message: str
    inference_ms: float
    decision_threshold: float
    predicted_defect: bool
    uncertainty_entropy: float
    confidence_margin: float
    quality: Dict[str, float]
    risk: Dict[str, Any]
    drift: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: str
    model_info: Optional[Dict[str, Any]] = None


class MockPredictor:
    """Fallback predictor for environments without model checkpoint."""

    class_names = ["Good", "Defect"]

    def __init__(self) -> None:
        self.device = "cpu-mock"
        self.img_size = 224

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        gray = arr.mean(axis=2)
        brightness = float(gray.mean() / 255.0)
        contrast = float(gray.std() / 64.0)
        edge_score = float(np.mean(np.abs(np.diff(gray, axis=0))) / 32.0)

        defect_prob = max(0.02, min(0.98, 0.16 + (contrast * 0.38) + (edge_score * 0.42) - (brightness * 0.14)))
        good_prob = 1.0 - defect_prob
        class_id = 1 if defect_prob >= good_prob else 0
        confidence = defect_prob if class_id == 1 else good_prob

        return {
            "class": class_id,
            "label": self.class_names[class_id],
            "confidence": float(confidence),
            "probabilities": {"Good": float(good_prob), "Defect": float(defect_prob)},
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "parameters": 0,
            "classes": self.class_names,
            "input_size": self.img_size,
            "mode": "mock",
        }


app = FastAPI(
    title="InspectAI - Defect Intelligence API",
    description="Industrial defect detection with advanced analytics and monitoring.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = Path(__file__).parent.parent
static_dir = base_dir / "static"
templates_dir = base_dir / "templates"
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

predictor: Optional[Any] = None
predictor_mode = "mock"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_quality_metrics(image: Image.Image) -> Dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    laplacian_var = float(np.var(np.diff(gray, axis=0))) if gray.shape[0] > 1 else 0.0
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 128.0)
    saturation = float(np.std(arr, axis=2).mean() / 128.0)

    quality_score = max(
        0.0,
        min(1.0, (0.36 * min(laplacian_var / 1000.0, 1.0)) + (0.26 * min(contrast, 1.0)) + (0.22 * min(saturation, 1.0)) + (0.16 * (1.0 - abs(0.5 - brightness)))),
    )

    return {
        "sharpness": round(min(laplacian_var / 1000.0, 1.0), 4),
        "brightness": round(brightness, 4),
        "contrast": round(min(contrast, 1.0), 4),
        "saturation": round(min(saturation, 1.0), 4),
        "quality_score": round(quality_score, 4),
    }


def entropy_from_probs(probabilities: Dict[str, float]) -> float:
    values = [max(1e-8, float(v)) for v in probabilities.values()]
    total = sum(values) or 1.0
    normed = [v / total for v in values]
    return float(-sum(p * math.log(p, 2) for p in normed))


def infer_risk(probabilities: Dict[str, float], quality: Dict[str, float], decision_threshold: float) -> Dict[str, Any]:
    defect_prob = float(probabilities.get("Defect", 0.0))
    confidence_margin = abs(float(probabilities.get("Good", 0.0)) - defect_prob)
    uncertainty = entropy_from_probs(probabilities)
    quality_penalty = 1.0 - float(quality.get("quality_score", 0.0))

    risk_score = min(1.0, max(0.0, (0.55 * defect_prob) + (0.2 * uncertainty) + (0.25 * quality_penalty)))
    if risk_score >= 0.72:
        band = "critical"
        action = "quarantine_item_and_manual_review"
    elif risk_score >= 0.44:
        band = "warning"
        action = "secondary_inspection"
    else:
        band = "stable"
        action = "allow_pass"

    if defect_prob >= decision_threshold and band == "stable":
        band = "warning"
        action = "secondary_inspection"

    return {
        "risk_score": round(risk_score, 4),
        "risk_band": band,
        "recommended_action": action,
        "confidence_margin": round(confidence_margin, 4),
        "uncertainty_entropy": round(uncertainty, 4),
    }


def compute_drift_signal(current_confidence: float) -> Dict[str, Any]:
    if len(INFERENCE_HISTORY) < 8:
        return {"drift_score": 0.0, "drift_state": "insufficient_history", "baseline_confidence": 0.0}

    baseline = [float(x["confidence"]) for x in INFERENCE_HISTORY]
    baseline_mean = statistics.fmean(baseline)
    baseline_std = statistics.pstdev(baseline) or 1e-6
    z = abs((current_confidence - baseline_mean) / baseline_std)
    drift_score = min(1.0, z / 3.0)
    if drift_score >= 0.75:
        state = "high_shift"
    elif drift_score >= 0.4:
        state = "moderate_shift"
    else:
        state = "stable"
    return {
        "drift_score": round(drift_score, 4),
        "drift_state": state,
        "baseline_confidence": round(baseline_mean, 4),
    }


def summarize_history(limit: int = 100) -> Dict[str, Any]:
    rows = list(INFERENCE_HISTORY)[-limit:]
    if not rows:
        return {
            "total": 0,
            "defect_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0,
            "risk_distribution": {"stable": 0, "warning": 0, "critical": 0},
        }

    total = len(rows)
    defect_rate = sum(1 for r in rows if r["class_name"] == "Defect") / total
    avg_conf = statistics.fmean(float(r["confidence"]) for r in rows)
    avg_latency = statistics.fmean(float(r["inference_ms"]) for r in rows)
    risk_distribution = {"stable": 0, "warning": 0, "critical": 0}
    for row in rows:
        band = str(row["risk"]["risk_band"])
        risk_distribution[band] = risk_distribution.get(band, 0) + 1

    return {
        "total": total,
        "defect_rate": round(defect_rate, 4),
        "avg_confidence": round(avg_conf, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "risk_distribution": risk_distribution,
    }


def load_model(model_path: str = "models/checkpoints/best_acc.pth") -> None:
    global predictor, predictor_mode
    model_file = Path(model_path)
    if model_file.exists():
        try:
            from inference import DefectPredictor  # lazy import for serverless compatibility
            predictor = DefectPredictor(str(model_file), device="cuda")
            predictor_mode = "trained"
            return
        except Exception:
            pass
    predictor = MockPredictor()
    predictor_mode = "mock"


@app.on_event("startup")
async def startup_event() -> None:
    load_model()


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    model_loaded = predictor is not None
    model_info = predictor.get_model_info() if model_loaded and hasattr(predictor, "get_model_info") else None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        mode=predictor_mode,
        model_info=model_info,
    )


@app.get("/analytics/history")
async def analytics_history(limit: int = Query(default=30, ge=1, le=300)):
    rows = list(INFERENCE_HISTORY)[-limit:]
    return {"history": rows, "summary": summarize_history(limit)}


@app.post("/predict", response_model=PredictionResponse)
async def predict_defect(
    file: UploadFile = File(...),
    decision_threshold: float = Query(default=0.5, ge=0.05, le=0.95),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor unavailable")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    start = time.perf_counter()
    result = predictor.predict(image)
    latency_ms = (time.perf_counter() - start) * 1000

    quality = compute_quality_metrics(image)
    risk = infer_risk(result["probabilities"], quality, decision_threshold)
    drift = compute_drift_signal(float(result["confidence"]))

    record = {
        "timestamp": now_iso(),
        "filename": file.filename or "unknown",
        "class_id": int(result["class"]),
        "class_name": str(result["label"]),
        "confidence": float(result["confidence"]),
        "probabilities": result["probabilities"],
        "inference_ms": round(latency_ms, 2),
        "quality": quality,
        "risk": risk,
    }
    INFERENCE_HISTORY.append(record)

    defect_prob = float(result["probabilities"].get("Defect", 0.0))
    predicted_defect = defect_prob >= decision_threshold

    return PredictionResponse(
        class_id=int(result["class"]),
        class_name=str(result["label"]),
        confidence=float(result["confidence"]),
        probabilities={k: float(v) for k, v in result["probabilities"].items()},
        message="Inference complete",
        inference_ms=round(latency_ms, 2),
        decision_threshold=decision_threshold,
        predicted_defect=predicted_defect,
        uncertainty_entropy=risk["uncertainty_entropy"],
        confidence_margin=risk["confidence_margin"],
        quality=quality,
        risk=risk,
        drift=drift,
        timestamp=record["timestamp"],
    )


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    decision_threshold: float = Query(default=0.5, ge=0.05, le=0.95),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor unavailable")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    outputs: List[Dict[str, Any]] = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            outputs.append({"filename": file.filename, "error": "not_an_image"})
            continue

        raw_bytes = await file.read()
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            outputs.append({"filename": file.filename, "error": "file_too_large"})
            continue

        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            start = time.perf_counter()
            result = predictor.predict(image)
            latency_ms = (time.perf_counter() - start) * 1000
            quality = compute_quality_metrics(image)
            risk = infer_risk(result["probabilities"], quality, decision_threshold)
            defect_prob = float(result["probabilities"].get("Defect", 0.0))

            row = {
                "timestamp": now_iso(),
                "filename": file.filename or "unknown",
                "class_id": int(result["class"]),
                "class_name": str(result["label"]),
                "confidence": float(result["confidence"]),
                "probabilities": {k: float(v) for k, v in result["probabilities"].items()},
                "inference_ms": round(latency_ms, 2),
                "predicted_defect": defect_prob >= decision_threshold,
                "quality_score": quality["quality_score"],
                "risk_band": risk["risk_band"],
                "risk_score": risk["risk_score"],
            }
            outputs.append(row)

            INFERENCE_HISTORY.append(
                {
                    "timestamp": row["timestamp"],
                    "filename": row["filename"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "confidence": row["confidence"],
                    "probabilities": row["probabilities"],
                    "inference_ms": row["inference_ms"],
                    "quality": quality,
                    "risk": risk,
                }
            )
        except Exception as exc:
            outputs.append({"filename": file.filename, "error": str(exc)})

    valid_rows = [x for x in outputs if "error" not in x]
    summary = {
        "total_files": len(files),
        "processed": len(valid_rows),
        "failed": len(files) - len(valid_rows),
        "defect_rate": round(sum(1 for x in valid_rows if x["predicted_defect"]) / max(len(valid_rows), 1), 4),
        "avg_confidence": round(statistics.fmean(x["confidence"] for x in valid_rows), 4) if valid_rows else 0.0,
        "avg_latency_ms": round(statistics.fmean(x["inference_ms"] for x in valid_rows), 2) if valid_rows else 0.0,
        "critical_risk_count": sum(1 for x in valid_rows if x["risk_band"] == "critical"),
    }
    return JSONResponse(content={"predictions": outputs, "summary": summary})


@app.get("/docs/meta")
async def docs_meta():
    return {
        "app": "InspectAI",
        "version": APP_VERSION,
        "mode": predictor_mode,
        "endpoints": {
            "GET /": "Web console",
            "GET /health": "Service health + model mode",
            "POST /predict": "Single image inference with analytics",
            "POST /predict/batch": "Batch inference + summary",
            "GET /analytics/history": "Recent inference history and KPI summary",
            "GET /docs/meta": "API metadata",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
