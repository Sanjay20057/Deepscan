"""
video_api.py — FastAPI backend for video deepfake detection (CNN-Only)

- Pure CNN prediction using deepfake_model.h5
- No Sightengine, no hybrid ensemble
- Model trained with sigmoid: HIGH = REAL, LOW = FAKE
  →  fake_prob = 1.0 - raw_sigmoid
  →  Threshold: fake_prob >= 0.5  →  FAKE
- Groq AI explanation still supported

Run with:  uvicorn video_api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import os
import cv2
import uuid
import tempfile
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMG_SIZE          = 224
THUMBNAIL_MAX_DIM = 240
FRAMES_PER_VIDEO  = 20
FAKE_THRESHOLD    = 0.5   # applied to fake_prob = 1 - raw_sigmoid

# Change this in video_api.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_model.h5")

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
VIDEO_EXTENSIONS      = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ─── GLOBALS ─────────────────────────────────────────────────────────────────
model          = None
_startup_error = ""

# ─── MODEL LOADING ───────────────────────────────────────────────────────────
def _load_model():
    global model
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found.\n"
            "Fix: copy deepfake_model.h5 to the same directory as video_api.py"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from '{MODEL_PATH}'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_error
    try:
        _load_model()
    except Exception as exc:
        _startup_error = str(exc)
        print(f"\n❌ Model failed to load:\n{_startup_error}\n")
    yield


# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API (CNN-Only)",
    version="1.0.0",
    description="Pure CNN deepfake detection — no third-party APIs.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─── SCHEMAS ─────────────────────────────────────────────────────────────────
class FrameResult(BaseModel):
    frame_index:       int
    timestamp_sec:     float
    cnn_label:         str
    cnn_confidence:    float
    fake_prob:         float
    verdict:           str
    thumbnail_b64:     str
    frame_explanation: str


class AnalysisResponse(BaseModel):
    media_type:            str
    verdict:               str
    fake_probability:      float
    confidence_pct:        float
    cnn_label:             str
    cnn_confidence:        float
    frame_results:         Optional[list[FrameResult]]
    fake_frame_count:      Optional[int]
    total_frames_analysed: Optional[int]
    message:               str


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _require_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                + (_startup_error or "Check server logs.")
            ),
        )


def _preprocess_pil(img: Image.Image) -> np.ndarray:
    arr = np.array(
        img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    return arr


def _make_thumbnail(pil_img: Image.Image) -> str:
    w, h  = pil_img.size
    scale = THUMBNAIL_MAX_DIM / max(w, h)
    thumb = pil_img.convert("RGB").resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
    )
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_cnn_pil(pil_image: Image.Image) -> tuple[str, float, float]:
    """
    Single image inference.
    Returns (label, confidence, fake_prob).

    Model sigmoid: HIGH = REAL, LOW = FAKE
    ∴  fake_prob = 1.0 - raw_sigmoid
    """
    arr      = _preprocess_pil(pil_image)
    raw      = float(model(np.expand_dims(arr, 0), training=False).numpy()[0][0])
    fake_prob = 1.0 - raw                              # ← INVERTED
    label    = "fake" if fake_prob >= FAKE_THRESHOLD else "real"
    conf     = fake_prob if label == "fake" else (1.0 - fake_prob)
    return label, conf, fake_prob


def predict_cnn_batch(
    pil_images: list[Image.Image],
) -> tuple[list[str], list[float], list[float]]:
    """
    Batched CNN inference.
    Returns (labels, confidences, fake_probs).

    Model sigmoid: HIGH = REAL, LOW = FAKE
    ∴  fake_prob = 1.0 - raw_sigmoid  for every frame
    """
    if not pil_images:
        return [], [], []
    batch      = np.stack([_preprocess_pil(img) for img in pil_images])
    raws       = model(batch, training=False).numpy().flatten()
    labels, confs, fake_probs = [], [], []
    for raw in raws:
        fake_p = float(1.0 - raw)                      # ← INVERTED
        label  = "fake" if fake_p >= FAKE_THRESHOLD else "real"
        conf   = fake_p if label == "fake" else (1.0 - fake_p)
        labels.append(label)
        confs.append(conf)
        fake_probs.append(fake_p)
    return labels, confs, fake_probs


def _frame_explanation(verdict: str, fake_prob: float,
                        cnn_label: str, cnn_conf: float) -> str:
    if verdict == "fake":
        return (
            f"Frame classified as FAKE with {fake_prob * 100:.1f}% fake probability "
            f"(CNN confidence: {cnn_conf * 100:.0f}%). "
            "Likely artefacts: blending boundaries, inconsistent skin texture, "
            "or unnatural lighting."
        )
    return (
        f"Frame classified as REAL with {(1 - fake_prob) * 100:.1f}% confidence "
        f"(CNN confidence: {cnn_conf * 100:.0f}%). "
        "No significant deepfake artefacts detected."
    )


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "model_loaded": False,
                     "error": _startup_error or "Model not loaded"},
        )
    return {"status": "ok", "model_loaded": True, "model_file": MODEL_PATH}


@app.post("/analyse/image", response_model=AnalysisResponse)
async def analyse_image(file: UploadFile = File(...)):
    _require_model()
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415,
            detail=f"Unsupported type '{file.content_type}'. Use JPG/PNG/WebP.")
    raw_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read image: {exc}")

    cnn_label, cnn_conf, fake_prob = predict_cnn_pil(pil_image)

    return AnalysisResponse(
        media_type="image",
        verdict=cnn_label,
        fake_probability=round(fake_prob, 4),
        confidence_pct=round(fake_prob * 100, 1),
        cnn_label=cnn_label,
        cnn_confidence=round(cnn_conf, 4),
        frame_results=None,
        fake_frame_count=None,
        total_frames_analysed=None,
        message=f"Image classified as {cnn_label.upper()} "
                f"with {fake_prob * 100:.1f}% fake probability.",
    )


@app.post("/analyse/video", response_model=AnalysisResponse)
async def analyse_video(file: UploadFile = File(...)):
    _require_model()

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415,
            detail=f"Unsupported format '{suffix}'.")

    raw_bytes = await file.read()
    tmp_path  = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{suffix}")
    cap       = None

    try:
        with open(tmp_path, "wb") as f_out:
            f_out.write(raw_bytes)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Cannot open video file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total_frames < 1:
            raise HTTPException(status_code=422, detail="Video appears to be empty.")

        n_samples  = min(FRAMES_PER_VIDEO, total_frames)
        frame_idxs = np.linspace(0, total_frames - 1, n_samples, dtype=int)

        # ── Decode frames ─────────────────────────────────────────────────────
        pil_frames:      list[Image.Image] = []
        sampled_indices: list[int]         = []
        sampled_ts:      list[float]       = []

        for fi in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue
            pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            sampled_indices.append(int(fi))
            sampled_ts.append(round(fi / fps, 2))

        cap.release()
        cap = None

        if not pil_frames:
            raise HTTPException(status_code=422, detail="Could not extract any frames.")

        # ── Batch CNN inference ───────────────────────────────────────────────
        labels, confs, fake_probs = predict_cnn_batch(pil_frames)

        # ── Build per-frame results ───────────────────────────────────────────
        frame_results: list[FrameResult] = []

        for i, (lbl, conf, fake_p) in enumerate(zip(labels, confs, fake_probs)):
            verdict     = "fake" if fake_p >= FAKE_THRESHOLD else "real"
            explanation = _frame_explanation(verdict, fake_p, lbl, conf)

            frame_results.append(FrameResult(
                frame_index=sampled_indices[i],
                timestamp_sec=sampled_ts[i],
                cnn_label=lbl,
                cnn_confidence=round(conf, 4),
                fake_prob=round(fake_p, 4),
                verdict=verdict,
                thumbnail_b64=_make_thumbnail(pil_frames[i]),
                frame_explanation=explanation,
            ))

        # ── Aggregate verdict ─────────────────────────────────────────────────
        mean_fake_prob  = float(np.mean(fake_probs))   # already inverted per-frame
        overall_verdict = "fake" if mean_fake_prob >= FAKE_THRESHOLD else "real"
        overall_conf    = mean_fake_prob if overall_verdict == "fake" else (1.0 - mean_fake_prob)
        fake_frame_count = sum(1 for fr in frame_results if fr.verdict == "fake")

        return AnalysisResponse(
            media_type="video",
            verdict=overall_verdict,
            fake_probability=round(mean_fake_prob, 4),
            confidence_pct=round(mean_fake_prob * 100, 1),
            cnn_label=overall_verdict,
            cnn_confidence=round(overall_conf, 4),
            frame_results=frame_results,
            fake_frame_count=fake_frame_count,
            total_frames_analysed=len(frame_results),
            message=(
                f"Video analysed — {fake_frame_count}/{len(frame_results)} frames "
                f"classified as fake. Overall verdict: {overall_verdict.upper()} "
                f"({mean_fake_prob * 100:.1f}% avg fake probability)."
            ),
        )

    finally:
        if cap is not None:
            cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_api:app", host="0.0.0.0", port=8000, reload=True)
