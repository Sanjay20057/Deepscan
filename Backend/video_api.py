"""
video_api.py — FastAPI backend for deepfake detection
Image path -> deepfake_model.keras (from-scratch CNN, 128x128, /255.0)
Video path -> deepfake_model.h5    (MobileNetV2 transfer model, 224x224, preprocess_input)
WHY TWO MODELS:
Inspecting the two uploaded files shows they are NOT the same architecture:
  - deepfake_model.keras is the custom from-scratch CNN (no pretrained backbone),
    with an input layer of (128, 128, 3).
  - deepfake_model.h5 is the earlier MobileNetV2 transfer-learning model, with an
    input layer of (224, 224, 3) and a nested "mobilenetv2_1.00_224" submodel.
This is exactly why IMAGE_INVERT_SIGMOID and VIDEO_INVERT_SIGMOID have different
defaults below — they aren't two settings for one model, they're the correct
sigmoid convention for two different models trained separately.
FIXES CARRIED OVER FROM THE PREVIOUS PASS:
- /diagnose no longer crashes (_raw_to_fake_prob was missing its `invert` arg).
- `transformers` import is optional so a missing package doesn't take down
  the whole API.
- IMG_SIZE for each model is read from that model's own input_shape rather
  than a hardcoded constant, so this can't silently drift again.
Everything else — thresholds, video frame sampling/aggregation, response
shapes — is unchanged from your version.
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

# transformers is optional — the HF ViT ensemble is a bonus for the image
# path only. A missing package shouldn't take down the whole API.
try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    hf_pipeline = None
    _HF_AVAILABLE = False

load_dotenv()

hf_detector = None

# ─── CONFIG ──────────────────────────────────────────────────────────────────
THUMBNAIL_MAX_DIM = 240
FRAMES_PER_VIDEO  = int(os.getenv("FRAMES_PER_VIDEO", "25"))
FAKE_THRESHOLD       = float(os.getenv("FAKE_THRESHOLD", "0.40"))        # kept for compat
IMAGE_FAKE_THRESHOLD = float(os.getenv("IMAGE_FAKE_THRESHOLD", "0.35"))  # image: flag fake if fake_prob >= 0.35
VIDEO_FAKE_THRESHOLD = float(os.getenv("VIDEO_FAKE_THRESHOLD", "0.50"))  # video threshold

# ── INVERT_SIGMOID — unchanged, exactly as specified ─────────────────────────
IMAGE_INVERT_SIGMOID = os.getenv("IMAGE_INVERT_SIGMOID", "false").lower() != "false"
VIDEO_INVERT_SIGMOID = os.getenv("VIDEO_INVERT_SIGMOID", "true").lower() != "false"

# ── Per-model preprocessing ───────────────────────────────────────────────────
# Hardcoded to match each model's own architecture (not an env-configurable
# global anymore, since image and video now use two different models):
#   image model (from-scratch CNN)  -> divide by 255  -> [0, 1]
#   video model (MobileNetV2)       -> preprocess_input -> [-1, 1]
IMAGE_PREPROCESS_MODE = "divide255"
VIDEO_PREPROCESS_MODE = "mobilenet"

# IMG_SIZE fallbacks only — both are overwritten with the real value read off
# each model's input_shape in _load_models().
IMAGE_IMG_SIZE = 128
VIDEO_IMG_SIZE = 224

IMAGE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_model.keras")
VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_model.h5")

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
VIDEO_EXTENSIONS      = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ─── GLOBALS ─────────────────────────────────────────────────────────────────
image_model    = None
video_model    = None
_startup_error = ""
_mobilenet_preprocess_fn = None  # set during model load, used for the video model only


# ─── MODEL LOADING ───────────────────────────────────────────────────────────
def _load_models():
    global image_model, video_model, hf_detector, _mobilenet_preprocess_fn
    global IMAGE_IMG_SIZE, VIDEO_IMG_SIZE
    import tensorflow as tf

    # -- Image model (from-scratch CNN) --------------------------------------
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise RuntimeError(
            f"Image model file '{IMAGE_MODEL_PATH}' not found.\n"
            f"Fix: copy your trained model to this directory as "
            f"'{os.path.basename(IMAGE_MODEL_PATH)}'."
        )
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    IMAGE_IMG_SIZE = _infer_img_size(image_model, IMAGE_IMG_SIZE, "image")

    # -- Video model (MobileNetV2 transfer model) ----------------------------
    if not os.path.exists(VIDEO_MODEL_PATH):
        raise RuntimeError(
            f"Video model file '{VIDEO_MODEL_PATH}' not found.\n"
            f"Fix: copy your trained model to this directory as "
            f"'{os.path.basename(VIDEO_MODEL_PATH)}'."
        )
    video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
    VIDEO_IMG_SIZE = _infer_img_size(video_model, VIDEO_IMG_SIZE, "video")

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    _mobilenet_preprocess_fn = preprocess_input

    # -- HuggingFace ViT detector (optional, image path only) ----------------
    if _HF_AVAILABLE:
        try:
            hf_detector = hf_pipeline(
                "image-classification",
                model="Wvolf/ViT_Deepfake_Detection",
                device=-1  # CPU
            )
            print("HuggingFace ViT detector loaded")
        except Exception as e:
            print(f"HuggingFace detector failed to load: {e}")
    else:
        print("'transformers' not installed -- image path is CNN-only (no HF ensemble).")

    print(f"Image model loaded from '{IMAGE_MODEL_PATH}'  (size={IMAGE_IMG_SIZE}, "
          f"preprocess={IMAGE_PREPROCESS_MODE}, invert={IMAGE_INVERT_SIGMOID})")
    print(f"Video model loaded from '{VIDEO_MODEL_PATH}'  (size={VIDEO_IMG_SIZE}, "
          f"preprocess={VIDEO_PREPROCESS_MODE}, invert={VIDEO_INVERT_SIGMOID})")
    print(f"   IMAGE_FAKE_THRESHOLD = {IMAGE_FAKE_THRESHOLD}")
    print(f"   VIDEO_FAKE_THRESHOLD = {VIDEO_FAKE_THRESHOLD}")

    # -- Sanity checks (non-fatal) --------------------------------------------
    try:
        dummy_img = np.zeros((1, IMAGE_IMG_SIZE, IMAGE_IMG_SIZE, 3), dtype=np.float32) / 255.0
        raw_img   = float(image_model(dummy_img, training=False).numpy()[0][0])
        print(f"   Image sanity check: raw_sigmoid={raw_img:.4f}  "
              f"fake_prob={_raw_to_fake_prob(raw_img, IMAGE_INVERT_SIGMOID):.4f}")
    except Exception as e:
        print(f"   Image sanity check failed (non-fatal): {e}")

    try:
        dummy_vid = _mobilenet_preprocess_fn(
            np.zeros((1, VIDEO_IMG_SIZE, VIDEO_IMG_SIZE, 3), dtype=np.float32)
        )
        raw_vid = float(video_model(dummy_vid, training=False).numpy()[0][0])
        print(f"   Video sanity check: raw_sigmoid={raw_vid:.4f}  "
              f"fake_prob={_raw_to_fake_prob(raw_vid, VIDEO_INVERT_SIGMOID):.4f}")
    except Exception as e:
        print(f"   Video sanity check failed (non-fatal): {e}")


def _infer_img_size(model, fallback: int, label: str) -> int:
    input_shape = model.input_shape  # e.g. (None, 128, 128, 3)
    if input_shape and len(input_shape) == 4 and input_shape[1]:
        detected = int(input_shape[1])
        if detected != fallback:
            print(f"{label} model expects {detected}px input "
                  f"(config default was {fallback}px) -- using {detected}px.")
        return detected
    print(f"Could not infer {label} input size from input_shape={input_shape}; "
          f"keeping fallback {fallback}px.")
    return fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_error
    try:
        _load_models()
    except Exception as exc:
        _startup_error = str(exc)
        print(f"\nModel failed to load:\n{_startup_error}\n")
    yield


# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API",
    version="3.0.0",
    description="Image path: from-scratch CNN. Video path: MobileNetV2 transfer model.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def _require_models():
    if image_model is None or video_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model(s) not loaded. "
                + (_startup_error or "Check server logs.")
            ),
        )


def _preprocess_image_pil(img: Image.Image) -> np.ndarray:
    """For the image model (from-scratch CNN): resize to IMAGE_IMG_SIZE, x/255.0 -> [0,1]."""
    rgb = img.convert("RGB").resize((IMAGE_IMG_SIZE, IMAGE_IMG_SIZE), Image.LANCZOS)
    arr = np.array(rgb, dtype=np.float32)
    return arr / 255.0


def _preprocess_video_pil(img: Image.Image) -> np.ndarray:
    """For the video model (MobileNetV2): resize to VIDEO_IMG_SIZE, preprocess_input -> [-1,1]."""
    rgb = img.convert("RGB").resize((VIDEO_IMG_SIZE, VIDEO_IMG_SIZE), Image.LANCZOS)
    arr = np.array(rgb, dtype=np.float32)
    return _mobilenet_preprocess_fn(arr)


def _make_thumbnail(pil_img: Image.Image) -> str:
    w, h  = pil_img.size
    scale = THUMBNAIL_MAX_DIM / max(w, h)
    thumb = pil_img.convert("RGB").resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
    )
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _raw_to_fake_prob(raw_sigmoid: float, invert: bool) -> float:
    """Convert raw sigmoid to fake probability.
    invert=True  -> fake_prob = 1 - sigmoid  (HIGH sigmoid = REAL)
    invert=False -> fake_prob = sigmoid      (HIGH sigmoid = FAKE)
    """
    return (1.0 - raw_sigmoid) if invert else raw_sigmoid


def predict_cnn_pil(pil_image: Image.Image) -> tuple[str, float, float]:
    """Single IMAGE inference — from-scratch CNN, optionally ensembled with HF ViT."""

    # -- CNN score (image model) --------------------------------
    arr      = _preprocess_image_pil(pil_image)
    raw      = float(image_model(np.expand_dims(arr, 0), training=False).numpy()[0][0])
    cnn_fake = _raw_to_fake_prob(raw, invert=IMAGE_INVERT_SIGMOID)

    # -- HuggingFace ViT score (optional) -------------------------
    hf_fake = cnn_fake  # fallback if HF not loaded
    if hf_detector is not None:
        try:
            results = hf_detector(pil_image)
            for r in results:
                if "fake" in r["label"].lower():
                    hf_fake = r["score"]
                    break
        except Exception as e:
            print(f"   [HF] inference failed: {e}")

    # -- Ensemble: take the higher of the two --------------------
    fake_prob = max(cnn_fake, hf_fake)

    label = "fake" if fake_prob >= IMAGE_FAKE_THRESHOLD else "real"
    conf  = fake_prob if label == "fake" else (1.0 - fake_prob)

    print(f"   [IMAGE] cnn_fake={cnn_fake:.4f}  hf_fake={hf_fake:.4f}  "
          f"ensemble={fake_prob:.4f}  label={label.upper()}")
    return label, conf, fake_prob


def predict_cnn_batch(
    pil_images: list[Image.Image],
) -> tuple[list[str], list[float], list[float]]:
    """Batched VIDEO frame inference — MobileNetV2 model. Unchanged logic,
    now just pointed at video_model / VIDEO_IMG_SIZE / mobilenet preprocessing."""
    if not pil_images:
        return [], [], []
    batch = np.stack([_preprocess_video_pil(img) for img in pil_images])
    raws  = video_model(batch, training=False).numpy().flatten()

    labels, confs, fake_probs = [], [], []
    for raw in raws:
        fake_p = _raw_to_fake_prob(float(raw), invert=VIDEO_INVERT_SIGMOID)
        label  = "fake" if fake_p >= VIDEO_FAKE_THRESHOLD else "real"
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
    if image_model is None or video_model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status":        "error",
                "model_loaded":  False,  # legacy field the frontend's status badge checks
                "models_loaded": False,
                "error":         _startup_error or "Model(s) not loaded",
            },
        )
    return {
        "status":               "ok",
        "model_loaded":         True,   # legacy field the frontend's status badge checks
        "models_loaded":        True,
        "image_model_file":     IMAGE_MODEL_PATH,
        "video_model_file":     VIDEO_MODEL_PATH,
        "image_img_size":       IMAGE_IMG_SIZE,
        "video_img_size":       VIDEO_IMG_SIZE,
        "image_preprocess":     IMAGE_PREPROCESS_MODE,
        "video_preprocess":     VIDEO_PREPROCESS_MODE,
        "image_invert_sigmoid": IMAGE_INVERT_SIGMOID,
        "video_invert_sigmoid": VIDEO_INVERT_SIGMOID,
        "image_fake_threshold": IMAGE_FAKE_THRESHOLD,
        "video_fake_threshold": VIDEO_FAKE_THRESHOLD,
        "image_input_shape":    str(image_model.input_shape),
        "video_input_shape":    str(video_model.input_shape),
        "hf_ensemble_loaded":   hf_detector is not None,
    }


# ── DIAGNOSTIC ENDPOINT (image model only) ────────────────────────────────────
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    """
    Upload an image to see the raw sigmoid output from the IMAGE model and
    what fake_prob would be under both invert settings.
    curl -X POST http://localhost:8000/diagnose -F "file=@your_image.jpg"
    """
    _require_models()
    raw_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read image: {exc}")

    arr = _preprocess_image_pil(pil_image)
    raw = float(image_model(np.expand_dims(arr, 0), training=False).numpy()[0][0])

    fake_prob_not_inverted = _raw_to_fake_prob(raw, invert=False)
    fake_prob_inverted     = _raw_to_fake_prob(raw, invert=True)

    def verdict_for(fake_prob):
        return "FAKE" if fake_prob >= IMAGE_FAKE_THRESHOLD else "REAL"

    return {
        "filename":             file.filename,
        "raw_sigmoid":          round(raw, 6),
        "img_size_used":        IMAGE_IMG_SIZE,
        "preprocess_mode":      IMAGE_PREPROCESS_MODE,
        "current_image_invert": IMAGE_INVERT_SIGMOID,
        "not_inverted": {
            "fake_prob": round(fake_prob_not_inverted, 6),
            "verdict":   verdict_for(fake_prob_not_inverted),
        },
        "inverted": {
            "fake_prob": round(fake_prob_inverted, 6),
            "verdict":   verdict_for(fake_prob_inverted),
        },
        "advice": (
            "Upload a KNOWN real image. Whichever of 'not_inverted' or 'inverted' "
            "gives the correct REAL verdict tells you the right IMAGE_INVERT_SIGMOID "
            "setting for your .env."
        ),
    }


@app.post("/analyse/image", response_model=AnalysisResponse)
async def analyse_image(file: UploadFile = File(...)):
    _require_models()

    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{file.content_type}'. Use JPG/PNG/WebP.",
        )

    raw_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read image: {exc}")

    print(f"\n[IMAGE] Analysing: {file.filename}  "
          f"size={len(raw_bytes)//1024}KB  "
          f"dims={pil_image.size}")

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
        message=(
            f"Image '{file.filename}' classified as {cnn_label.upper()} "
            f"with {fake_prob * 100:.1f}% fake probability."
        ),
    )


@app.post("/analyse/video", response_model=AnalysisResponse)
async def analyse_video(file: UploadFile = File(...)):
    _require_models()

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported format '{suffix}'.")

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

        print(f"\n[VIDEO] Analysing: {file.filename}  "
              f"total_frames={total_frames}  fps={fps:.1f}")

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

        # ── Batch CNN inference (video model) ─────────────────────────────────
        labels, confs, fake_probs = predict_cnn_batch(pil_frames)

        # ── Build per-frame results ───────────────────────────────────────────
        frame_results: list[FrameResult] = []

        for i, (lbl, conf, fake_p) in enumerate(zip(labels, confs, fake_probs)):
            verdict     = "fake" if fake_p >= VIDEO_FAKE_THRESHOLD else "real"
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
        mean_fake_prob   = float(np.mean(fake_probs))
        overall_verdict  = "fake" if mean_fake_prob >= VIDEO_FAKE_THRESHOLD else "real"
        overall_conf     = mean_fake_prob if overall_verdict == "fake" else (1.0 - mean_fake_prob)
        fake_frame_count = sum(1 for fr in frame_results if fr.verdict == "fake")

        print(f"[VIDEO] Result: {overall_verdict.upper()}  "
              f"mean_fake_prob={mean_fake_prob:.4f}  "
              f"fake_frames={fake_frame_count}/{len(frame_results)}")

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
