"""
flask_app.py — Flask web server for DeepScan (CNN-Only)
"""

import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

load_dotenv()

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent / "Frontend"

# ── CONFIG ────────────────────────────────────────────────────
FASTAPI_BASE_URL  = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "").strip()
FLASK_SECRET_KEY  = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
HISTORY_FILE      = Path(os.getenv("HISTORY_FILE", str(_HERE / "data" / "history.json")))

# ── FILE SIZE: 5 MB hard limit (matches frontend) ─────────────
MAX_UPLOAD_MB     = int(os.getenv("MAX_UPLOAD_MB", "5"))

PORT              = int(os.getenv("PORT", os.getenv("FLASK_PORT", 7860)))
IS_PRODUCTION     = os.getenv("RENDER", "") != ""
FLASK_DEBUG       = False if IS_PRODUCTION else os.getenv("FLASK_DEBUG", "true").lower() == "true"

GROQ_API_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_PRIMARY  = "llama-3.1-8b-instant"
GROQ_MODEL_FALLBACK = "llama3-70b-8192"
GROQ_REQUEST_TIMEOUT = 30

ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "webm", "flv"}

IMAGE_FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.50"))


# ── FACTORY ───────────────────────────────────────────────────
def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(_ROOT / "static"),
        template_folder=str(_ROOT)
    )
    app.secret_key = FLASK_SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── CORS — allows Netlify / GitHub Pages to call this API ─
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── STATIC FRONTEND ───────────────────────────────────────
    @app.route("/")
    def index():
        return send_from_directory(str(_ROOT), "index.html")

    @app.route("/app")
    def app_ui():
        return send_from_directory(str(_ROOT), "app.html")

    @app.route("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(str(_ROOT / "static"), filename)

    # ── HEALTH / STATUS ───────────────────────────────────────
    @app.route("/health")
    def health_check():
        return jsonify({"status": "ok", "service": "flask"}), 200

    @app.route("/api/status")
    def api_status():
        fastapi_ok     = False
        model_loaded   = False
        fastapi_detail = {}
        try:
            r = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=4)
            fastapi_detail = r.json()
            fastapi_ok     = r.status_code == 200
            model_loaded   = fastapi_detail.get("model_loaded", False)
        except Exception as exc:
            fastapi_detail = {"error": str(exc)}

        return jsonify({
            "flask":                  "ok",
            "fastapi_url":            FASTAPI_BASE_URL,
            "fastapi_ok":             fastapi_ok,
            "model_loaded":           model_loaded,
            "fastapi_detail":         fastapi_detail,
            "groq_configured":        bool(GROQ_API_KEY),
            "sightengine_configured": False,
            "pipeline":               "cnn-only",
            "fake_threshold":         IMAGE_FAKE_THRESHOLD,
            "max_upload_mb":          MAX_UPLOAD_MB,
            "environment":            "production" if IS_PRODUCTION else "development",
            "timestamp":              datetime.utcnow().isoformat() + "Z",
        })

    # ── PROXY: IMAGE ANALYSIS ─────────────────────────────────
    @app.route("/api/analyse/image", methods=["POST"])
    def analyse_image():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]
        if not _allowed_file(file.filename, ALLOWED_IMAGE_EXTS):
            return jsonify({"error": "Unsupported image type"}), 415

        file_bytes = file.read()

        # ── Server-side 5 MB guard (second layer after frontend) ──
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            return jsonify({
                "error": f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB. "
                         f"Received {len(file_bytes) / 1048576:.2f} MB."
            }), 413

        filename = secure_filename(file.filename or "upload.jpg")

        try:
            resp = requests.post(
                f"{FASTAPI_BASE_URL}/analyse/image",
                files={"file": (filename, io.BytesIO(file_bytes), file.content_type or "image/jpeg")},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            _append_history(filename, result, "image")
            return jsonify(result)
        except requests.exceptions.ConnectionError:
            return jsonify({
                "error": (
                    "FastAPI backend is offline. "
                    "Start it with: uvicorn video_api:app --host 0.0.0.0 --port 8000"
                )
            }), 503
        except requests.exceptions.Timeout:
            return jsonify({"error": "CNN inference timed out."}), 504
        except requests.exceptions.RequestException as exc:
            return jsonify({"error": str(exc)}), 502

    # ── PROXY: VIDEO ANALYSIS ─────────────────────────────────
    @app.route("/api/analyse/video", methods=["POST"])
    def analyse_video():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]
        if not _allowed_file(file.filename, ALLOWED_VIDEO_EXTS):
            return jsonify({"error": "Unsupported video type"}), 415

        file_bytes = file.read()

        # ── Server-side 5 MB guard ────────────────────────────
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            return jsonify({
                "error": f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB. "
                         f"Received {len(file_bytes) / 1048576:.2f} MB."
            }), 413

        filename     = secure_filename(file.filename or "upload.mp4")
        content_type = file.content_type or "video/mp4"

        try:
            resp = requests.post(
                f"{FASTAPI_BASE_URL}/analyse/video",
                files={"file": (filename, io.BytesIO(file_bytes), content_type)},
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            _append_history(filename, result, "video")
            return jsonify(result)
        except requests.exceptions.Timeout:
            return jsonify({"error": "Video analysis timed out (300 s). Try a shorter clip."}), 504
        except requests.exceptions.ConnectionError:
            return jsonify({
                "error": (
                    "FastAPI backend is offline. "
                    "Start it with: uvicorn video_api:app --host 0.0.0.0 --port 8000"
                )
            }), 503
        except requests.exceptions.RequestException as exc:
            return jsonify({"error": str(exc)}), 502

    # ── GROQ EXPLAIN ──────────────────────────────────────────
    @app.route("/api/groq/explain", methods=["POST"])
    def groq_explain():
        if not GROQ_API_KEY:
            return jsonify({"error": "GROQ_API_KEY not configured on server"}), 503

        data       = request.get_json(silent=True) or {}
        prompt     = data.get("prompt", "").strip()
        max_tokens = int(data.get("max_tokens", 400))

        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        text, err = _call_groq(prompt, max_tokens)
        if err:
            return jsonify({"error": err}), 502
        return jsonify({"text": text})

    # ── HISTORY ───────────────────────────────────────────────
    @app.route("/api/history", methods=["GET"])
    def get_history():
        return jsonify(_load_history())

    @app.route("/api/history", methods=["DELETE"])
    def clear_history():
        _save_history([])
        return jsonify({"cleared": True})

    # ── ERROR HANDLERS ────────────────────────────────────────
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({
            "error": f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB."
        }), 413

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

    return app


# ── HELPERS ───────────────────────────────────────────────────
def _allowed_file(filename: str, allowed: set) -> bool:
    if not filename:
        return False
    ext = Path(filename).suffix.lstrip(".").lower()
    return ext in allowed


def _call_groq(prompt: str, max_tokens: int = 400) -> tuple:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": 0.4,
    }

    for model_name in (GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK):
        try:
            r = requests.post(
                GROQ_API_URL,
                headers=headers,
                json={**payload, "model": model_name},
                timeout=GROQ_REQUEST_TIMEOUT,
            )
            if r.status_code in (400, 404):
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return text, None
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            if status == 401:
                return "", "Invalid GROQ_API_KEY"
            if status == 429:
                return "", "Groq rate limit reached — retry shortly"
            if status == 503:
                return "", "Groq service temporarily unavailable"
        except requests.exceptions.Timeout:
            return "", "Groq API request timed out"
        except requests.exceptions.RequestException as exc:
            return "", f"Network error: {exc}"
        except (KeyError, IndexError):
            return "", "Unexpected Groq response format"

    return "", f"Both Groq models failed ({GROQ_MODEL_PRIMARY}, {GROQ_MODEL_FALLBACK})"


def _load_history() -> list:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text())
    except Exception:
        pass
    return []


def _save_history(items: list) -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(items, indent=2))
    except Exception:
        pass


def _append_history(filename: str, result: dict, media_type: str) -> None:
    items = _load_history()
    items.insert(0, {
        "id":         str(uuid.uuid4()),
        "filename":   filename,
        "media_type": media_type,
        "verdict":    result.get("verdict", "unknown"),
        "fake_prob":  result.get("fake_probability", 0.0),
        "timestamp":  datetime.utcnow().isoformat() + "Z",
    })
    _save_history(items[:100])


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()

    print(f"\n🚀 DeepScan Flask server starting on http://0.0.0.0:{PORT}")
    print(f"   Landing page    : http://localhost:{PORT}/")
    print(f"   App interface   : http://localhost:{PORT}/app")
    print(f"   FastAPI backend : {FASTAPI_BASE_URL}")
    print(f"   Groq configured : {'✅' if GROQ_API_KEY else '❌  (add GROQ_API_KEY to .env)'}")
    print(f"   Pipeline        : CNN-only (MobileNetV2 deepfake_model.h5)")
    print(f"   Fake threshold  : {IMAGE_FAKE_THRESHOLD}")
    print(f"   Max upload size : {MAX_UPLOAD_MB} MB")
    print(f"   Environment     : {'🌐 Production (Render)' if IS_PRODUCTION else '💻 Development'}")
    print(f"   Debug mode      : {FLASK_DEBUG}")
    print(f"   .env path       : {_HERE / '.env'}\n")

    app.run(host="0.0.0.0", port=PORT, debug=FLASK_DEBUG)
