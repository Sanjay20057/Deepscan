<div align="center">

# 🔍 DeepScan — AI Deepfake Detector

**Detect AI-generated images and deepfake videos using a CNN + ViT ensemble pipeline.**

[![Backend](https://img.shields.io/badge/Backend-Hugging%20Face%20Spaces-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/sanjay72005/deepscan)
[![Frontend](https://img.shields.io/badge/Frontend-Netlify-00C7B7?style=flat-square&logo=netlify)](https://deepscan.netlify.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

DeepScan is a full-stack deepfake detection web application that analyses images and videos to determine whether they are AI-generated or authentic. It uses a fine-tuned **MobileNetV2 CNN** ensembled with a **HuggingFace ViT** (`Wvolf/ViT_Deepfake_Detection`) as its core classification models, served via a **Flask + FastAPI** backend hosted on **Hugging Face Spaces**, with a polished **chat-style frontend** deployed on **Netlify**.

The scoring pipeline differs by media type:

**Image:**

```
cnn_fake_prob  = raw_sigmoid                        (IMAGE_INVERT_SIGMOID = False)
hf_fake_prob   = ViT score for "fake" label
fake_prob      = max(cnn_fake_prob, hf_fake_prob)   ← ensemble
threshold      = 0.35
```

**Video:**

```
fake_prob  = 1.0 − raw_sigmoid                      (VIDEO_INVERT_SIGMOID = True)
threshold  = 0.50
```

---

## ✨ Features

- 🖼️ **Image analysis** — Upload JPG / PNG / WebP images for CNN + ViT ensemble deepfake scoring
- 🎬 **Video analysis** — Frame-by-frame CNN analysis (25 frames sampled) with per-frame thumbnails, timeline chart, and frame table
- 📊 **Probability bar** — Visual fake probability gauge with threshold marker
- 🤖 **AI explanations** — Groq LLM (LLaMA 3.1 8B / 70B) generates human-readable forensic explanations per scan
- 📄 **PDF reports** — Professional multi-page PDF report download (image & video variants)
- 🕓 **History view** — Local scan history with thumbnails and verdict pills
- ⚙️ **Settings panel** — Configurable thresholds, Groq API key, Flask server URL
- 🔬 **Diagnose endpoint** — `/diagnose` API to inspect raw sigmoid values and validate model inversion settings
- 📱 **Fully responsive** — Mobile-first design with slide-in sidebar and touch-friendly UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│         FRONTEND                │        │            BACKEND               │
│  Netlify (Static HTML/CSS/JS)   │◄──────►│  Hugging Face Spaces             │
│                                 │  REST  │  Flask (port 7860)               │
│  • app.html  (chat UI)          │  API   │  + FastAPI (port 8000)           │
│  • static/style.css             │        │  • /api/analyse/image            │
│  • static/app.js                │        │  • /api/analyse/video            │
│  • index.html  (landing page)   │        │  • /api/groq/explain             │
└─────────────────────────────────┘        │  • /api/status                   │
                                           │  • /diagnose                     │
                                           │                                  │
                                           │  MobileNetV2 CNN                 │
                                           │  deepfake_model.h5               │
                                           │  + ViT (HuggingFace ensemble)    │
                                           └──────────────────────────────────┘
```

| Layer | Technology | Hosting |
|---|---|---|
| Frontend | Vanilla HTML / CSS / JavaScript | **Netlify** |
| Backend API | Flask + FastAPI (Python) | **Hugging Face Spaces** |
| ML Model (Image) | MobileNetV2 CNN + ViT ensemble (TensorFlow / Keras + HuggingFace) | Hugging Face Spaces |
| ML Model (Video) | MobileNetV2 CNN — batch frame inference | Hugging Face Spaces |
| AI Explanations | Groq API — LLaMA 3.1 8B / 70B | Client-side or Flask proxy |
| PDF Generation | jsPDF 2.5.1 (client-side) | Browser |

---

## 📁 Project Structure

```
Deepscan/
├── Backend/
│   ├── flask_app.py
│   ├── deepfake_model.h5
│   ├── __init__.py
│   ├── start.sh
│   ├── video_api.py
│   └── requirements.txt
├── Frontend/
│   ├── static/
│   │   ├── style.css
│   │   └── app.js
│   ├── app.html
│   └── index.html
└── README.md
```

---

## 🧠 Model & Dataset

### Architecture

| Detail | Value |
|---|---|
| Base model | MobileNetV2 (ImageNet pre-trained) |
| Ensemble model | ViT — `Wvolf/ViT_Deepfake_Detection` (HuggingFace, images only) |
| Fine-tuned on | Deepfake vs Real image/video dataset |
| Output | Sigmoid (behaviour differs by media type — see scoring above) |
| Image scoring | `fake_prob = max(cnn_sigmoid, vit_score)` · threshold **0.35** |
| Video scoring | `fake_prob = 1.0 − raw_sigmoid` · threshold **0.50** |
| Frames per video | **25** (evenly sampled) |
| Input size | 224 × 224 px |
| Preprocessing | `mobilenet_v2.preprocess_input()` — scales to **[−1, 1]** |

### Datasets

#### 🖼️ Image Dataset

**Deepfake vs Real — 60K** · Prithiv Sakthiur · Kaggle

https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k

A balanced dataset of 60,000 images split evenly between AI-generated deepfake faces and real photographs, used to train and validate the image classification head.

#### 🎬 Video Dataset

**FaceForensics++** · ondyari · GitHub

https://github.com/ondyari/FaceForensics

A large-scale benchmark dataset of manipulated facial videos covering multiple manipulation methods (DeepFakes, Face2Face, FaceSwap, NeuralTextures). Video frames were extracted and used to extend the training set for video-mode inference.

---

## 🚀 Getting Started

### Prerequisites

```
python >= 3.10
pip
node (optional — for local frontend dev)
```

### 1. Clone the repository

```bash
git clone https://github.com/Sanjay20057/Deepscan.git
cd Deepscan
```

### 2. Install Python dependencies

```bash
pip install -r Backend/requirements.txt
```

### 3. Add your model weights

Place your trained `deepfake_model.h5` in the `Backend/` directory, or train from scratch:

```bash
python train.py
```

### 4. Configure environment variables

Create a `.env` file in the `Backend/` directory:

```env
GROQ_API_KEY=gsk_your_groq_key_here

# Optional overrides (defaults shown)
FASTAPI_BASE_URL=http://localhost:8000
FLASK_PORT=7860
MAX_UPLOAD_MB=5
FRAMES_PER_VIDEO=25
IMAGE_FAKE_THRESHOLD=0.35
VIDEO_FAKE_THRESHOLD=0.50
IMAGE_INVERT_SIGMOID=false
VIDEO_INVERT_SIGMOID=true
PREPROCESS_MODE=mobilenet
```

> **Groq API key** is optional — without it, the AI explanation feature is disabled. Users can supply their own key in the UI settings panel.

### 5. Start the backend

```bash
# Option A — convenience script
bash Backend/start.sh

# Option B — manually in two terminals
python Backend/flask_app.py
uvicorn video_api:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the frontend

```bash
python -m http.server 3000
# Then visit http://localhost:3000/app.html
```

---

## 🌐 Deployment

### Backend — Hugging Face Spaces

The Flask + FastAPI backend is hosted as a **Gradio SDK Space** on Hugging Face:

```
https://huggingface.co/spaces/sanjay72005/deepscan
```

All CNN inference, ViT ensemble scoring, frame extraction (video), and Groq proxy calls are handled here. The frontend calls this origin directly via `fetch`.

### Frontend — Netlify

The static frontend (`app.html`, `static/`, `index.html`) is deployed via **Netlify**:

```
https://deepscan.netlify.app
```

No build step required — Netlify serves the files as-is. Push to the connected GitHub repository or drag-and-drop the `Frontend/` folder into the Netlify dashboard to redeploy.

---

## 🔌 API Reference

All endpoints are served from `https://sanjay72005-deepscan.hf.space`.

### `GET /api/status`

Returns server health, model load status, pipeline info, and Groq configuration flag.

```json
{
  "fastapi_ok": true,
  "model_loaded": true,
  "groq_configured": true,
  "pipeline": "cnn-only",
  "fake_threshold": 0.5,
  "max_upload_mb": 5
}
```

### `POST /api/analyse/image`

Analyse a single image using the CNN + ViT ensemble.

**Request:** `multipart/form-data` — field `file` (JPG / PNG / WebP, max 5 MB)

**Response:**

```json
{
  "media_type": "image",
  "verdict": "fake",
  "fake_probability": 0.7823,
  "confidence_pct": 78.2,
  "cnn_label": "fake",
  "cnn_confidence": 0.7823
}
```

### `POST /api/analyse/video`

Analyse a video file frame-by-frame (25 frames sampled by default).

**Request:** `multipart/form-data` — field `file` (MP4 / MOV / AVI / MKV / WebM / FLV, max 5 MB)

**Response:**

```json
{
  "media_type": "video",
  "verdict": "fake",
  "fake_probability": 0.6541,
  "fake_frame_count": 14,
  "total_frames_analysed": 25,
  "frame_results": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "cnn_label": "fake",
      "cnn_confidence": 0.712,
      "fake_prob": 0.712,
      "verdict": "fake",
      "thumbnail_b64": "..."
    }
  ]
}
```

### `POST /api/groq/explain`

Proxy endpoint for Groq LLM explanation generation (uses server-side API key).

**Request:**

```json
{ "prompt": "...", "max_tokens": 460 }
```

### `POST /diagnose`

Debug endpoint — upload any image to inspect raw sigmoid values and verify `INVERT_SIGMOID` is set correctly for your model weights.

**Response:**

```json
{
  "raw_sigmoid": 0.123456,
  "fake_prob_current": 0.876544,
  "verdict_current": "FAKE",
  "fake_prob_if_flipped": 0.123456,
  "verdict_if_flipped": "REAL",
  "advice": "If verdict_current is wrong for a KNOWN real image, set INVERT_SIGMOID=false..."
}
```

---

## 🖥️ UI Walkthrough

| Step | Action |
|---|---|
| **1** | Click the 📎 attach button or drag & drop a file (max 5 MB) |
| **2** | Select **Image** or **Video** mode |
| **3** | *(Optional)* Click ⚙️ to enter your Groq API key for AI explanations |
| **4** | Press the 🔍 **Analyse** button (or press `Enter`) |
| **5** | View verdict, probability bar, signal table, and AI explanation |
| **6** | Download a **Professional PDF Report** |

> If the Hugging Face backend is offline, the UI automatically falls back to a **demo simulation** so you can preview the full interface without a live server.

---

## 📦 Dependencies

### Python (Backend)

```
flask
flask-cors
fastapi
uvicorn
tensorflow / keras
opencv-python
numpy
pillow
python-dotenv
requests
transformers
```

### JavaScript (Frontend — CDN)

```
jsPDF 2.5.1 — client-side PDF generation
```

---

## ⚠️ Limitations & Caveats

- Image mode uses a CNN + ViT **ensemble** (max of both scores); video mode uses CNN only
- Preprocessing uses `mobilenet_v2.preprocess_input()` — scales pixels to **[−1, 1]**, not plain `/255`
- `IMAGE_INVERT_SIGMOID = False` (sigmoid used directly for images); `VIDEO_INVERT_SIGMOID = True` (`fake_prob = 1 − sigmoid` for video)
- Image fake threshold is **0.35**; video fake threshold is **0.50**
- Video inference can take **30–90 seconds** depending on file length and server load
- Results are probabilistic — not a certified forensic determination
- A **5 MB** file size limit is enforced on both frontend and backend for all uploads

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Prithiv Sakthiur](https://www.kaggle.com/prithivsakthiur) — Deepfake vs Real 60K image dataset
- [ondyari / FaceForensics](https://github.com/ondyari/FaceForensics) — FaceForensics++ video benchmark
- [Google MobileNetV2](https://arxiv.org/abs/1801.04381) — Backbone CNN architecture
- [Wvolf / ViT_Deepfake_Detection](https://huggingface.co/Wvolf/ViT_Deepfake_Detection) — HuggingFace ViT ensemble model
- [Groq](https://groq.com) — Ultra-fast LLM inference for AI explanations
- [Hugging Face Spaces](https://huggingface.co/spaces) — Free GPU-backed backend hosting
- [Netlify](https://netlify.com) — Zero-config static frontend hosting

---

<div align="center">

Made with ❤️ by [Sanjay](https://github.com/Sanjay20057)

**[🌐 Live Demo](https://deepscan.netlify.app)** · **[🤗 Backend API](https://huggingface.co/spaces/sanjay72005/deepscan)** · **[📦 Dataset (Images)](https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k)** · **[📦 Dataset (Videos)](https://github.com/ondyari/FaceForensics)**

</div>
