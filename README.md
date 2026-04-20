<div align="center">

<img src="https://img.shields.io/badge/DeepScan-AI%20Deepfake%20Detector-4285f4?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMTQgMkwyNiAxNEwxNCAyNkwyIDE0WiIgZmlsbD0id2hpdGUiLz48L3N2Zz4=" />

# 🔍 DeepScan — AI Deepfake Detector

**Detect AI-generated images and deepfake videos using a CNN (MobileNetV2) pipeline.**

[![Backend](https://img.shields.io/badge/Backend-Hugging%20Face%20Spaces-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/sanjay72005/deepscan)
[![Frontend](https://img.shields.io/badge/Frontend-Netlify-00C7B7?style=flat-square&logo=netlify)](https://deepscan.netlify.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

DeepScan is a full-stack deepfake detection web application that analyses images and videos to determine whether they are AI-generated or authentic. It uses a fine-tuned **MobileNetV2 CNN** as its core classification model, served via a **Flask + FastAPI** backend hosted on **Hugging Face Spaces**, with a polished **chat-style frontend** deployed on **Netlify**.

The scoring formula is:

```
fake_prob = 1.0 − raw_sigmoid(model_output)
```

A threshold of **0.50** is applied — scores at or above 0.50 are classified as **FAKE**, below as **REAL**.

---

## ✨ Features

- 🖼️ **Image analysis** — Upload JPG / PNG / WebP images for instant CNN deepfake scoring
- 🎬 **Video analysis** — Frame-by-frame CNN analysis with per-frame thumbnails, timeline chart, and frame table
- 📊 **Probability bar** — Visual fake probability gauge with 50% threshold marker
- 🤖 **AI explanations** — Groq LLM (LLaMA 3) generates human-readable forensic explanations per scan
- 📄 **PDF reports** — Professional multi-page PDF report download (image & video variants)
- 🕓 **History view** — Local scan history with thumbnails and verdict pills
- ⚙️ **Settings panel** — Configurable thresholds, Groq API key, Flask server URL
- 📱 **Fully responsive** — Mobile-first design with slide-in sidebar and touch-friendly UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│         FRONTEND                │        │            BACKEND               │
│  Netlify (Static HTML/CSS/JS)   │◄──────►│  Hugging Face Spaces             │
│                                 │  REST  │  Flask  +  FastAPI               │
│  • app.html  (chat UI)          │  API   │  • /api/analyse/image            │
│  • static/style.css             │        │  • /api/analyse/video            │
│  • static/app.js                │        │  • /api/groq/explain             │
│  • index.html  (landing page)   │        │  • /api/status                   │
└─────────────────────────────────┘        │                                  │
                                           │  MobileNetV2 CNN                 │
                                           │  deepfake_model.h5               │
                                           └──────────────────────────────────┘
```

| Layer | Technology | Hosting |
|---|---|---|
| Frontend | Vanilla HTML / CSS / JavaScript | **Netlify** |
| Backend API | Flask + FastAPI (Python) | **Hugging Face Spaces** |
| ML Model | MobileNetV2 (TensorFlow / Keras) | Hugging Face Spaces |
| AI Explanations | Groq API — LLaMA 3.1 8B / 70B | Client-side or Flask proxy |
| PDF Generation | jsPDF (client-side) | Browser |

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
| Fine-tuned on | Deepfake vs Real image/video dataset |
| Output | Sigmoid (0 = fake tendency, 1 = real tendency) |
| Scoring | `fake_prob = 1.0 − raw_sigmoid` |
| Threshold | **0.50** |
| Input size | 224 × 224 px |

### Datasets

#### 🖼️ Image Dataset
**Deepfake vs Real — 60K**
> Prithiv Sakthiur · Kaggle
> [https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k](https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k)

A balanced dataset of 60,000 images split evenly between AI-generated deepfake faces and real photographs, used to train and validate the image classification head.

#### 🎬 Video Dataset
**FaceForensics++**
> ondyari · GitHub
> [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)

A large-scale benchmark dataset of manipulated facial videos covering multiple manipulation methods (DeepFakes, Face2Face, FaceSwap, NeuralTextures). Video frames were extracted and used to extend the training set for video-mode inference.

---

## 🚀 Getting Started

### Prerequisites

```bash
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
pip install -r requirements.txt
```

### 3. Add your model weights

Place your trained `deepfake_model.h5` in the project root, or train from scratch:

```bash
python train.py
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_key_here
```

> **Groq API key** is optional — without it, the AI explanation feature is disabled (users can supply their own key in the UI settings panel).

### 5. Start the backend

```bash
python flask_app.py
```

The Flask server starts at `http://localhost:7860` and automatically spawns the FastAPI inference service internally.

### 6. Open the frontend

Open `app.html` in your browser, or serve statically:

```bash
# Quick static server (Python)
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

All CNN inference, frame extraction (video), and Groq proxy calls are handled here. The frontend calls this origin directly via `fetch`.

### Frontend — Netlify

The static frontend (`app.html`, `static/`, `index.html`) is deployed via **Netlify**:

```
https://deepscan.netlify.app
```

No build step required — Netlify serves the files as-is. To redeploy, simply push to the connected GitHub repository or drag-and-drop the folder into the Netlify dashboard.

---

## 🔌 API Reference

All endpoints are served from the Hugging Face Spaces backend.

### `GET /api/status`
Returns server health, model load status, and Groq configuration flag.

```json
{
  "fastapi_ok": true,
  "model_loaded": true,
  "groq_configured": true
}
```

### `POST /api/analyse/image`
Analyse a single image for deepfake content.

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
Analyse a video file frame-by-frame.

**Request:** `multipart/form-data` — field `file` (MP4 / MOV / AVI / WebM, max 5 MB)

**Response:**
```json
{
  "media_type": "video",
  "verdict": "fake",
  "fake_probability": 0.6541,
  "fake_frame_count": 14,
  "total_frames_analysed": 20,
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

---

## 🖥️ UI Walkthrough

| Step | Action |
|---|---|
| **1** | Click the 📎 attach button or drag & drop a file |
| **2** | Select **Image** or **Video** mode |
| **3** | *(Optional)* Click ⚙️ to enter your Groq API key for AI explanations |
| **4** | Press the 🔍 **Analyse** button |
| **5** | View verdict, probability bar, signal table, and AI explanation |
| **6** | Download a **Professional PDF Report** |

> If the Hugging Face backend is offline, the UI automatically falls back to a **demo simulation** so you can preview the full interface without a live server.

---

## 📦 Dependencies

### Python (Backend)
```
flask
fastapi
uvicorn
tensorflow / keras
opencv-python
numpy
pillow
python-dotenv
requests
groq
```

### JavaScript (Frontend — CDN)
```
jsPDF  2.5.1  — client-side PDF generation
```

---

## ⚠️ Limitations & Caveats

- CNN-only pipeline — no ensemble or second-opinion model (e.g. Sightengine was removed)
- `fake_prob = 1 − sigmoid` means the model's raw output is **inverted** — high sigmoid = more real
- Video inference can take **30–90 seconds** depending on file length and server load
- Results are probabilistic — not a certified forensic determination
- A 5 MB file size limit is enforced on both image and video uploads

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
- [Groq](https://groq.com) — Ultra-fast LLM inference for AI explanations
- [Hugging Face Spaces](https://huggingface.co/spaces) — Free GPU-backed backend hosting
- [Netlify](https://netlify.com) — Zero-config static frontend hosting

---

<div align="center">

Made with ❤️ by [Sanjay](https://github.com/Sanjay20057)

**[🌐 Live Demo](https://deepscan.netlify.app)** · **[🤗 Backend API](https://huggingface.co/spaces/sanjay72005/deepscan)** · **[📦 Dataset (Images)](https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-60k)** · **[📦 Dataset (Videos)](https://github.com/ondyari/FaceForensics)**

</div>
