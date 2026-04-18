"""
groq_explain.py — Natural-language explanations via Groq API (v3)

Fixes vs v2:
  - GROQ_MODEL updated: llama3-8b-8192 was deprecated by Groq in Jan 2025.
    Primary model is now llama-3.1-8b-instant (fast, cheap, always available).
    GROQ_MODEL_FALLBACK = "llama3-70b-8192" tried automatically on 400/404.
  - HTTP error handler fixed: status code is now always an int before comparison,
    so the 401 / 429 / 400 / 404 branches actually trigger correctly.
    Previously `exc.response` could exist but `.status_code` was never cast,
    causing all errors to fall through to the "HTTP unknown" catch-all.
  - 400 Bad Request now returns a specific, actionable message.
  - 404 Not Found (retired model) automatically retries with the fallback model.
  - Response body logged inside the 400 message to aid future debugging.
  - max_tokens raised to 300 (llama-3.1 is more concise; 250 sometimes truncates).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL      = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT   = 25   # seconds

# Primary model — llama-3.1-8b-instant is the current recommended fast model.
# Fallback used automatically when the primary returns 400 / 404 (retired model).
GROQ_MODEL          = "llama-3.1-8b-instant"
GROQ_MODEL_FALLBACK = "llama3-70b-8192"


def _build_prompt(prediction: str, confidence: float) -> str:
    """Build the user prompt sent to Groq."""
    is_fake  = prediction.lower() == "fake"
    conf_pct = confidence * 100

    artifact_hints = (
        "Common deepfake artefacts include: subtle blending boundaries around the "
        "face, inconsistent skin texture or lighting direction, unnatural eye "
        "reflections, soft or warped edges near hairlines, and colour banding."
        if is_fake else
        "Signs of an authentic image include: consistent natural lighting across "
        "the face and background, realistic skin pores and texture, coherent "
        "facial symmetry, and sharp, stable edges with no blending artifacts."
    )

    return (
        "You are a computer-vision expert explaining an AI deepfake detection "
        "result to a non-technical user.\n\n"
        f"Result:     {'FAKE (AI-generated / deepfake)' if is_fake else 'REAL (authentic photograph)'}\n"
        f"Confidence: {conf_pct:.1f}%\n\n"
        f"{artifact_hints}\n\n"
        "Write a clear, professional explanation of 3–5 sentences:\n"
        "1. State the verdict and confidence plainly.\n"
        "2. Explain the most likely visual reasons behind this classification.\n"
        "3. Note any caveats if confidence is below 75%.\n\n"
        "Do NOT use bullet points. Reply in plain prose only."
    )


def _call_groq(prompt: str, model: str) -> requests.Response:
    """Send one request to the Groq chat completions endpoint."""
    return requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  300,
            "temperature": 0.4,
        },
        timeout=REQUEST_TIMEOUT,
    )


def explain_result(prediction: str, confidence: float) -> str:
    """
    Generate a concise explanation for a deepfake prediction via Groq.

    Parameters
    ----------
    prediction : str
        Class label — typically 'fake' or 'real' (case-insensitive).
    confidence : float
        Fake probability in [0, 1] as produced by ensemble_verdict().

    Returns
    -------
    str
        Plain-text explanation, or an error message prefixed with '⚠️'.
    """
    if not GROQ_API_KEY:
        return (
            "⚠️ GROQ_API_KEY is not set. "
            "Add it to your .env file to enable AI explanations."
        )

    # Defensive clamp — ensemble_verdict() should always return [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    prompt     = _build_prompt(prediction, confidence)

    try:
        response = _call_groq(prompt, GROQ_MODEL)

        # ── Auto-retry with fallback model on 400 / 404 ──────────────────────
        # 400 = bad request (often a retired / renamed model ID)
        # 404 = model not found
        if response.status_code in (400, 404):
            response = _call_groq(prompt, GROQ_MODEL_FALLBACK)

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "⚠️ Groq API timed out. Please try again in a moment."

    except requests.exceptions.HTTPError as exc:
        # Always cast to int so the comparisons below work correctly.
        # exc.response is guaranteed non-None here because raise_for_status()
        # only raises when a response was received.
        status: int = exc.response.status_code

        if status == 400:
            # Include the response body — it usually names the exact problem
            # (e.g. "model_not_found", "context_length_exceeded")
            try:
                detail = exc.response.json().get("error", {}).get("message", exc.response.text)
            except Exception:
                detail = exc.response.text
            return (
                f"⚠️ Groq returned 400 Bad Request: {detail}\n"
                f"Both models tried: '{GROQ_MODEL}' and '{GROQ_MODEL_FALLBACK}'.\n"
                "Check that your GROQ_API_KEY is valid and the models are available "
                "in your Groq account (https://console.groq.com/docs/models)."
            )

        if status == 401:
            return "⚠️ Invalid GROQ_API_KEY — check your .env file."

        if status == 404:
            return (
                f"⚠️ Groq model not found (HTTP 404).\n"
                f"Both models tried: '{GROQ_MODEL}' and '{GROQ_MODEL_FALLBACK}'.\n"
                "Visit https://console.groq.com/docs/models for current model IDs."
            )

        if status == 429:
            return "⚠️ Groq rate limit reached — wait a few seconds and retry."

        if status == 503:
            return "⚠️ Groq service temporarily unavailable — try again shortly."

        return f"⚠️ Groq API returned HTTP {status}: {exc}"

    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot reach Groq API — check your internet connection."

    except requests.exceptions.RequestException as exc:
        return f"⚠️ Network error contacting Groq: {exc}"

    except (KeyError, IndexError, ValueError) as exc:
        return f"⚠️ Unexpected Groq response format: {exc}"