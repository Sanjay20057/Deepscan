'use strict';

// ── STATE ─────────────────────────────────────────────────────
let currentMode     = 'image';
let selectedFile    = null;
let isAnalysing     = false;
let analysisHistory = [];
let lastResult      = null;
let lastPreviewUrl  = null;

try { analysisHistory = JSON.parse(localStorage.getItem('deepscan_history') || '[]'); }
catch (_) { analysisHistory = []; }

const FLASK_ORIGIN = 'https://sanjay72005-deepscan.hf.space';

// Pipeline: CNN-only. fake_prob = 1.0 - raw_sigmoid. Threshold = 0.50.
const FAKE_THRESHOLD = 0.50;

// ─────────────────────────────────────────────────────────────
//  INIT
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  addBotMessage(introHTML());
  setupDragDrop();
  setupFileInputs();
  renderHistory();
  restoreSettings();
  checkApiStatus();
  setInterval(checkApiStatus, 20000);

  document.addEventListener('click', (e) => {
    const menu = document.getElementById('attach-menu');
    const btn  = document.getElementById('attach-btn');
    if (menu && !menu.contains(e.target) && btn && !btn.contains(e.target)) {
      menu.style.display = 'none';
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) {
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      e.preventDefault();
      runAnalysis();
    }
  });

  const obs = new MutationObserver(muts => {
    muts.forEach(m => m.addedNodes.forEach(node => {
      if (node.nodeType !== 1) return;
      node.querySelectorAll('canvas.timeline-chart[data-chart]').forEach(drawTimeline);
    }));
  });
  obs.observe(document.getElementById('messages'), { childList: true, subtree: true });
});

// ─────────────────────────────────────────────────────────────
//  UTILS
// ─────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function pct(v) { return (parseFloat(v) || 0) * 100; }
function timeStr() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ─────────────────────────────────────────────────────────────
//  FILE TYPE DETECTION — always derive from actual file, never
//  trust currentMode alone (fixes mobile vs desktop mismatch)
// ─────────────────────────────────────────────────────────────
function fileIsImage(file) {
  if (!file) return false;
  const n = file.name.toLowerCase();
  return /\.(jpe?g|png|webp|gif|bmp|tiff?)$/.test(n) || file.type.startsWith('image/');
}
function fileIsVideo(file) {
  if (!file) return false;
  const n = file.name.toLowerCase();
  return /\.(mp4|avi|mov|mkv|webm|flv|wmv|m4v)$/.test(n) || file.type.startsWith('video/');
}

// ─────────────────────────────────────────────────────────────
//  NORMALISE SERVER RESPONSE
//  Trust the server verdict. Only derive verdict from fake_prob
//  if server did not send one. Never auto-invert.
// ─────────────────────────────────────────────────────────────
function normaliseServerResult(raw, isImage) {
  const result = Object.assign({}, raw);

  result.media_type = isImage ? 'image' : 'video';

  let fp = parseFloat(result.fake_probability);
  if (isNaN(fp) || fp < 0 || fp > 1) fp = 0;
  result.fake_probability = fp;

  // Trust server verdict if present and valid; only derive if missing
  if (result.verdict !== 'fake' && result.verdict !== 'real') {
    result.verdict = fp >= FAKE_THRESHOLD ? 'fake' : 'real';
  }

  // Sync confidence_pct
  result.confidence_pct = parseFloat((fp * 100).toFixed(1));

  // Sync cnn_label to match verdict if missing
  if (!result.cnn_label) {
    result.cnn_label = result.verdict;
  }

  // Normalise frame_results for video — trust per-frame verdict from server
  if (!isImage && Array.isArray(result.frame_results)) {
    result.frame_results = result.frame_results.map(fr => {
      const frCopy = Object.assign({}, fr);
      let frFp = parseFloat(frCopy.fake_prob);
      if (isNaN(frFp) || frFp < 0 || frFp > 1) frFp = 0;
      frCopy.fake_prob = frFp;

      // Trust server frame verdict if valid; derive only if missing
      if (frCopy.verdict !== 'fake' && frCopy.verdict !== 'real') {
        frCopy.verdict = frFp >= FAKE_THRESHOLD ? 'fake' : 'real';
      }
      if (!frCopy.cnn_label) frCopy.cnn_label = frCopy.verdict;
      return frCopy;
    });

    // Recount from normalised frames
    result.fake_frame_count      = result.frame_results.filter(f => f.verdict === 'fake').length;
    result.total_frames_analysed = result.frame_results.length;
  }

  return result;
}

// ─────────────────────────────────────────────────────────────
//  INTRO
// ─────────────────────────────────────────────────────────────
function introHTML() {
  return `
    <div class="intro-card">
      <div class="intro-title">Welcome to DeepScan</div>
      <div class="intro-body">
        Detect AI-generated images and deepfake videos using a
        <strong style="color:var(--text)">CNN (MobileNetV2)</strong> pipeline.
        <br><span style="font-size:0.78rem;color:var(--text-3)">
          Pure CNN inference — <code>fake_prob = 1 − sigmoid</code> · Threshold: 0.50
        </span>
      </div>
      <div class="intro-steps">
        <b>1.</b> Click the <b>📎 attach</b> button or drag &amp; drop your file<br>
        <b>2.</b> Choose <b>Image</b> or <b>Video</b> mode<br>
        <b>3.</b> Click the ⚙️ icon to add a <b>Groq API key</b> for AI explanations<br>
        <b>4.</b> Press <b style="color:var(--accent-blue)">Analyse</b>
      </div>
      <div style="font-size:0.78rem;color:var(--text-3)">
        No server? A demo simulation runs automatically so you can preview the UI.
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────
//  ATTACH MENU
// ─────────────────────────────────────────────────────────────
function openAttachMenu(e) {
  if (e) e.stopPropagation();
  const menu = document.getElementById('attach-menu');
  if (!menu) return;
  menu.style.display = menu.style.display === 'none' ? 'flex' : 'none';
}

function closeAttachMenu() {
  const menu = document.getElementById('attach-menu');
  if (menu) menu.style.display = 'none';
}

// ─────────────────────────────────────────────────────────────
//  ADVANCED PANEL TOGGLE
// ─────────────────────────────────────────────────────────────
function toggleAdvanced() {
  const panel = document.getElementById('gemini-advanced-panel');
  const btn   = document.getElementById('advanced-toggle');
  if (!panel) return;
  const isOpen = panel.classList.contains('open');
  panel.classList.toggle('open', !isOpen);
  if (btn) btn.classList.toggle('active', !isOpen);
}

// ─────────────────────────────────────────────────────────────
//  VIEW SWITCHING
// ─────────────────────────────────────────────────────────────
function switchView(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('view-' + view).classList.add('active');
  document.querySelector(`[data-view="${view}"]`).classList.add('active');
}

// ─────────────────────────────────────────────────────────────
//  MODE SWITCHING — never open file picker when only switching mode
// ─────────────────────────────────────────────────────────────
function selectMode(mode, openPicker = false) {
  currentMode = mode;
  document.querySelectorAll('.gemini-mode-pill').forEach(t => t.classList.remove('active'));
  const pill = document.getElementById('gpill-' + mode);
  if (pill) pill.classList.add('active');
  const ph = document.getElementById('gemini-placeholder');
  if (ph) ph.textContent = mode === 'image' ? 'Upload an image to analyse…' : 'Upload a video to analyse…';
  if (openPicker && !selectedFile) {
    triggerFileInput();
  }
}

// ─────────────────────────────────────────────────────────────
//  FILE HANDLING
// ─────────────────────────────────────────────────────────────
function setupFileInputs() {
  ['image', 'video'].forEach(type => {
    const el = document.getElementById('file-input-' + type);
    if (el) el.addEventListener('change', e => {
      if (e.target.files[0]) handleFile(e.target.files[0]);
    });
  });
}

function triggerFileInput() {
  const inputId = currentMode === 'image' ? 'file-input-image' : 'file-input-video';
  const el = document.getElementById(inputId);
  if (el) el.click();
}

function setupDragDrop() {
  const card = document.querySelector('.gemini-input-card');
  if (!card) return;

  ['dragover', 'dragenter'].forEach(ev =>
    card.addEventListener(ev, e => {
      e.preventDefault();
      card.classList.add('drag-active');
    }));

  ['dragleave', 'dragend'].forEach(ev =>
    card.addEventListener(ev, e => {
      if (!card.contains(e.relatedTarget)) card.classList.remove('drag-active');
    }));

  card.addEventListener('drop', e => {
    e.preventDefault();
    card.classList.remove('drag-active');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function handleFile(file) {
  const isImg = fileIsImage(file);
  const isVid = fileIsVideo(file);
  if (!isImg && !isVid) {
    showError(`Unsupported file: <b>${file.name}</b>. Use JPG/PNG for images or MP4/MOV for videos.`);
    return;
  }
  // Switch mode WITHOUT opening file picker
  if (isImg && currentMode !== 'image') selectMode('image', false);
  if (isVid && currentMode !== 'video') selectMode('video', false);
  selectedFile = file;
  showGeminiPreview(file, isImg);
  closeAttachMenu();
}

function showGeminiPreview(file, isImage) {
  const strip  = document.getElementById('gemini-preview-strip');
  const fname  = document.getElementById('gemini-file-name');
  const img    = document.getElementById('gemini-preview-img');
  const icon   = document.getElementById('gemini-file-icon');
  const ph     = document.getElementById('gemini-placeholder');

  if (strip)  strip.style.display  = 'block';
  if (fname)  fname.textContent    = file.name;
  if (ph)     ph.style.display     = 'none';

  if (isImage) {
    const url = URL.createObjectURL(file);
    lastPreviewUrl = url;
    if (img)  { img.src = url; img.style.display = 'block'; }
    if (icon) icon.style.display = 'none';
  } else {
    lastPreviewUrl = null;
    if (img)  img.style.display  = 'none';
    if (icon) icon.style.display = 'flex';
  }
}

function removeFile() {
  selectedFile   = null;
  lastPreviewUrl = null;
  const strip  = document.getElementById('gemini-preview-strip');
  const img    = document.getElementById('gemini-preview-img');
  const icon   = document.getElementById('gemini-file-icon');
  const ph     = document.getElementById('gemini-placeholder');
  if (strip) strip.style.display  = 'none';
  if (img)   { img.src = ''; img.style.display = 'none'; }
  if (icon)  icon.style.display = 'none';
  if (ph)    {
    ph.style.display = 'block';
    ph.textContent = currentMode === 'image' ? 'Upload an image to analyse…' : 'Upload a video to analyse…';
  }
  ['file-input-image', 'file-input-video'].forEach(id => {
    const e = document.getElementById(id);
    if (e) e.value = '';
  });
}

// ─────────────────────────────────────────────────────────────
//  SETTINGS
// ─────────────────────────────────────────────────────────────
function saveSettings() {
  const url  = document.getElementById('api-url-input');
  const groq = document.getElementById('groq-key-input');
  if (url)  localStorage.setItem('ds_api_url', url.value);
  if (groq) localStorage.setItem('ds_groq_key', groq.value);
}

function restoreSettings() {
  const url  = document.getElementById('api-url-input');
  const groq = document.getElementById('groq-key-input');
  if (url)  url.value  = localStorage.getItem('ds_api_url')  || '';
  if (groq) groq.value = localStorage.getItem('ds_groq_key') || '';
}

function getGroqKey() {
  const el = document.getElementById('groq-key-input');
  return el ? el.value.trim() : '';
}

// ─────────────────────────────────────────────────────────────
//  API STATUS
// ─────────────────────────────────────────────────────────────
async function checkApiStatus() {
  const ab = document.getElementById('api-status-badge');
  const cb = document.getElementById('cnn-status-badge');
  if (!ab || !cb) return;

  try {
    const r = await fetch(`${FLASK_ORIGIN}/api/status`, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) throw new Error('non-200');
    const d = await r.json();

    if (d.fastapi_ok && d.model_loaded) {
      ab.innerHTML = '<span class="status-dot dot-on"></span><span class="status-text">Flask + FastAPI</span>';
      cb.innerHTML = '<span class="status-dot dot-on"></span><span class="status-text">CNN Loaded</span>';
    } else if (d.flask === 'ok') {
      ab.innerHTML = '<span class="status-dot dot-on"></span><span class="status-text">Online</span>';
      cb.innerHTML = '<span class="status-dot dot-on"></span><span class="status-text">CNN Ready</span>';
    } else {
      ab.innerHTML = '<span class="status-dot dot-warn"></span><span class="status-text">Partial</span>';
      cb.innerHTML = '<span class="status-dot dot-off"></span><span class="status-text">CNN Offline</span>';
    }
  } catch {
    ab.innerHTML = '<span class="status-dot dot-off"></span><span class="status-text">Server Offline</span>';
    cb.innerHTML = '<span class="status-dot dot-off"></span><span class="status-text">Demo Mode</span>';
  }
}

async function flaskIsAlive() {
  try {
    const r = await fetch(`${FLASK_ORIGIN}/api/status`, { signal: AbortSignal.timeout(3000) });
    return r.ok;
  } catch { return false; }
}

// ═════════════════════════════════════════════════════════════
//  MAIN ANALYSIS
//  FIX: Always derive isImage from the actual file, not currentMode.
//       This is the root cause of mobile vs desktop result mismatch.
// ═════════════════════════════════════════════════════════════
async function runAnalysis() {
  if (isAnalysing) return;
  if (!selectedFile) { showError('Please upload a file first.'); return; }

  isAnalysing = true;
  setSendBtn(true);
  saveSettings();
  lastResult = null;

  // CRITICAL FIX: derive isImage from actual file MIME/extension, not currentMode
  const isImage = fileIsImage(selectedFile);
  const isVideo = fileIsVideo(selectedFile);

  // Sync the mode pill to match actual file type (no picker trigger)
  if (isImage && currentMode !== 'image') selectMode('image', false);
  if (isVideo && currentMode !== 'video') selectMode('video', false);

  addUserMessage(
    `<div style="font-size:0.78rem;color:var(--text-2);margin-bottom:4px">${isImage ? 'Image' : 'Video'} · ${(selectedFile.size / 1e6).toFixed(2)} MB</div>` +
    `<div style="font-weight:500">${selectedFile.name}</div>`
  );

  if (lastPreviewUrl && isImage) {
    addBotMessage(`<img src="${lastPreviewUrl}" class="analysis-img" alt="uploaded" />`);
  }

  let typingId = addTyping('Checking server…');
  let result   = null;
  let isDemo   = false;

  try {
    const alive = await flaskIsAlive();

    if (alive) {
      removeTyping(typingId);
      typingId = addTyping(isImage ? 'Running CNN model…' : 'Extracting frames… (30–90s)');
      const fd = new FormData();
      fd.append('file', selectedFile, selectedFile.name);
      let res;
      try {
        res    = await fetch(`${FLASK_ORIGIN}/api/analyse/${isImage ? 'image' : 'video'}`, { method: 'POST', body: fd });
        result = await res.json();
      } catch (err) {
        removeTyping(typingId);
        addBotMessage(`<div class="demo-banner"><div class="demo-banner-title">Network error</div><div style="font-size:0.78rem;color:var(--text-2)">${err.message}. Running demo simulation…</div></div>`);
        result = buildDemoResult(isImage); isDemo = true;
      }
      if (!isDemo && (!res.ok || result.error)) {
        removeTyping(typingId);
        addBotMessage(`<span style="color:var(--fake)">Server error: ${result.error || res.statusText}. Falling back to demo…</span>`);
        result = buildDemoResult(isImage); isDemo = true;
      }
      // Normalise server response — trust verdict, only fill gaps
      if (!isDemo) {
        result = normaliseServerResult(result, isImage);
      }
    } else {
      removeTyping(typingId);
      addBotMessage(`
        <div class="demo-banner">
          <div class="demo-banner-title">⚡ Demo mode</div>
          <div style="font-size:0.8rem;color:var(--text-2)">
            Flask server not found. Start it with <code>python flask_app.py</code><br>
            Showing simulated results below.
          </div>
        </div>`);
      typingId = addTyping('Simulating CNN analysis…');
      await sleep(1400);
      result = buildDemoResult(isImage); isDemo = true;
    }

    removeTyping(typingId);
    lastResult = result;
    await renderResult(result, isImage, isDemo);
    pushHistory(selectedFile, result);
    renderHistory();

  } catch (err) {
    removeTyping(typingId);
    addBotMessage(`<span style="color:var(--fake)">Unexpected error: ${err.message}</span>`);
    console.error('[DeepScan]', err);
  } finally {
    isAnalysing = false;
    setSendBtn(false);
  }
}

// ─────────────────────────────────────────────────────────────
//  DEMO BUILDER — self-consistent, verdict always matches fake_prob
// ─────────────────────────────────────────────────────────────
function buildDemoResult(isImage) {
  // Avoid ambiguous values near threshold
  let fakeProbRaw;
  do { fakeProbRaw = Math.random(); } while (fakeProbRaw > 0.42 && fakeProbRaw < 0.58);

  const isFake  = fakeProbRaw >= FAKE_THRESHOLD;
  const verdict = isFake ? 'fake' : 'real';
  const cnnConf = isFake ? fakeProbRaw : (1.0 - fakeProbRaw);

  const frameResults = isImage ? null : Array.from({ length: 12 }, (_, i) => {
    const baseBias = isFake ? 0.65 : 0.28;
    let fp = clamp(baseBias + (Math.random() - 0.5) * 0.5, 0.05, 0.95);
    const fv = fp >= FAKE_THRESHOLD ? 'fake' : 'real';
    const fc = fv === 'fake' ? fp : (1 - fp);
    return {
      frame_index:       i * 2,
      timestamp_sec:     parseFloat((i * 1.4).toFixed(2)),
      cnn_label:         fv,
      cnn_confidence:    parseFloat(fc.toFixed(4)),
      fake_prob:         parseFloat(fp.toFixed(4)),
      verdict:           fv,
      thumbnail_b64:     '',
      frame_explanation: fv === 'fake'
        ? `Frame ${i * 2}: CNN detected synthetic patterns with ${(fp * 100).toFixed(0)}% fake probability.`
        : `Frame ${i * 2}: CNN found no significant deepfake artefacts.`
    };
  });

  let overallFakeProb = fakeProbRaw;
  if (frameResults) {
    overallFakeProb = frameResults.reduce((s, f) => s + f.fake_prob, 0) / frameResults.length;
  }
  const finalVerdict = overallFakeProb >= FAKE_THRESHOLD ? 'fake' : 'real';
  const fakeCount    = frameResults ? frameResults.filter(f => f.verdict === 'fake').length : null;

  return {
    media_type:            isImage ? 'image' : 'video',
    verdict:               finalVerdict,
    fake_probability:      parseFloat(overallFakeProb.toFixed(4)),
    confidence_pct:        parseFloat((overallFakeProb * 100).toFixed(1)),
    cnn_label:             finalVerdict,
    cnn_confidence:        parseFloat(cnnConf.toFixed(4)),
    frame_results:         frameResults,
    fake_frame_count:      fakeCount,
    total_frames_analysed: frameResults ? frameResults.length : null,
    message:               `[DEMO] ${finalVerdict.toUpperCase()} — ${(overallFakeProb * 100).toFixed(1)}% fake probability`,
    _demo: true
  };
}

// ═════════════════════════════════════════════════════════════
//  RENDER RESULT
// ═════════════════════════════════════════════════════════════
async function renderResult(result, isImage, isDemo) {
  const verdict  = result.verdict  || 'unknown';
  const fakeProb = parseFloat(result.fake_probability) || 0;
  const isFake   = verdict === 'fake';
  const demoTag  = isDemo
    ? `<span style="background:rgba(251,188,4,0.15);color:var(--warn);font-size:0.64rem;padding:2px 8px;border-radius:50px;margin-left:8px;font-weight:600">DEMO</span>`
    : '';

  addBotMessage(`
    <div class="verdict-banner ${isFake ? 'fake' : 'real'}">
      <div class="verdict-icon">${isFake ? '⚠️' : '✅'}</div>
      <div>
        <div class="verdict-label">${isFake ? 'Fake — AI-generated / deepfake' : 'Real — authentic content'}${demoTag}</div>
        <div class="verdict-sub">${(fakeProb * 100).toFixed(1)}% fake probability</div>
      </div>
    </div>
    <div class="prob-section">
      <div class="prob-header">
        <span>Fake probability</span>
        <span style="font-weight:600;color:${isFake ? 'var(--fake)' : 'var(--real)'}">${(fakeProb * 100).toFixed(1)}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${isFake ? 'is-fake' : 'is-real'}" style="width:${clamp(fakeProb * 100, 0, 100).toFixed(1)}%"></div>
      </div>
    </div>
    <div class="chip-grid">
      <div class="chip">
        <div class="chip-label">Verdict</div>
        <div class="chip-value ${isFake ? 'is-fake' : 'is-real'}">${verdict.toUpperCase()}</div>
      </div>
      <div class="chip">
        <div class="chip-label">CNN label</div>
        <div class="chip-value is-neutral">${(result.cnn_label || '–').toUpperCase()}</div>
      </div>
      <div class="chip">
        <div class="chip-label">CNN confidence</div>
        <div class="chip-value is-neutral">${result.cnn_confidence != null ? pct(result.cnn_confidence).toFixed(1) + '%' : '–'}</div>
      </div>
      <div class="chip">
        <div class="chip-label">Threshold</div>
        <div class="chip-value is-neutral">50.0%</div>
      </div>
      ${!isImage && result.fake_frame_count != null ? `
      <div class="chip">
        <div class="chip-label">Fake frames</div>
        <div class="chip-value ${isFake ? 'is-fake' : 'is-real'}">${result.fake_frame_count} / ${result.total_frames_analysed}</div>
      </div>` : ''}
    </div>`);

  if (isImage) {
    const previewUrl = lastPreviewUrl || '';
    const cnnConf    = result.cnn_confidence != null ? pct(result.cnn_confidence).toFixed(1) + '%' : '–';

    addBotMessage(`
      <div class="signal-wrap">
        ${previewUrl ? `
          <div>
            <div style="font-size:0.7rem;color:var(--text-3);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em">Analysed image</div>
            <img src="${previewUrl}" class="analysis-img" style="max-width:200px;border:2px solid ${isFake ? 'var(--fake)' : 'var(--real)'}" alt="analysed" />
          </div>` : ''}
        <div class="signal-table">
          <table>
            <thead><tr><th>Source</th><th>Score</th><th>Role</th></tr></thead>
            <tbody>
              <tr>
                <td>CNN model</td>
                <td class="${result.cnn_label === 'fake' ? 'td-fake' : 'td-real'}">${cnnConf}</td>
                <td style="color:var(--text-3)">Sole signal</td>
              </tr>
              <tr>
                <td><b>Verdict</b></td>
                <td class="${isFake ? 'td-fake' : 'td-real'}"><b>${(fakeProb * 100).toFixed(1)}%</b></td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
          <div style="padding:8px 12px;font-size:0.68rem;color:var(--text-3)">
            Threshold 0.50 · fake_prob = 1 − sigmoid · ${isFake ? 'Above → Fake' : 'Below → Real'} · CNN-only pipeline
          </div>
        </div>
      </div>`);
  }

  if (!isImage && result.frame_results && result.frame_results.length > 0) {
    addBotMessage(buildFrameGrid(result.frame_results));
    addBotMessage(buildTimeline(result.frame_results));
    addBotMessage(buildFrameTable(result.frame_results));
  }

  const groqKey    = getGroqKey();
  const serverGroq = !groqKey ? await checkServerGroq() : false;

  if (groqKey || serverGroq) {
    if (isImage) await groqExplainImage(verdict, fakeProb, result, groqKey || null);
    else         await groqExplainVideo(verdict, fakeProb, result, groqKey || null);
  } else {
    addBotMessage(`
      <div class="explanation-block">
        <div class="exp-header">AI explanation</div>
        Click the ⚙️ icon and paste your <b>Groq API key</b> to get a detailed AI explanation,
        or add <code>GROQ_API_KEY=…</code> to your Flask <code>.env</code>.
      </div>`);
  }

  const pdfBtnId = 'pdf-btn-' + Date.now();
  addBotMessage(`
    <button class="download-pdf-btn" id="${pdfBtnId}" onclick="downloadPDF('${pdfBtnId}')">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Download Professional PDF Report
    </button>`);
}

async function checkServerGroq() {
  try {
    const r = await fetch(`${FLASK_ORIGIN}/api/status`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) return false;
    const d = await r.json();
    return d.groq_configured === true;
  } catch { return false; }
}

// ═════════════════════════════════════════════════════════════
//  FRAME GRID
// ═════════════════════════════════════════════════════════════
function buildFrameGrid(frames) {
  const cards = frames.map(fr => {
    const isFake = fr.verdict === 'fake';
    const prob   = clamp(parseFloat(fr.fake_prob || 0), 0, 1);
    const thumb  = fr.thumbnail_b64 ? `data:image/jpeg;base64,${fr.thumbnail_b64}` : null;
    return `
      <div class="frame-card ${isFake ? 'fc-fake' : 'fc-real'}">
        ${thumb
          ? `<img src="${thumb}" class="frame-thumb" alt="frame ${fr.frame_index}" loading="lazy" />`
          : `<div class="frame-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--text-3);font-size:0.7rem">${fr.timestamp_sec}s</div>`}
        <div class="frame-info">
          <span class="frame-badge ${isFake ? 'fake' : 'real'}">${isFake ? 'Fake' : 'Real'}</span><br>
          <span class="frame-stat">⏱ ${fr.timestamp_sec}s · ${(prob * 100).toFixed(0)}%</span>
          <div class="frame-bar"><div class="frame-bar-fill ${isFake ? 'fake' : 'real'}" style="width:${(prob * 100).toFixed(0)}%"></div></div>
        </div>
      </div>`;
  }).join('');
  return `
    <div>
      <div class="frames-header">Frame-by-frame CNN analysis — ${frames.length} frames sampled</div>
      <div class="frame-grid">${cards}</div>
    </div>`;
}

// ═════════════════════════════════════════════════════════════
//  TIMELINE
// ═════════════════════════════════════════════════════════════
function buildTimeline(frames) {
  const id    = 'tl_' + Math.random().toString(36).slice(2, 8);
  const ts    = frames.map(f => f.timestamp_sec);
  const probs = frames.map(f => clamp(parseFloat(f.fake_prob || 0), 0, 1));
  return `
    <div class="timeline-wrap">
      <div class="timeline-title">Fake probability timeline (CNN)</div>
      <canvas class="timeline-chart" id="${id}" data-chart='${JSON.stringify({ ts, probs })}'></canvas>
    </div>`;
}

function drawTimeline(canvas) {
  requestAnimationFrame(() => {
    try {
      const { ts, probs } = JSON.parse(canvas.dataset.chart);
      if (!ts || ts.length < 2) return;
      const W = canvas.parentElement.clientWidth || 600;
      const H = 110;
      canvas.width = W; canvas.height = H;
      const ctx = canvas.getContext('2d');
      const pL = 44, pR = 12, pT = 12, pB = 28;
      const cw = W - pL - pR, ch = H - pT - pB;
      const xO = t => pL + ((t - ts[0]) / (ts[ts.length - 1] - ts[0] || 1)) * cw;
      const yO = p => pT + ch * (1 - p);
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pT + (ch / 4) * i;
        ctx.beginPath(); ctx.moveTo(pL, y); ctx.lineTo(pL + cw, y); ctx.stroke();
      }
      const THR = FAKE_THRESHOLD;
      const thrY = yO(THR);
      ctx.strokeStyle = 'rgba(251,188,4,0.5)'; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(pL, thrY); ctx.lineTo(pL + cw, thrY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(251,188,4,0.7)'; ctx.font = '9px JetBrains Mono,monospace';
      ctx.fillText('50%', pL + 4, thrY - 4);
      const g = ctx.createLinearGradient(0, pT, 0, pT + ch);
      g.addColorStop(0, 'rgba(234,67,53,0.3)'); g.addColorStop(1, 'rgba(234,67,53,0.01)');
      ctx.beginPath(); ctx.moveTo(xO(ts[0]), yO(probs[0]));
      ts.forEach((t, i) => ctx.lineTo(xO(t), yO(probs[i])));
      ctx.lineTo(xO(ts[ts.length - 1]), pT + ch); ctx.lineTo(xO(ts[0]), pT + ch);
      ctx.closePath(); ctx.fillStyle = g; ctx.fill();
      ctx.beginPath(); ctx.moveTo(xO(ts[0]), yO(probs[0]));
      ts.forEach((t, i) => ctx.lineTo(xO(t), yO(probs[i])));
      ctx.strokeStyle = '#ea4335'; ctx.lineWidth = 1.8; ctx.stroke();
      ts.forEach((t, i) => {
        ctx.beginPath(); ctx.arc(xO(t), yO(probs[i]), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = probs[i] >= THR ? '#ea4335' : '#34a853'; ctx.fill();
      });
      ctx.fillStyle = 'rgba(154,160,166,0.8)'; ctx.textAlign = 'right'; ctx.font = '9px JetBrains Mono,monospace';
      for (let i = 0; i <= 4; i++) ctx.fillText(`${100 - i * 25}%`, pL - 4, pT + (ch / 4) * i + 3);
      ctx.textAlign = 'center';
      [0, Math.floor(ts.length / 2), ts.length - 1].forEach(i => {
        if (ts[i] !== undefined) ctx.fillText(`${ts[i]}s`, xO(ts[i]), H - 6);
      });
    } catch (e) { console.warn('[DeepScan] canvas draw:', e); }
  });
}

// ═════════════════════════════════════════════════════════════
//  FRAME TABLE
// ═════════════════════════════════════════════════════════════
function buildFrameTable(frames) {
  const rows = frames.map(fr => {
    const isFake = fr.verdict === 'fake';
    const prob   = clamp(parseFloat(fr.fake_prob || 0), 0, 1);
    return `<tr>
      <td style="color:var(--text-3)">${fr.frame_index}</td>
      <td>${fr.timestamp_sec}s</td>
      <td class="${isFake ? 'td-fake' : 'td-real'}">${isFake ? 'Fake' : 'Real'}</td>
      <td>${fr.cnn_confidence != null ? pct(fr.cnn_confidence).toFixed(1) + '%' : '–'}</td>
      <td class="${isFake ? 'td-fake' : 'td-real'}">${(prob * 100).toFixed(1)}%</td>
    </tr>`;
  }).join('');
  return `
    <details class="frame-table-details">
      <summary>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
        Show full frame table (${frames.length} frames)
      </summary>
      <div style="overflow-x:auto;margin-top:8px">
        <table class="detail-table">
          <thead><tr><th>Frame</th><th>Time</th><th>Verdict</th><th>CNN Conf.</th><th>Fake Prob.</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>`;
}

// ═════════════════════════════════════════════════════════════
//  GROQ — IMAGE
// ═════════════════════════════════════════════════════════════
async function groqExplainImage(verdict, fakeProb, result, groqKey) {
  const prompt =
`You are a computer-vision deepfake expert. IMAGE RESULT (CNN-only pipeline):
- Verdict: ${verdict.toUpperCase()}, fake probability: ${(fakeProb * 100).toFixed(1)}%
- Model: MobileNetV2 CNN (deepfake_model.h5), sigmoid output inverted: fake_prob = 1 - raw_sigmoid
- CNN label: ${(result.cnn_label || '?').toUpperCase()} at ${result.cnn_confidence != null ? pct(result.cnn_confidence).toFixed(1) + '%' : 'unknown'} confidence
- Decision threshold: 50%
Write 3-5 sentences plain prose: state verdict and confidence, which CNN signals likely triggered this classification, likely visual artefacts the model would detect, and caveats if confidence < 75%.`;
  const mid = addBotMessage(`<div class="explanation-block"><div class="exp-header">AI Analysis — Image explanation</div><span class="spinner"></span> Querying Groq…</div>`);
  const text = await callGroq(prompt, groqKey);
  updateMessage(mid, `<div class="explanation-block"><div class="exp-header">AI Analysis — Image explanation</div><div style="margin-top:8px">${text.replace(/\n/g, '<br>')}</div></div>`);
}

// ═════════════════════════════════════════════════════════════
//  GROQ — VIDEO
// ═════════════════════════════════════════════════════════════
async function groqExplainVideo(verdict, fakeProb, result, groqKey) {
  const frames     = result.frame_results || [];
  const fakeCount  = result.fake_frame_count ?? frames.filter(f => f.verdict === 'fake').length;
  const totalCount = result.total_frames_analysed ?? frames.length;
  const prompt =
`You are a deepfake expert. VIDEO result (CNN-only pipeline):
- Overall verdict: ${verdict.toUpperCase()}, avg fake prob: ${(fakeProb * 100).toFixed(1)}%
- Fake frames: ${fakeCount}/${totalCount}
- Model: MobileNetV2 CNN — fake_prob = 1 − sigmoid, threshold 50%
- CNN label: ${(result.cnn_label || '?').toUpperCase()} at ${result.cnn_confidence != null ? pct(result.cnn_confidence).toFixed(1) + '%' : 'N/A'}
Write 4-6 sentences plain prose: state verdict/confidence, what the fake frame ratio indicates, likely artefacts detected per-frame, and caveats about CNN-only analysis.`;
  const mid = addBotMessage(`<div class="explanation-block"><div class="exp-header">AI Analysis — Video verdict</div><span class="spinner"></span> Generating…</div>`);
  const text = await callGroq(prompt, groqKey);
  updateMessage(mid, `<div class="explanation-block"><div class="exp-header">AI Analysis — Video verdict</div><div style="margin-top:8px">${text.replace(/\n/g, '<br>')}</div></div>`);

  const topFakes = [...frames]
    .filter(f => f.verdict === 'fake')
    .sort((a, b) => parseFloat(b.fake_prob || 0) - parseFloat(a.fake_prob || 0))
    .slice(0, 5);
  if (!topFakes.length) return;

  const fid = addBotMessage(`<div class="explanation-block"><div class="exp-header">AI Analysis — Top ${topFakes.length} suspicious frames</div><span class="spinner"></span> Analysing…</div>`);
  const cards = [];
  for (const fr of topFakes) {
    const prob  = clamp(parseFloat(fr.fake_prob || 0), 0, 1);
    const thumb = fr.thumbnail_b64 ? `data:image/jpeg;base64,${fr.thumbnail_b64}` : null;
    const fp2 =
`Frame ${fr.frame_index} @ ${fr.timestamp_sec}s: FAKE, CNN fake_prob ${(prob * 100).toFixed(1)}%, CNN label ${(fr.cnn_label || '?').toUpperCase()} @ ${fr.cnn_confidence != null ? pct(fr.cnn_confidence).toFixed(1) + '%' : 'N/A'}. In 2-3 sentences explain why this frame is flagged. Be specific about artefacts.`;
    const ft = await callGroq(fp2, groqKey);
    cards.push(`
      <div style="display:flex;gap:12px;padding:12px 0;border-bottom:1px solid var(--border);align-items:flex-start">
        ${thumb
          ? `<img src="${thumb}" style="width:80px;min-width:80px;border-radius:8px;border:2px solid var(--fake);object-fit:cover" alt="fr${fr.frame_index}" />`
          : `<div style="width:80px;min-width:80px;height:52px;border-radius:8px;border:2px solid var(--fake);background:var(--bg3);display:flex;align-items:center;justify-content:center;font-size:0.65rem;color:var(--fake)">Fr.${fr.frame_index}</div>`}
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap">
            <span style="background:rgba(234,67,53,0.15);color:var(--fake);font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:50px">Frame ${fr.frame_index}</span>
            <span style="font-size:0.65rem;color:var(--text-3)">⏱ ${fr.timestamp_sec}s · ${(prob * 100).toFixed(1)}% fake</span>
          </div>
          <div style="font-size:0.82rem;line-height:1.75">${ft.replace(/\n/g, '<br>')}</div>
          <div class="frame-bar" style="margin-top:8px"><div class="frame-bar-fill fake" style="width:${(prob * 100).toFixed(0)}%"></div></div>
        </div>
      </div>`);
  }
  updateMessage(fid, `<div class="explanation-block"><div class="exp-header">AI Analysis — Top ${topFakes.length} suspicious frames</div>${cards.join('')}</div>`);
}

// ═════════════════════════════════════════════════════════════
//  GROQ CALLER
// ═════════════════════════════════════════════════════════════
async function callGroq(prompt, groqKey) {
  if (!groqKey) {
    try {
      const r = await fetch(`${FLASK_ORIGIN}/api/groq/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 460 }),
        signal: AbortSignal.timeout(30000)
      });
      const d = await r.json();
      if (!r.ok) return `⚠ Groq server error: ${d.error || r.statusText}`;
      return d.text || '⚠ Empty response.';
    } catch (e) { return `⚠ Flask unreachable: ${e.message}`; }
  }
  const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
  const models   = ['llama-3.1-8b-instant', 'llama3-70b-8192'];
  for (const model of models) {
    try {
      const r = await fetch(GROQ_URL, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${groqKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: 460, temperature: 0.4 }),
        signal: AbortSignal.timeout(28000)
      });
      if (r.status === 400 || r.status === 404) continue;
      if (!r.ok) {
        if (r.status === 401) return '⚠ Invalid Groq API key.';
        if (r.status === 429) return '⚠ Groq rate limit — wait a moment.';
        const e = await r.json().catch(() => ({}));
        return `⚠ Groq ${r.status}: ${e?.error?.message || r.statusText}`;
      }
      const d = await r.json();
      return d.choices?.[0]?.message?.content?.trim() || '⚠ Empty Groq response.';
    } catch (e) {
      if (models.indexOf(model) < models.length - 1) continue;
      return `⚠ Network error: ${e.message}`;
    }
  }
  return '⚠ All Groq models unavailable.';
}

// ═════════════════════════════════════════════════════════════
//  PDF DOWNLOAD
// ═════════════════════════════════════════════════════════════
async function groqWriteSection(prompt, groqKey, fallback = '') {
  if (!groqKey) {
    try {
      const r = await fetch(`${FLASK_ORIGIN}/api/groq/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 500 }),
        signal: AbortSignal.timeout(30000)
      });
      const d = await r.json();
      if (r.ok && d.text) return d.text;
    } catch (_) {}
    return fallback;
  }
  const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
  const models   = ['llama-3.1-8b-instant', 'llama3-70b-8192'];
  for (const model of models) {
    try {
      const r = await fetch(GROQ_URL, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${groqKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: 500, temperature: 0.4 }),
        signal: AbortSignal.timeout(28000)
      });
      if (r.status === 400 || r.status === 404) continue;
      if (!r.ok) return fallback;
      const d = await r.json();
      const txt = d.choices?.[0]?.message?.content?.trim();
      if (txt) return txt;
    } catch (_) { if (models.indexOf(model) < models.length - 1) continue; }
  }
  return fallback;
}

async function loadImageForPDF(src) {
  return new Promise(resolve => {
    if (!src) return resolve(null);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const c = document.createElement('canvas');
        c.width = img.naturalWidth || img.width;
        c.height = img.naturalHeight || img.height;
        c.getContext('2d').drawImage(img, 0, 0);
        resolve({ dataUrl: c.toDataURL('image/jpeg', 0.88), w: c.width, h: c.height });
      } catch { resolve(null); }
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

async function downloadPDF(btnId) {
  const btn = document.getElementById(btnId);
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Generating PDF...'; }
  try {
    const { jsPDF } = window.jspdf;
    if (!jsPDF) throw new Error('jsPDF not loaded');
    const result  = lastResult || {};
    const isImage = result.media_type === 'image' || !result.frame_results;
    if (isImage) await buildImagePDF(jsPDF, result);
    else         await buildVideoPDF(jsPDF, result);
  } catch (err) {
    console.error('[DeepScan] PDF error:', err);
    showError('PDF generation failed: ' + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download Professional PDF Report`;
    }
  }
}

function palette() {
  return {
    bg0:[10,11,14],bg1:[20,22,28],bg2:[28,30,36],bg3:[36,38,44],hdr:[14,15,20],
    fake:[220,53,53],real:[40,167,69],blue:[66,133,244],warn:[240,173,78],
    t0:[230,232,235],t1:[170,175,182],t2:[100,107,118],border:[42,44,50],accent:[50,52,70]
  };
}

function makeHelpers(doc, C, W, M) {
  function safeStr(val, fallback) {
    if (val === null || val === undefined) return fallback !== undefined ? String(fallback) : '';
    return String(val);
  }
  function T(text, x, y, opts) {
    const str = (text === null || text === undefined) ? '' : String(text);
    if (opts) doc.text(str, x, y, opts);
    else      doc.text(str, x, y);
  }
  function fillBg() { doc.setFillColor(...C.bg0); doc.rect(0, 0, W, 297, 'F'); }
  function sectionHead(label, y) {
    doc.setFillColor(...C.bg1); doc.rect(M, y, W - M * 2, 7, 'F');
    doc.setFillColor(...C.blue); doc.rect(M, y, 2, 7, 'F');
    doc.setTextColor(...C.blue); doc.setFontSize(7); doc.setFont('helvetica', 'bold');
    doc.text(label, M + 5, y + 4.8);
    return y + 10;
  }
  function chip(x, y, w, h, label, value, valColor) {
    doc.setFillColor(...C.bg1); doc.roundedRect(x, y, w, h, 3, 3, 'F');
    doc.setDrawColor(...C.border); doc.setLineWidth(0.25); doc.roundedRect(x, y, w, h, 3, 3, 'S');
    doc.setTextColor(...C.t2); doc.setFontSize(5.5); doc.setFont('helvetica', 'bold');
    doc.text(label.toUpperCase(), x + w / 2, y + 5, { align: 'center' });
    doc.setTextColor(...valColor); doc.setFontSize(9.5); doc.setFont('helvetica', 'bold');
    doc.text(value, x + w / 2, y + 13, { align: 'center' });
  }
  function prose(text, x, y, maxW, fontSize, color, bold = false) {
    doc.setTextColor(...color); doc.setFontSize(fontSize);
    doc.setFont('helvetica', bold ? 'bold' : 'normal');
    const lines = doc.splitTextToSize(text || '', maxW);
    doc.text(lines, x, y);
    return y + lines.length * (fontSize * 0.4 + 1.6);
  }
  function hr(y) {
    doc.setDrawColor(...C.border); doc.setLineWidth(0.25);
    doc.line(M, y, W - M, y); return y + 4;
  }
  function footer(pageNum, totalPages) {
    doc.setFillColor(10, 11, 15); doc.rect(0, 287, W, 10, 'F');
    doc.setDrawColor(...C.border); doc.setLineWidth(0.25); doc.line(0, 287, W, 287);
    doc.setTextColor(...C.t2); doc.setFontSize(6.5); doc.setFont('helvetica', 'normal');
    doc.text('DeepScan -- AI Deepfake Detection  |  CNN-only pipeline  |  For informational purposes only', M, 293);
    doc.setTextColor(...C.blue);
    doc.text(`Page ${pageNum} of ${totalPages}`, W - M, 293, { align: 'right' });
  }
  return { fillBg, sectionHead, chip, prose, hr, footer, safeStr, T };
}

// ═════════════════════════════════════════════════════════════
//  IMAGE PDF
// ═════════════════════════════════════════════════════════════
async function buildImagePDF(jsPDF, result) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const C = palette(); const W = 210, M = 16;
  const { fillBg, sectionHead, chip, prose, hr, footer } = makeHelpers(doc, C, W, M);
  const verdict = result.verdict || 'unknown'; const fakeProb = parseFloat(result.fake_probability) || 0;
  const isFake = verdict === 'fake'; const isDemo = !!result._demo; const verdictC = isFake ? C.fake : C.real;
  const fileName = selectedFile ? selectedFile.name : 'unknown'; const now = new Date().toLocaleString();
  const groqKey = getGroqKey();
  const introPrompt = `You are a professional forensic AI analyst writing a formal deepfake detection report. Pipeline: CNN-only (MobileNetV2). fake_prob = 1 - raw_sigmoid. Threshold: 50%. Verdict: ${verdict.toUpperCase()}. Fake probability: ${(fakeProb * 100).toFixed(1)}%. Write exactly 3 professional plain-prose sentences: 1. State the verdict and confidence. 2. Describe what CNN signals drove this result. 3. Add one sentence of caveats.`;
  const introCopy = await groqWriteSection(introPrompt, groqKey, `This image has been assessed as ${verdict.toUpperCase()} with a fake probability of ${(fakeProb * 100).toFixed(1)}% by the CNN pipeline. The MobileNetV2 model analysed pixel-level patterns and produced an inverted sigmoid score. Results should be interpreted alongside human review for high-stakes decisions.`);
  const techPrompt = `CNN label: ${(result.cnn_label || '?').toUpperCase()}, confidence ${result.cnn_confidence != null ? (result.cnn_confidence * 100).toFixed(1) + '%' : 'unknown'}. Fake prob: ${(fakeProb * 100).toFixed(1)}%. Threshold: 50%. Write 3-4 plain-prose sentences explaining the CNN signal, likely visual artefacts, and the inverted-sigmoid scoring.`;
  const techCopy = await groqWriteSection(techPrompt, groqKey, `The CNN model evaluated pixel-level patterns using a MobileNetV2 architecture. The raw sigmoid output was inverted (fake_prob = 1 − sigmoid) to derive the fake probability of ${(fakeProb * 100).toFixed(1)}%. This ${isFake ? 'exceeds' : 'falls below'} the 50% decision threshold.`);
  const recPrompt = `An image classified as ${verdict.toUpperCase()} (${(fakeProb * 100).toFixed(1)}% fake prob). Write exactly 3 short, actionable plain-prose sentences recommending next steps.`;
  const recCopy = await groqWriteSection(recPrompt, groqKey, `Cross-verify this image with a reverse image search and additional forensic tools. If intended for publication or legal use, obtain a certified forensic review. A single-model CNN result should not be treated as conclusive.`);
  fillBg();
  doc.setFillColor(...C.hdr); doc.rect(0, 0, W, 42, 'F');
  doc.setFillColor(...verdictC); doc.rect(0, 0, 4, 42, 'F');
  doc.setTextColor(...C.blue); doc.setFontSize(24); doc.setFont('helvetica', 'bold'); doc.text('DeepScan', M, 16);
  doc.setTextColor(...C.t2); doc.setFontSize(8); doc.setFont('helvetica', 'normal'); doc.text('AI Image Authenticity Report  —  CNN-Only Pipeline', M, 24);
  doc.setFontSize(7.5); doc.text(now, W - M, 13, { align: 'right' });
  if (isDemo) { doc.setFillColor(...C.warn); doc.roundedRect(W - M - 24, 17, 24, 7, 2, 2, 'F'); doc.setTextColor(20,20,20); doc.setFontSize(6.5); doc.setFont('helvetica','bold'); doc.text('DEMO', W - M - 12, 22, { align:'center' }); }
  doc.setFillColor(...(isFake ? [55,18,18] : [15,50,25])); doc.roundedRect(M, 29, 110, 9, 2.5, 2.5, 'F');
  doc.setDrawColor(...verdictC); doc.setLineWidth(0.5); doc.roundedRect(M, 29, 110, 9, 2.5, 2.5, 'S');
  doc.setTextColor(...verdictC); doc.setFontSize(9); doc.setFont('helvetica','bold');
  doc.text(isFake ? 'FAKE DETECTED — AI-generated or manipulated' : 'AUTHENTIC — No deepfake indicators found', M + 4, 35);
  let y = 48;
  doc.setFillColor(...C.bg1); doc.roundedRect(M, y, W - M * 2, 16, 3, 3, 'F');
  doc.setDrawColor(...C.border); doc.setLineWidth(0.2); doc.roundedRect(M, y, W - M * 2, 16, 3, 3, 'S');
  [['FILE', M+4],['TYPE', M+90],['FAKE PROBABILITY', M+132]].forEach(([lbl,x]) => { doc.setTextColor(...C.t2); doc.setFontSize(6); doc.setFont('helvetica', 'bold'); doc.text(lbl,x,y+5.5); });
  const shortName = fileName.length > 42 ? fileName.slice(0,39)+'...' : fileName;
  doc.setTextColor(...C.t0); doc.setFontSize(8.5); doc.setFont('helvetica','normal'); doc.text(shortName, M+4, y+12); doc.text('IMAGE', M+90, y+12);
  doc.setTextColor(...verdictC); doc.setFont('helvetica','bold'); doc.text(`${(fakeProb*100).toFixed(1)}%`, M+132, y+12);
  y += 22;
  y = sectionHead('ANALYSED IMAGE', y);
  const imgData = lastPreviewUrl ? await loadImageForPDF(lastPreviewUrl) : null;
  if (imgData) {
    const maxW=82,maxH=72,ratio=imgData.w/imgData.h; let dW=maxW,dH=maxW/ratio; if(dH>maxH){dH=maxH;dW=maxH*ratio;}
    doc.setDrawColor(...verdictC); doc.setLineWidth(0.8); doc.rect(M,y,dW,dH); doc.addImage(imgData.dataUrl,'JPEG',M,y,dW,dH);
    const bx=M+dW+6,bw=W-M-bx; doc.setFillColor(...(isFake?[50,15,15]:[15,44,22])); doc.roundedRect(bx,y,bw,dH,4,4,'F');
    doc.setDrawColor(...verdictC); doc.setLineWidth(0.4); doc.roundedRect(bx,y,bw,dH,4,4,'S');
    let by=y+dH*0.18; doc.setTextColor(...verdictC); doc.setFontSize(18); doc.setFont('helvetica','bold'); doc.text(isFake?'FAKE':'REAL',bx+bw/2,by,{align:'center'}); by+=9;
    doc.setTextColor(...C.t1); doc.setFontSize(7); doc.setFont('helvetica','normal'); doc.text('Fake probability',bx+bw/2,by,{align:'center'}); by+=6;
    doc.setTextColor(...verdictC); doc.setFontSize(18); doc.setFont('helvetica','bold'); doc.text(`${(fakeProb*100).toFixed(1)}%`,bx+bw/2,by,{align:'center'}); by+=10;
    const barX=bx+6,barW=bw-12; doc.setFillColor(...C.bg3); doc.roundedRect(barX,by,barW,4.5,2,2,'F'); doc.setFillColor(...verdictC); doc.roundedRect(barX,by,Math.max(2,fakeProb*barW),4.5,2,2,'F'); by+=10;
    doc.setTextColor(...C.t2); doc.setFontSize(6); doc.text(`${imgData.w} x ${imgData.h} px`,bx+bw/2,by,{align:'center'});
    y+=dH+8;
  }
  y=sectionHead('EXECUTIVE SUMMARY',y);
  doc.setFillColor(18,22,36); doc.roundedRect(M,y,W-M*2,36,3,3,'F');
  doc.setDrawColor(...C.border); doc.setLineWidth(0.25); doc.roundedRect(M,y,W-M*2,36,3,3,'S');
  y=prose(introCopy,M+5,y+7,W-M*2-10,7.5,C.t1)+6;
  y+=4; y=sectionHead('FAKE PROBABILITY SCORE',y);
  const barFull=W-M*2; doc.setFillColor(...C.bg3); doc.roundedRect(M,y,barFull,6,3,3,'F'); doc.setFillColor(...verdictC); doc.roundedRect(M,y,Math.max(4,fakeProb*barFull),6,3,3,'F');
  const thrX=M+FAKE_THRESHOLD*barFull; doc.setDrawColor(...C.warn); doc.setLineWidth(0.6); doc.line(thrX,y-1,thrX,y+7); doc.setTextColor(...C.warn); doc.setFontSize(6.5); doc.text('50% threshold',thrX,y+11,{align:'center'}); doc.setTextColor(...C.t2); doc.setFontSize(6.5); doc.text('0%',M,y+11); doc.text('100%',W-M,y+11,{align:'right'}); y+=18;
  y=sectionHead('DETECTION METRICS',y);
  const metrics=[{label:'Verdict',value:verdict.toUpperCase(),color:verdictC},{label:'CNN Label',value:(result.cnn_label||'-').toUpperCase(),color:C.t0},{label:'CNN Confidence',value:result.cnn_confidence!=null?`${(result.cnn_confidence*100).toFixed(1)}%`:'-',color:C.t0},{label:'Fake Prob.',value:`${(fakeProb*100).toFixed(1)}%`,color:verdictC}];
  const gap=4,chipW=(W-M*2-gap*(metrics.length-1))/metrics.length;
  metrics.forEach((m,i)=>chip(M+i*(chipW+gap),y,chipW,18,m.label,m.value,m.color)); y+=26;
  doc.addPage(); fillBg();
  doc.setFillColor(...C.hdr); doc.rect(0,0,W,16,'F'); doc.setFillColor(...verdictC); doc.rect(0,0,4,16,'F');
  doc.setTextColor(...C.blue); doc.setFontSize(11); doc.setFont('helvetica','bold'); doc.text('DeepScan',M,11);
  doc.setTextColor(...C.t2); doc.setFontSize(7); doc.text('AI Image Authenticity Report  —  CNN-Only  —  '+fileName,M+36,11);
  doc.setTextColor(...verdictC); doc.setFontSize(8); doc.setFont('helvetica','bold'); doc.text(verdict.toUpperCase(),W-M,11,{align:'right'});
  y=22; y=sectionHead('TECHNICAL ANALYSIS',y);
  doc.setFillColor(16,20,34); doc.roundedRect(M,y,W-M*2,42,3,3,'F'); doc.setDrawColor(...C.border); doc.setLineWidth(0.25); doc.roundedRect(M,y,W-M*2,42,3,3,'S');
  doc.setFillColor(...C.blue); doc.rect(M,y+4,2,34,'F');
  y=prose(techCopy,M+6,y+8,W-M*2-12,7.8,C.t1)+6; y=Math.max(y,y+4);
  y=sectionHead('RECOMMENDATIONS & NEXT STEPS',y);
  doc.setFillColor(16,24,20); doc.roundedRect(M,y,W-M*2,36,3,3,'F'); doc.setDrawColor(...C.border); doc.setLineWidth(0.25); doc.roundedRect(M,y,W-M*2,36,3,3,'S');
  doc.setFillColor(...C.real); doc.rect(M,y+4,2,28,'F');
  y=prose(recCopy,M+6,y+8,W-M*2-12,7.8,C.t1);
  const totalPages=doc.internal.getNumberOfPages();
  for(let p=1;p<=totalPages;p++){doc.setPage(p);footer(p,totalPages);}
  const safeName=(selectedFile?selectedFile.name.replace(/\.[^.]+$/,''):'deepscan')+'_image_report.pdf';
  doc.save(safeName);
}

async function decodeThumbnail(b64) {
  if (!b64 || b64.length < 100) return null;
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      try {
        const w=img.naturalWidth||img.width||160,h=img.naturalHeight||img.height||90;
        const c=document.createElement('canvas'); c.width=w; c.height=h; c.getContext('2d').drawImage(img,0,0);
        resolve({dataUrl:c.toDataURL('image/jpeg',0.82),w,h,ratio:w/h});
      } catch { resolve(null); }
    };
    img.onerror=()=>resolve(null);
    img.src='data:image/jpeg;base64,'+b64;
  });
}

// ═════════════════════════════════════════════════════════════
//  VIDEO PDF
// ═════════════════════════════════════════════════════════════
async function buildVideoPDF(jsPDF, result) {
  const doc=new jsPDF({orientation:'portrait',unit:'mm',format:'a4'});
  const C=palette(); const W=210,M=16;
  const {fillBg,sectionHead,chip,prose,hr,footer,safeStr,T}=makeHelpers(doc,C,W,M);
  function s(val,fallback){return safeStr(val,fallback!==undefined?fallback:'');}
  const verdict=s(result.verdict,'unknown'),fakeProb=parseFloat(result.fake_probability)||0,isFake=verdict==='fake',isDemo=!!result._demo,verdictC=isFake?C.fake:C.real;
  const frames=Array.isArray(result.frame_results)?result.frame_results:[],fakeFrames=frames.filter(f=>parseFloat(f.fake_prob||0)>=FAKE_THRESHOLD),realFrames=frames.filter(f=>parseFloat(f.fake_prob||0)<FAKE_THRESHOLD);
  const fileName=selectedFile?s(selectedFile.name):'unknown',now=s(new Date().toLocaleString()),groqKey=getGroqKey(),fakeRatio=frames.length>0?fakeFrames.length/frames.length:0;
  const thumbCache=[];
  for(const fr of frames){const t=await decodeThumbnail(fr.thumbnail_b64||'');thumbCache.push(t);}
  function addContPage(subtitle){
    doc.addPage();fillBg();doc.setFillColor(...C.hdr);doc.rect(0,0,W,16,'F');doc.setFillColor(...verdictC);doc.rect(0,0,4,16,'F');
    doc.setTextColor(...C.blue);doc.setFontSize(10);doc.setFont('helvetica','bold');T('DeepScan',M,11);
    doc.setTextColor(...C.t2);doc.setFontSize(7);const sf=fileName.length>40?fileName.slice(0,37)+'...':fileName;T('Video Report  --  '+s(subtitle)+'  --  '+sf,M+34,11);
    doc.setTextColor(...verdictC);doc.setFontSize(8);doc.setFont('helvetica','bold');T(verdict.toUpperCase(),W-M,11,{align:'right'});return 22;
  }
  const summaryText=s(await groqWriteSection(`Senior forensic analyst writing executive summary. Video: ${fileName}. Verdict: ${verdict.toUpperCase()}. Fake prob: ${(fakeProb*100).toFixed(1)}%. Frames: ${frames.length}. Fake frames: ${fakeFrames.length} (${(fakeRatio*100).toFixed(0)}%). Write 4 concise professional sentences.`,groqKey,`This video has been classified as ${verdict.toUpperCase()} with a CNN fake probability of ${(fakeProb*100).toFixed(1)}%. Frame-level analysis found ${fakeFrames.length} of ${frames.length} frames exhibiting deepfake indicators. The MobileNetV2 CNN identified face manipulation artefacts. This result warrants further investigation.`));
  const techText=s(await groqWriteSection(`Explain CNN-only video deepfake analysis: ${frames.length} frames, MobileNetV2, fake_prob=1-sigmoid, threshold 50%. 3-4 plain-prose sentences.`,groqKey,`The video was sampled into ${frames.length} frames, each scored independently by the MobileNetV2 CNN. The fake probability per frame was derived by inverting the raw sigmoid output. The overall verdict was derived from the mean fake probability across all sampled frames.`));
  const frameNarratives=[];
  for(const fr of frames){const frProb=clamp(parseFloat(fr.fake_prob||0),0,1),frIsFake=frProb>=FAKE_THRESHOLD;const narrative=await groqWriteSection(`Frame #${s(fr.frame_index)} at ${s(fr.timestamp_sec)}s. CNN verdict: ${frIsFake?'FAKE':'REAL'}. Fake prob: ${(frProb*100).toFixed(1)}%. Write exactly 2 plain-prose sentences.`,groqKey,frIsFake?`Frame ${s(fr.frame_index)} was flagged with ${(frProb*100).toFixed(0)}% fake confidence. Visual artefacts including boundary blending were detected.`:`Frame ${s(fr.frame_index)} was assessed as authentic with ${((1-frProb)*100).toFixed(0)}% real confidence. Natural facial geometry confirmed.`);frameNarratives.push(s(narrative));}
  const conclusionText=s(await groqWriteSection(`Forensic conclusion for CNN-only video deepfake report. Verdict: ${verdict.toUpperCase()}, ${(fakeProb*100).toFixed(1)}% fake prob, ${fakeFrames.length}/${frames.length} fake frames. 3-4 sentences.`,groqKey,`The frame-level CNN evidence indicates this video ${isFake?'has been manipulated using AI-based face synthesis':'does not exhibit significant AI manipulation indicators'}. The automated CNN-only pipeline provides a strong probabilistic assessment but cannot substitute for expert human review. Certified forensic analysis is recommended for legal or evidentiary use.`));
  fillBg();doc.setFillColor(...C.hdr);doc.rect(0,0,W,50,'F');doc.setFillColor(...verdictC);doc.rect(0,0,4,50,'F');
  doc.setTextColor(...C.blue);doc.setFontSize(28);doc.setFont('helvetica','bold');T('DeepScan',M,18);
  doc.setTextColor(...C.t2);doc.setFontSize(9);doc.setFont('helvetica','normal');T('AI Video Deepfake Analysis Report  --  CNN-Only Pipeline',M,27);doc.setFontSize(7.5);T(now,W-M,14,{align:'right'});
  doc.setFillColor(...(isFake?[65,20,20]:[18,58,28]));doc.roundedRect(M,33,W-M*2,13,3,3,'F');doc.setDrawColor(...verdictC);doc.setLineWidth(0.6);doc.roundedRect(M,33,W-M*2,13,3,3,'S');
  doc.setTextColor(...verdictC);doc.setFontSize(13);doc.setFont('helvetica','bold');T(isFake?'DEEPFAKE DETECTED':'AUTHENTIC VIDEO',M+6,41);
  doc.setTextColor(...C.t1);doc.setFontSize(8);doc.setFont('helvetica','normal');const shortFN=fileName.length>40?fileName.slice(0,37)+'...':fileName;T('CNN fake probability: '+(fakeProb*100).toFixed(1)+'%  |  Frames: '+fakeFrames.length+'/'+frames.length+' flagged  |  '+shortFN,M+6,46.5);
  if(isDemo){doc.setFillColor(...C.warn);doc.roundedRect(W-M-26,34,26,8,2,2,'F');doc.setTextColor(20,20,20);doc.setFontSize(6.5);doc.setFont('helvetica','bold');T('DEMO MODE',W-M-13,39.5,{align:'center'});}
  let y=56;
  const sumItems=[{label:'Verdict',value:verdict.toUpperCase(),color:verdictC},{label:'Fake Prob.',value:(fakeProb*100).toFixed(1)+'%',color:verdictC},{label:'Total Frames',value:s(frames.length),color:C.t0},{label:'Fake Frames',value:s(fakeFrames.length),color:fakeFrames.length>0?C.fake:C.t0},{label:'Real Frames',value:s(realFrames.length),color:C.real},{label:'Fake Ratio',value:(fakeRatio*100).toFixed(0)+'%',color:verdictC}];
  const chipW2=(W-M*2-3*5)/6;sumItems.forEach((si,i)=>chip(M+i*(chipW2+3),y,chipW2,18,si.label,si.value,si.color));y+=26;
  y=sectionHead('EXECUTIVE SUMMARY',y);doc.setFillColor(16,18,28);doc.roundedRect(M,y,W-M*2,44,3,3,'F');doc.setDrawColor(...C.border);doc.setLineWidth(0.25);doc.roundedRect(M,y,W-M*2,44,3,3,'S');doc.setFillColor(...verdictC);doc.rect(M,y+4,2.5,36,'F');prose(summaryText,M+7,y+8,W-M*2-12,8,C.t1);y+=52;
  const totalPages=doc.internal.getNumberOfPages();
  for(let p=1;p<=totalPages;p++){doc.setPage(p);footer(p,totalPages);}
  const safeName=(selectedFile?selectedFile.name.replace(/\.[^.]+$/,''):'deepscan')+'_video_report.pdf';
  doc.save(safeName);
}

// ═════════════════════════════════════════════════════════════
//  HISTORY
// ═════════════════════════════════════════════════════════════
function pushHistory(file, result) {
  const previewUrl = fileIsImage(file) ? (lastPreviewUrl || '') : '';
  analysisHistory.unshift({
    id: Date.now(), name: file.name,
    type: result.media_type || currentMode,
    verdict: result.verdict || 'unknown',
    fakeProb: parseFloat(result.fake_probability) || 0,
    preview: previewUrl, ts: timeStr()
  });
  if (analysisHistory.length > 50) analysisHistory.length = 50;
  try { localStorage.setItem('deepscan_history', JSON.stringify(analysisHistory)); } catch (_) {}
}

function renderHistory() {
  const el = document.getElementById('history-list');
  if (!analysisHistory.length) {
    el.innerHTML = '<p class="empty-state">No analyses yet. Run a scan to see history here.</p>';
    return;
  }
  el.innerHTML = analysisHistory.map(h => `
    <div class="history-item">
      ${h.preview
        ? `<img src="${h.preview}" class="history-thumb" alt="" />`
        : `<div class="history-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">${h.type==='video'?'<polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/>':'<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>'}</svg></div>`}
      <div class="history-info">
        <div class="history-name">${h.name}</div>
        <div class="history-meta">${h.type.toUpperCase()} · ${h.ts} · ${(h.fakeProb*100).toFixed(1)}% fake</div>
      </div>
      <span class="verdict-pill ${h.verdict}">${h.verdict.toUpperCase()}</span>
    </div>`).join('');
}

function clearHistory() {
  analysisHistory = [];
  try { localStorage.removeItem('deepscan_history'); } catch (_) {}
  renderHistory();
}

// ═════════════════════════════════════════════════════════════
//  MESSAGE HELPERS
// ═════════════════════════════════════════════════════════════
function addBotMessage(html) {
  const id = 'msg_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
  const el = document.createElement('div');
  el.className = 'msg bot'; el.id = id;
  el.innerHTML = `
    <div class="msg-sender">
      <div class="gem-avatar">
        <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
          <path d="M14 2L26 14L14 26L2 14Z" fill="white" opacity="0.9"/>
          <path d="M14 6L22 14L14 22L6 14Z" fill="white" opacity="0.25"/>
        </svg>
      </div>
      <span class="sender-name">DeepScan</span>
      <span class="sender-time">${timeStr()}</span>
    </div>
    <div class="msg-content">${html}</div>`;
  document.getElementById('messages').appendChild(el);
  scrollBottom();
  return id;
}

function addUserMessage(html) {
  const id = 'msg_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
  const el = document.createElement('div');
  el.className = 'msg user'; el.id = id;
  el.innerHTML = `<div class="msg-bubble">${html}</div>`;
  document.getElementById('messages').appendChild(el);
  scrollBottom();
  return id;
}

function updateMessage(id, html) {
  const el = document.getElementById(id);
  if (el) { el.querySelector('.msg-content').innerHTML = html; scrollBottom(); }
}

function addTyping(text) {
  const id = 'typing_' + Date.now();
  const el = document.createElement('div');
  el.className = 'msg bot'; el.id = id;
  el.innerHTML = `
    <div class="msg-sender">
      <div class="gem-avatar"><svg width="14" height="14" viewBox="0 0 28 28" fill="none"><path d="M14 2L26 14L14 26L2 14Z" fill="white" opacity="0.9"/></svg></div>
      <span class="sender-name">DeepScan</span>
    </div>
    <div class="msg-content" style="display:flex;align-items:center;gap:10px">
      <div class="typing-dots"><span></span><span></span><span></span></div>
      ${text ? `<span style="font-size:0.78rem;color:var(--text-3)">${text}</span>` : ''}
    </div>`;
  document.getElementById('messages').appendChild(el);
  scrollBottom();
  return id;
}

function removeTyping(id) { const e = document.getElementById(id); if (e) e.remove(); }
function showError(msg)   { addBotMessage(`<span style="color:var(--fake)">${msg}</span>`); }

function clearChat() {
  document.getElementById('messages').innerHTML = '';
  addBotMessage(introHTML());
  removeFile();
  lastResult     = null;
  lastPreviewUrl = null;
}

function setSendBtn(disabled) {
  const btn = document.getElementById('send-btn');
  const lbl = document.getElementById('send-label');
  btn.disabled = disabled;
  lbl.innerHTML = disabled
    ? '<span class="spinner" style="border-top-color:#fff"></span>'
    : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`;
}

function scrollBottom() {
  const m = document.getElementById('messages');
  requestAnimationFrame(() => { m.scrollTop = m.scrollHeight; });
}
