// ==============================
// content.js — DeepVoice Guard
// ==============================

let dvRunning = false;
let audioCtx = null;
let workletReady = false;
let sources = new Map(); // audioEl -> {source, node}
let overlay = null;
let smoother = { avg: 0, alpha: 0.2 };

// 🔹 Load model module (MV3-compliant)
const modelURL = chrome.runtime.getURL("model.js");
import(modelURL)
  .then(mod => (window.DVModel = mod))
  .catch(e => console.warn("Model module error:", e));

// ==============================
// Overlay UI
// ==============================
function ensureOverlay() {
  if (overlay) return overlay;
  overlay = document.createElement("div");
  overlay.id = "dv-overlay";
  overlay.innerHTML = `
    <div class="dv-card">
      <div class="dv-row">
        <div class="dv-dot" id="dv-status-dot"></div>
        <div>
          <div class="dv-title">DeepVoice Guard</div>
          <div class="dv-sub">Giám sát deepfake theo thời gian thực</div>
        </div>
      </div>
      <div class="dv-meter">
        <div class="dv-meter-bar"><span id="dv-meter-fill"></span></div>
        <div class="dv-meter-label">Xác suất deepfake (tổng hợp)</div>
      </div>
      <div class="dv-detail" id="dv-detail"></div>
    </div>
  `;
  document.documentElement.appendChild(overlay);
  return overlay;
}

function setStatus(prob, detail) {
  const dot = document.getElementById("dv-status-dot");
  const fill = document.getElementById("dv-meter-fill");
  const det = document.getElementById("dv-detail");
  if (!dot || !fill || !det) return;

  smoother.avg = smoother.alpha * prob + (1 - smoother.alpha) * smoother.avg;
  const p = Math.max(0, Math.min(1, smoother.avg));
  fill.style.width = (p * 100).toFixed(1) + "%";

  if (p > 0.85) {
    dot.style.background = "#e53935"; // red
    det.textContent = detail || "Nguy cơ cao: dấu hiệu giả mạo giọng nói rõ rệt.";
  } else if (p > 0.6) {
    dot.style.background = "#fb8c00"; // orange
    det.textContent = detail || "Nguy cơ trung bình: có dấu hiệu bất thường.";
  } else {
    dot.style.background = "#43a047"; // green
    det.textContent = detail || "An toàn: chưa thấy dấu hiệu rõ rệt.";
  }
}

// ==============================
// Core control
// ==============================
async function start() {
  if (dvRunning) return;
  dvRunning = true;
  ensureOverlay();
  smoother.avg = 0;

  try {
    if (!audioCtx) {
      audioCtx = new AudioContext({ latencyHint: "interactive" });
    }
    if (!workletReady) {
      const workletUrl = chrome.runtime.getURL("worklet-processor.js");
      await audioCtx.audioWorklet.addModule(workletUrl);
      workletReady = true;
    }

    attachToAudioTags();
    observeAudioTags();
    setStatus(0, "Đang khởi động theo dõi...");
  } catch (e) {
    console.error("DV start error:", e);
    setStatus(0, "Không thể khởi động AudioWorklet: " + (e?.message || e));
  }
}

function stop() {
  dvRunning = false;

  for (const [el, obj] of sources.entries()) {
    try {
      obj.source.disconnect();
    } catch {}
    try {
      obj.node.port.close();
    } catch {}
  }
  sources.clear();

  if (audioCtx) {
    try {
      audioCtx.close();
    } catch {}
    audioCtx = null;
    workletReady = false;
  }

  if (overlay) {
    overlay.remove();
    overlay = null;
  }
}

// ==============================
// Audio chain for each <audio>
// ==============================
function makeChainFor(el) {
  if (!audioCtx || !workletReady) return null;

  // Ưu tiên WebRTC srcObject (Meet)
  const sourceNode = el.srcObject
    ? audioCtx.createMediaStreamSource(el.srcObject)
    : audioCtx.createMediaElementSource(el);

  const node = new AudioWorkletNode(audioCtx, "dv-analyzer", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: { sampleRate: audioCtx.sampleRate },
  });

  // Ngăn phát echo (âm lặp)
  try {
    el.muted = true;
  } catch {}

  // Pass-through giữ âm thanh có thể nghe
  sourceNode.connect(node).connect(audioCtx.destination);

  node.port.onmessage = async (ev) => {
    const d = ev.data;
    if (!d || d.type !== "features") return;
    try {
      const prob = await window.DVModel.predictProb(d.features);
      setStatus(prob, d.explain);
    } catch (err) {
      console.warn("Predict error:", err);
    }
  };

  return { source: sourceNode, node };
}

// ==============================
// Attach to <audio> Meet
// ==============================
function attachToAudioTags() {
  const audios = Array.from(document.querySelectorAll("audio"));
  for (const a of audios) {
    if (sources.has(a)) continue;
    if (a.srcObject || a.src) {
      const obj = makeChainFor(a);
      if (obj) sources.set(a, obj);
    }
  }
}

function observeAudioTags() {
  const mo = new MutationObserver(() => {
    if (!dvRunning) return;
    attachToAudioTags();
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });
}

// ==============================
// Listen from background
// ==============================
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "DV_START") start();
  if (msg?.type === "DV_STOP") stop();
});
