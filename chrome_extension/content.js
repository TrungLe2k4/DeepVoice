// content.js — DeepVoice Guard (realtime, no-echo, Flask-ready)

let dvRunning = false;
let audioCtx = null;
let workletReady = false;
let overlay = null;
// Map<HTMLMediaElement, { src, node }>
const sources = new Map();

// Observer theo dõi media element
let mediaObserver = null;

// Đảm bảo AudioContext chỉ start sau user gesture bên trong tab Meet
let pendingStart = false;
let gestureBound = false;

// Trạng thái smoothing cho meter
let smooth = { p: 0, alpha: 0.2 };

// Throttle infer để tránh spam API / model
let lastInferTime = 0;
let inferBusy = false;
// ✅ Phân tích tối đa 1 lần mỗi 2 giây
const INFER_INTERVAL_MS = 2000; // 2000ms = 2s giữa 2 lần infer

// 🔴 Banner cảnh báo trên Google Meet
let alertBox = null;
let alertTimeoutId = null;

// Tải module gọi API / fallback heuristic
import(chrome.runtime.getURL("model.js"))
  .then((mod) => {
    window.DVModel = mod;
  })
  .catch((e) => console.warn("DV model module error:", e));

/* ================= UI Overlay ================= */

function ensureOverlay() {
  if (overlay && document.contains(overlay)) return overlay;

  overlay = document.createElement("div");
  overlay.id = "dv-overlay";
  overlay.innerHTML = `
    <div class="dv-card">
      <div class="dv-row">
        <div class="dv-dot" id="dv-dot"></div>
        <div>
          <div class="dv-title">DeepVoice Guard</div>
          <div class="dv-sub" id="dv-sub">Giám sát deepfake (realtime)</div>
        </div>
      </div>
      <div class="dv-meter">
        <div class="dv-meter-bar"><span id="dv-meter"></span></div>
        <div class="dv-meter-label">Xác suất giả mạo</div>
      </div>
      <div class="dv-detail" id="dv-detail">Đang khởi động...</div>
    </div>`;
  document.documentElement.appendChild(overlay);
  return overlay;
}

// 🔴 Banner cảnh báo trên trang Meet (trên cùng màn hình)
function ensureAlertBox() {
  if (alertBox && document.contains(alertBox)) return alertBox;

  alertBox = document.createElement("div");
  alertBox.id = "dv-alert";
  alertBox.innerHTML = `<span id="dv-alert-text"></span>`;

  // Style inline để không cần CSS riêng
  Object.assign(alertBox.style, {
    position: "fixed",
    top: "12px",
    left: "50%",
    transform: "translateX(-50%) translateY(-6px)",
    zIndex: 999999,
    padding: "10px 16px",
    borderRadius: "999px",
    fontFamily:
      "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontSize: "13px",
    fontWeight: "500",
    color: "#fff",
    boxShadow: "0 4px 16px rgba(0,0,0,0.25)",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    opacity: "0",
    pointerEvents: "none",
    transition: "opacity 0.25s ease-out, transform 0.25s ease-out",
    transformOrigin: "top center",
  });

  document.documentElement.appendChild(alertBox);
  return alertBox;
}

function hideAlertBanner() {
  if (!alertBox) return;
  alertBox.style.opacity = "0";
  alertBox.style.transform = "translateX(-50%) translateY(-6px)";
}

function showAlertBanner(prob, level, reasons = []) {
  // Chỉ hiển thị cảnh báo cho amber / red
  let lv = level;
  if (!lv) {
    if (prob >= 0.85) lv = "red";
    else if (prob >= 0.6) lv = "amber";
    else lv = "green";
  }
  if (lv === "green") return;

  const box = ensureAlertBox();
  const textEl = box.querySelector("#dv-alert-text");
  if (!textEl) return;

  const filteredReasons = Array.isArray(reasons)
    ? reasons.filter(
        (r) =>
          r !== "local-vad-gate" &&
          r !== "server-throttle" &&
          r !== "api-fallback"
      )
    : [];

  const mainReason =
    filteredReasons.length > 0
      ? filteredReasons[0]
      : "Phát hiện tín hiệu bất thường trong giọng nói.";

  let titlePrefix = "";
  let bg = "";
  if (lv === "red") {
    titlePrefix = "⚠️ Nguy cơ deepfake cao – ";
    bg = "linear-gradient(90deg, #d32f2f, #f44336)";
  } else {
    titlePrefix = "⚠️ Cảnh báo deepfake – ";
    bg = "linear-gradient(90deg, #ef6c00, #fb8c00)";
  }

  box.style.backgroundImage = bg;
  textEl.textContent = titlePrefix + mainReason;

  // Hiện banner
  box.style.opacity = "1";
  box.style.transform = "translateX(-50%) translateY(0)";

  if (alertTimeoutId) {
    clearTimeout(alertTimeoutId);
  }
  alertTimeoutId = setTimeout(() => {
    hideAlertBanner();
  }, 8000); // auto-hide sau 8s
}

// ✅ Cập nhật overlay bên dưới góc
function setStatus(prob, reasons = [], level = null) {
  const dot = document.getElementById("dv-dot");
  const fill = document.getElementById("dv-meter");
  const det = document.getElementById("dv-detail");
  const sub = document.getElementById("dv-sub");
  if (!dot || !fill || !det) return;

  const p = Math.max(0, Math.min(1, Number(prob) || 0));
  smooth.p = smooth.alpha * p + (1 - smooth.alpha) * smooth.p;
  fill.style.width = smooth.p * 100 + "%";

  // Ghép lý do gọn gàng (bỏ các mã nội bộ như local-vad-gate, server-throttle)
  const filteredReasons = Array.isArray(reasons)
    ? reasons.filter(
        (r) =>
          r !== "local-vad-gate" &&
          r !== "server-throttle" &&
          r !== "api-fallback"
      )
    : [];
  const reasonText = filteredReasons.length ? filteredReasons.join(" · ") : "";

  // Nếu backend trả level thì ưu tiên dùng, không thì suy ra từ prob (đã smooth)
  let lv = level;
  if (!lv) {
    if (smooth.p >= 0.85) lv = "red";
    else if (smooth.p >= 0.6) lv = "amber";
    else lv = "green";
  }

  if (lv === "red") {
    dot.style.background = "#e53935";
    if (sub) sub.textContent = "Mức rủi ro: Cao — đang phân tích tín hiệu...";
    det.textContent = "🔴 Nguy cơ deepfake cao. " + reasonText;
  } else if (lv === "amber") {
    dot.style.background = "#fb8c00";
    if (sub) sub.textContent = "Mức rủi ro: Trung bình — đang phân tích...";
    det.textContent = "🟠 Có dấu hiệu bất thường. " + reasonText;
  } else {
    dot.style.background = "#43a047";
    if (sub) sub.textContent = "Mức rủi ro: Thấp — đang giám sát & phân tích...";
    det.textContent = "🟢 An toàn. " + reasonText;
  }
}

/* =============== Local heuristic fallback (nếu không có model.js) =============== */
// Payload thực tế từ worklet:
// {
//   mfcc, lfcc,
//   pcen_stats: { mean[64], std[64] },
//   spec: { zcr, flat, rolloff, entropy, contrast },
//   prosody: { f0, jitter, shimmer, cpp },
//   meta: { sr, win, hop, snr }
// }

function localHeuristic(payload) {
  if (!payload) return 0;

  const spec = payload.spec || {};
  const pros = payload.prosody || {};
  const meta = payload.meta || {};

  const zcr = Number(spec.zcr || 0); // 0..1
  const flat = Number(spec.flat || 0); // 0..1
  const ent = Number(spec.entropy || 0); // 0..1
  const contr = Number(spec.contrast || 0); // ~0..?
  const snr = Number(meta.snr || 20); // 0..40 dB

  const f0 = Number(pros.f0 || 0);
  const jitter = Number(pros.jitter || 0); // 0..500 (scaled)
  const shimmer = Number(pros.shimmer || 0); // 0..500
  const cpp = Number(pros.cpp || 0); // 0..~20

  let score = 0;

  // 1) Giọng "quá mượt/quá sạch": flat cao, entropy thấp, SNR cao
  const pClean =
    0.6 * flat + 0.4 * (1 - ent) + 0.3 * Math.max(0, (snr - 25) / 15); // snr > 25 dB
  score += pClean * 0.5;

  // 2) Prosody "robot" — jitter, shimmer rất thấp nhưng CPP cao
  const jNorm = Math.min(1, jitter / 100);
  const shNorm = Math.min(1, shimmer / 100);
  const cppNorm = Math.min(1, cpp / 20);
  const pRobot = 0.5 * (1 - jNorm) + 0.3 * (1 - shNorm) + 0.2 * cppNorm;
  score += pRobot * 0.3;

  // 3) ZCR quá cao cũng gợi ý tín hiệu tổng hợp / nhiễu kỳ lạ
  score += Math.max(0, (zcr - 0.15) * 2.0);

  // 4) Nếu F0 ngoài range giọng người (60–400), cộng nhẹ
  if (f0 < 60 || f0 > 400) {
    score += 0.1;
  }

  return Math.max(0, Math.min(1, score));
}

/* ================= Core ================= */

async function start() {
  if (dvRunning) return;
  dvRunning = true;

  ensureOverlay();
  setStatus(0.02, ["Đang khởi động mô-đun phân tích..."]);

  // Tạo AudioContext
  if (!audioCtx) {
    try {
      audioCtx = new AudioContext({ latencyHint: "interactive" });
    } catch (e) {
      console.error("Cannot create AudioContext:", e);
      setStatus(0, ["Không tạo được AudioContext"]);
      dvRunning = false;
      return;
    }
  }

  // Resume — bắt buộc phải nằm trong user gesture (handler phía dưới đảm bảo)
  if (audioCtx.state === "suspended") {
    try {
      await audioCtx.resume();
    } catch (e) {
      console.warn("AudioContext resume blocked (no user gesture?):", e);
      setStatus(0, [
        "Trình duyệt đang chặn AudioContext.",
        "Hãy click vào cửa sổ Meet hoặc bật micro/loa rồi bật lại.",
      ]);
      dvRunning = false;
      return;
    }
  }

  // Load AudioWorklet
  if (!workletReady) {
    try {
      await audioCtx.audioWorklet.addModule(
        chrome.runtime.getURL("worklet-processor.js")
      );
      workletReady = true;
    } catch (e) {
      console.error("AudioWorklet addModule error:", e);
      setStatus(0, ["Không tải được Worklet"]);
      dvRunning = false;
      return;
    }
  }

  // Đợi 1 frame để DOM ổn định rồi attach
  requestAnimationFrame(() => {
    attachToMediaTags();
    observeMediaTags();
    setStatus(0.05, ["Đang giám sát & phân tích giọng nói trong cuộc gọi..."]);
  });
}

function stop() {
  dvRunning = false;
  pendingStart = false;

  // Ngắt mọi chain
  for (const [el, obj] of sources.entries()) {
    try {
      obj.src.disconnect();
    } catch (e) {}
    try {
      obj.node.port.close();
    } catch (e) {}
    try {
      obj.node.disconnect();
    } catch (e) {}
  }
  sources.clear();

  if (audioCtx) {
    try {
      audioCtx.close();
    } catch (e) {}
    audioCtx = null;
    workletReady = false;
  }

  // Ngắt observer nếu còn
  if (mediaObserver) {
    try {
      mediaObserver.disconnect();
    } catch (e) {}
    mediaObserver = null;
  }

  if (overlay) {
    overlay.remove();
    overlay = null;
  }

  if (alertBox) {
    alertBox.remove();
    alertBox = null;
  }
  if (alertTimeoutId) {
    clearTimeout(alertTimeoutId);
    alertTimeoutId = null;
  }

  smooth.p = 0;
}

/* ================= Audio chain ================= */

function makeChainFor(mediaEl) {
  if (!audioCtx || !workletReady) return null;

  let srcNode;
  try {
    srcNode = audioCtx.createMediaElementSource(mediaEl);
  } catch (e) {
    // Nếu đã được tạo source trước đó (bởi script khác) sẽ lỗi
    console.warn("[DV] createMediaElementSource error:", e);
    return null;
  }

  const wnode = new AudioWorkletNode(audioCtx, "dv-analyzer", {
    numberOfInputs: 1,
    numberOfOutputs: 0, // không phát lại → tránh echo
    processorOptions: { sampleRate: audioCtx.sampleRate },
  });

  // Chỉ phân tích — không nối đến destination
  srcNode.connect(wnode);

  // Nhận payload đặc trưng và gọi API / heuristic (có throttle)
  wnode.port.onmessage = async (ev) => {
    const d = ev.data;
    if (!d || d.type !== "features") return;

    const now =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();

    // Throttle: chỉ infer mỗi INFER_INTERVAL_MS
    if (now - lastInferTime < INFER_INTERVAL_MS) {
      return;
    }
    // Không chạy chồng infer
    if (inferBusy) {
      return;
    }

    lastInferTime = now;
    inferBusy = true;

    const feats = d.payload;

    // 👉 Debug nhẹ (bật khi cần)
    // console.log("[DV] features tick:",
    //   "SNR=", feats.meta?.snr,
    //   "flat=", feats.spec?.flat,
    //   "zcr=", feats.spec?.zcr
    // );

    try {
      const model = window.DVModel;
      let prob = 0;
      let reasons = [];
      let level = null; // nhận level từ backend nếu có

      // Cho UI biết đang phân tích
      setStatus(smooth.p, ["Đang phân tích tín hiệu..."], level);

      if (model?.sendFeatures) {
        const out = await model.sendFeatures(feats);
        prob = out?.prob_fused ?? out?.prob ?? 0;
        reasons = out?.reason || out?.reasons || [];
        level = out?.level || null;
      } else if (model?.predictProb) {
        prob = await model.predictProb(feats);
      } else {
        prob = localHeuristic(feats);
      }

      // Cập nhật overlay
      setStatus(prob, reasons, level);

      // Tự quyết định khi nào show banner cảnh báo
      const effLevel =
        level ||
        (prob >= 0.85 ? "red" : prob >= 0.6 ? "amber" : "green");

      if (effLevel === "red" || effLevel === "amber") {
        showAlertBanner(prob, effLevel, reasons);
      }
    } catch (e) {
      // im lặng để không spam console khi chạy lâu
      // console.warn("[DV] infer error:", e);
    } finally {
      inferBusy = false;
    }
  };

  return { src: srcNode, node: wnode };
}

function collectMediaElements() {
  // Bắt cả audio lẫn video vì Meet hay dùng <video> chứa audio track
  return Array.from(document.querySelectorAll("audio, video"));
}

function attachToMediaTags() {
  if (!dvRunning) return;
  const medias = collectMediaElements();

  for (const m of medias) {
    if (sources.has(m)) continue;
    if (m.src || m.srcObject) {
      const ch = makeChainFor(m);
      if (ch) sources.set(m, ch);
    }
  }
}

function observeMediaTags() {
  if (mediaObserver) return;

  mediaObserver = new MutationObserver(() => {
    if (dvRunning) attachToMediaTags();
  });

  mediaObserver.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  // Định kỳ dọn phần tử media đã bị remove khỏi DOM
  const gc = setInterval(() => {
    if (!dvRunning) {
      clearInterval(gc);
      if (mediaObserver) {
        try {
          mediaObserver.disconnect();
        } catch (e) {}
        mediaObserver = null;
      }
      return;
    }
    for (const [el, obj] of sources.entries()) {
      if (!document.contains(el)) {
        try {
          obj.src.disconnect();
        } catch (e) {}
        try {
          obj.node.port.close();
        } catch (e) {}
        try {
          obj.node.disconnect();
        } catch (e) {}
        sources.delete(el);
      }
    }
  }, 2000);
}

/* ================= User gesture binding ================= */

function bindGestureOnce() {
  if (gestureBound) return;
  gestureBound = true;

  const handler = () => {
    gestureBound = false;
    window.removeEventListener("pointerdown", handler, true);
    window.removeEventListener("keydown", handler, true);

    if (pendingStart && !dvRunning) {
      // gọi start() trực tiếp trong handler → AudioContext resume hợp lệ
      start();
    }
  };

  window.addEventListener("pointerdown", handler, true);
  window.addEventListener("keydown", handler, true);
}

/* ================= Nhận message từ background ================= */

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || !msg.type) return;

  // Thông báo nhẹ khi user bật ở tab không phải Meet (nếu content.js có mặt)
  if (msg.type === "DV_INFO" && msg.message) {
    ensureOverlay();
    setStatus(0, [msg.message]);
    return;
  }

  // Flow chính: background gửi DV_TOGGLE
  if (msg.type === "DV_TOGGLE") {
    if (dvRunning || pendingStart) {
      // Đang ON → tắt
      pendingStart = false;
      if (dvRunning) {
        stop();
      } else {
        if (overlay) {
          overlay.remove();
          overlay = null;
        }
        if (alertBox) {
          alertBox.remove();
          alertBox = null;
        }
        if (alertTimeoutId) {
          clearTimeout(alertTimeoutId);
          alertTimeoutId = null;
        }
        smooth.p = 0;
      }
      return;
    }

    // Đang OFF → chuẩn bị bật, chờ user gesture trong tab
    pendingStart = true;
    ensureOverlay();
    setStatus(0.02, [
      "Nhấp vào cửa sổ Meet để bắt đầu giám sát & phân tích...",
    ]);
    bindGestureOnce();
    return;
  }

  // Tùy chọn: tương thích nếu sau này bạn muốn dùng DV_START / DV_STOP
  if (msg.type === "DV_START") {
    pendingStart = true;
    ensureOverlay();
    setStatus(0.02, [
      "Nhấp vào cửa sổ Meet để bắt đầu giám sát & phân tích...",
    ]);
    bindGestureOnce();
    return;
  }

  if (msg.type === "DV_STOP") {
    pendingStart = false;
    if (dvRunning) {
      stop();
    } else {
      if (overlay) {
        overlay.remove();
        overlay = null;
      }
      if (alertBox) {
        alertBox.remove();
        alertBox = null;
      }
      if (alertTimeoutId) {
        clearTimeout(alertTimeoutId);
        alertTimeoutId = null;
      }
      smooth.p = 0;
    }
  }
});

// (Tuỳ chọn) auto-start khi vào Meet — KHÔNG khuyến khích vì sẽ vi phạm AudioContext policy
// start();
