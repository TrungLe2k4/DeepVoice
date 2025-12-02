// model.js — gửi vector đặc trưng tới Flask /analyze + fallback heuristic

// === Cấu hình API ===
// Khi deploy, sửa BASE_URL_DEFAULT sang https://your-domain.com
// hoặc http://<server>:5000 khi dev
const BASE_URL_DEFAULT = "http://127.0.0.1:5000";
let API_BASE = BASE_URL_DEFAULT;

// Cho phép đổi API runtime (nếu cần) và lưu vào chrome.storage
export async function setApiBase(url) {
  API_BASE = url || BASE_URL_DEFAULT;
  try {
    if (typeof chrome !== "undefined" && chrome.storage?.local?.set) {
      chrome.storage.local.set({ dv_api_base: API_BASE });
    }
  } catch (e) {
    console.warn("[DV] setApiBase storage error:", e);
  }
}

// Lấy lại config đã lưu (nếu có)
(function restoreApiBaseFromStorage() {
  try {
    if (typeof chrome !== "undefined" && chrome.storage?.local?.get) {
      chrome.storage.local.get("dv_api_base", (st) => {
        if (chrome.runtime?.lastError) {
          console.warn(
            "[DV] chrome.storage.get error:",
            chrome.runtime.lastError
          );
          return;
        }
        if (st && st.dv_api_base) {
          API_BASE = st.dv_api_base;
          console.log("[DV] Restored API_BASE from storage:", API_BASE);
        }
      });
    }
  } catch (e) {
    console.warn("[DV] restoreApiBaseFromStorage error:", e);
  }
})();

// === Throttle để không spam server ===
// (đã có VAD ở worklet + throttle ở content.js, cái này chỉ là tầng bảo vệ thêm)
let lastServerCall = 0;
// Đồng bộ với content.js: khoảng 2s mới gọi server 1 lần
const MIN_SERVER_INTERVAL_MS = 2000;

/* ============================================================
 *  Helper: gọi backend /analyze qua background.js
 * ========================================================== */
async function callBackendAnalyze(body) {
  // Nếu chạy trong extension → dùng sendMessage sang service worker
  if (typeof chrome !== "undefined" && chrome.runtime?.id && chrome.runtime?.sendMessage) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        {
          type: "DV_API_ANALYZE",
          apiBase: API_BASE,
          body,
        },
        (resp) => {
          if (chrome.runtime?.lastError) {
            console.warn(
              "[DV] sendMessage DV_API_ANALYZE error:",
              chrome.runtime.lastError
            );
            reject(chrome.runtime.lastError);
            return;
          }
          if (!resp || !resp.ok) {
            reject(new Error(resp?.error || "No response from background"));
            return;
          }
          resolve(resp.data);
        }
      );
    });
  }

  // Fallback: gọi trực tiếp (chủ yếu để test ngoài môi trường extension)
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* ============================================================
 *  Gọi API /analyze
 * ========================================================== */
export async function sendFeatures(features = {}) {
  const now = Date.now();

  // 🔍 Tính xác suất heuristic trước để quyết định có cần gọi Flask không
  const heur = heuristicProb(features);
  const metaIn = (features && features.meta) || {};
  const snrIn = typeof metaIn.snr === "number" ? metaIn.snr : 0;

  // 🟢 GATE MỚI:
  // Chỉ chặn khi gần như im lặng hoàn toàn (SNR rất thấp và heuristic ≈ 0)
  // → các đoạn có tiếng nói (kể cả voice changer) vẫn được gửi lên backend
  if (snrIn < 0.5 && heur < 0.02) {
    return {
      prob_fast: heur,
      prob_deep: heur,
      prob_embed: heur,
      prob_fused: heur,
      prob_heur: heur,
      reason: ["local-vad-gate"],
      level: "",
      snr: snrIn,
      flags: {},
      alert: false,
      version: "dv-local",
      source: "local-gate",
    };
  }

  // 🕒 Throttle server: khi có tiếng nói nhưng không muốn spam backend
  if (now - lastServerCall < MIN_SERVER_INTERVAL_MS) {
    return {
      prob_fast: heur,
      prob_deep: heur,
      prob_embed: heur,
      prob_fused: heur,
      prob_heur: heur,
      reason: ["server-throttle"],
      level: "",
      snr: snrIn,
      flags: {},
      alert: false,
      version: "dv-local",
      source: "local-throttle",
    };
  }
  lastServerCall = now;

  // Chỉ sanitize khi thật sự gửi lên server
  const body = { features: sanitizeFeatures(features) };

  try {
    // ❗ GỌI BACKEND QUA BACKGROUND (tránh mixed-content)
    const out = await callBackendAnalyze(body);

    const prob_fused = out.prob_fused ?? heur;
    const level = out.level || "";
    const flags = out.flags || {};
    const reasons = Array.isArray(out.reason) ? out.reason : [];
    const snr = typeof out.snr === "number" ? out.snr : snrIn;

    // Nếu backend có trả alert thì dùng, không thì suy ra
    let alert = false;
    if (typeof out.alert === "boolean") {
      alert = out.alert;
    } else {
      alert = prob_fused >= 0.85 || level === "red";
    }

    // normalize output tối thiểu cần cho UI
    return {
      prob_fast: out.prob_fast ?? prob_fused,
      prob_deep: out.prob_deep ?? prob_fused,
      prob_embed: out.prob_embed ?? prob_fused,
      prob_fused,
      prob_heur: heur,
      reason: reasons,
      level,
      snr,
      flags,
      alert,
      version: out.version || "dv-unknown",
      source: "server",
    };
  } catch (e) {
    // fallback khi API lỗi/offline
    console.warn("[DV] API /analyze error:", e);
    return {
      prob_fast: heur,
      prob_deep: heur,
      prob_embed: heur,
      prob_fused: heur,
      prob_heur: heur,
      reason: ["api-fallback"],
      level: "",
      snr: snrIn,
      flags: {},
      alert: false,
      version: "dv-offline",
      source: "offline-heuristic",
    };
  }
}

// === API chính được content.js gọi ===
export async function predictProb(features = {}) {
  const res = await sendFeatures(features);
  // trả về 1 số duy nhất cho content.js
  return res.prob_fused ?? 0;
}

/* ============================================================
 *  Fallback nội bộ (heuristic)
 * ========================================================== */
// dùng một số đặc trưng nhẹ để ước lượng sơ bộ (chỉ cho demo/dev)
function heuristicProb(feats = {}) {
  // kết hợp flatness, entropy, zcr để ước lượng
  const s = feats.spec || {};
  const p1 = clamp01(
    0.55 * (s.flat || 0) +
      0.25 * normEntropy(s.entropy) +
      0.2 * (s.zcr || 0)
  );

  // thêm chút ảnh hưởng của prosody (giọng “quá mượt/quá đều” → nghi ngờ)
  const pros = feats.prosody || {};
  const f0 = pros.f0 || 0;
  let p2 = 0;
  if (f0 > 80 && f0 < 300) {
    // nếu jitter, shimmer thấp bất thường + cpp cao → đẩy nhẹ nghi ngờ
    const j = (pros.jitter || 0) / 5.0; // scale về 0..1
    const sh = (pros.shimmer || 0) / 5.0;
    const cpp = (pros.cpp || 0) / 20.0;
    p2 = clamp01(0.4 * (1 - j) + 0.3 * (1 - sh) + 0.3 * cpp);
  }
  return clamp01(0.7 * p1 + 0.3 * p2);
}

/* ============================================================
 *  Chuẩn hóa features trước khi gửi server
 * ========================================================== */
function sanitizeFeatures(feats = {}) {
  // đảm bảo đủ field theo contract server
  const mfcc = toFixedArray(feats.mfcc, 39);
  const lfcc = toFixedArray(feats.lfcc, 20);

  const pcen_stats = feats.pcen_stats || {};
  const pcen_mean = toFixedArray(pcen_stats.mean, 64);
  const pcen_std = toFixedArray(pcen_stats.std, 64);

  const spec = feats.spec || {};
  const specOut = {
    zcr: num(spec.zcr),
    flat: num(spec.flat),
    rolloff: num(spec.rolloff),
    entropy: num(spec.entropy),
    contrast: num(spec.contrast),
  };

  const pros = feats.prosody || {};
  const prosOut = {
    f0: num(pros.f0),
    jitter: num(pros.jitter),
    shimmer: num(pros.shimmer),
    cpp: num(pros.cpp),
  };

  const meta = feats.meta || {};
  const metaOut = {
    sr: meta.sr || 16000,
    win: meta.win || 1024,
    hop: meta.hop || 256,
    snr: num(meta.snr) || 25.0,
  };

  return {
    mfcc,
    lfcc,
    pcen_stats: { mean: pcen_mean, std: pcen_std },
    spec: specOut,
    pros: prosOut,
    meta: metaOut,
  };
}

/* ============ helpers ============ */
function toFixedArray(arr, n) {
  if (!Array.isArray(arr)) return new Array(n).fill(0);
  const v = arr.flat().map(num);
  if (v.length >= n) return v.slice(0, n);
  const out = v.slice();
  while (out.length < n) out.push(0);
  return out;
}

function num(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : 0;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function normEntropy(h) {
  // entropy đã chuẩn hoá 0..1 ở worklet (n_blocks-based) — ta clamp lại
  return clamp01(Number(h) || 0);
}
