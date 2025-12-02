# backend_flask/app.py
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import numpy as np
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =========================
# 1️⃣  NẠP MÔ HÌNH & CẤU HÌNH
# =========================

BASE_DIR = os.path.dirname(__file__)

MODEL_DIR = os.path.join(BASE_DIR, "Models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEAT_PATH = os.path.join(MODEL_DIR, "fast_features.json")
SCALER_STATS_PATH = os.path.join(MODEL_DIR, "fast_scaler_stats.json")
RES2NET_METRICS_PATH = os.path.join(MODEL_DIR, "res2net_metrics.json")

LOG_DIR = os.path.join(BASE_DIR, "Logs")
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

has_model = os.path.exists(MODEL_PATH)

if has_model:
    # ✅ MODEL LÀ PIPELINE: StandardScaler + XGBoost
    model = joblib.load(MODEL_PATH)
    print("✅ Fast model pipeline loaded:", MODEL_PATH)
else:
    print("⚠️ Chưa có model thật, dùng DummyModel tạm (chỉ để test API).")

    class DummyModel:
        def predict_proba(self, X):
            mean = float(np.mean(X))
            p = 1.0 / (1.0 + np.exp(-5 * mean))
            return np.array([[1 - p, p]], dtype=np.float32)

    model = DummyModel()

# Số lượng feature mà model mong đợi
if os.path.exists(FEAT_PATH):
    with open(FEAT_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    N_FEAT = len(cfg.get("features", [])) or 196
else:
    N_FEAT = 196  # 39 + 20 + 64 + 64 + 5 + 4

# Thông tin scaler (chỉ để show trong /health, inference dùng Pipeline)
if os.path.exists(SCALER_STATS_PATH):
    with open(SCALER_STATS_PATH, "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)
else:
    scaler_stats = {}

# Metrics Res2Net (nếu có) để show trong /health
if os.path.exists(RES2NET_METRICS_PATH):
    with open(RES2NET_METRICS_PATH, "r", encoding="utf-8") as f:
        res2net_metrics = json.load(f)
else:
    res2net_metrics = {}

# =========================
# 💾 BIẾN LƯU EVENT MỚI NHẤT CHO DASHBOARD
# =========================
LAST_EVENT = {
    "ts": None,
    "prob_fused": 0.0,
    "prob_fast": 0.0,
    "prob_deep": 0.0,
    "prob_embed": 0.0,
    "prob_heur": 0.0,
    "level": "green",
    "snr": 0.0,
    "flags": {},
    "reasons": [],
    "alert": False,
}


# =========================
# 🧠 HÀM QUYẾT ĐỊNH LEVEL
# =========================
def decide_level(prob_fused: float) -> str:
    # Giữ nguyên ngưỡng bạn đang dùng
    if prob_fused >= 0.85:
        return "red"
    if prob_fused >= 0.6:
        return "amber"
    return "green"


# =========================
# 0️⃣  ROOT: redirect / → /dashboard (đỡ 404)
# =========================
@app.route("/", methods=["GET"])
def index():
    return redirect("/dashboard")


# =========================
# 2️⃣  API: /health
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "model_file": os.path.basename(MODEL_PATH),
        "has_model": has_model,
        "n_features": N_FEAT,
        "scaler_stats": scaler_stats,
        "res2net_metrics": res2net_metrics,
        "version": "dv-1.0.0"
    })


# =========================
# 3️⃣  API: /analyze
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True) or {}
        feats = data.get("features", None)

        if feats is None:
            return jsonify({"error": "Thiếu trường 'features' trong JSON"}), 400

        # 1) Ghép vector 1D chuẩn (196 phần tử) cho model
        vec = extract_feature_vector(feats)

        if vec.size != N_FEAT:
            return jsonify({
                "error": f"Sai số lượng đặc trưng: cần {N_FEAT}, nhận {vec.size}"
            }), 400

        # 2) Dự đoán xác suất Deepfake bằng fast model (Pipeline có StandardScaler bên trong)
        prob_fast = float(model.predict_proba(vec.reshape(1, -1))[0, 1])

        # 3) Phân tích nhanh spec/prosody để sinh flags + extra reasons + snr
        flags, extra_reasons, snr = quick_flags(feats, prob_fast)

        # 4) Heuristic đơn giản (backend) để hỗ trợ fusion
        spec = feats.get("spec", {}) if isinstance(feats, dict) else {}
        entropy = float(spec.get("entropy", 0.0)) if spec else 0.0
        flat = float(spec.get("flat", 0.0)) if spec else 0.0
        h = 0.5 * flat + 0.3 * max(0.0, 1.0 - entropy) + 0.2 * max(0.0, (snr - 10.0) / 30.0)
        prob_heur = max(0.0, min(1.0, h))

        # 5) (tạm thời) prob_deep & prob_embed reuse fast cho demo kiến trúc nhiều tầng
        prob_deep = prob_fast
        prob_embed = prob_fast

        # 6) Fusion: kết hợp fast + deep + heuristic
        prob_fused = 0.6 * prob_fast + 0.3 * prob_deep + 0.1 * prob_heur

        # 7) Gán mức cảnh báo chính
        level = decide_level(prob_fused)
        if level == "red":
            base_reason = "Tín hiệu tổng hợp rõ rệt (MFCC/LFCC/PCEN lệch chuẩn)."
        elif level == "amber":
            base_reason = "Có dấu hiệu bất thường trong MFCC/LFCC/PCEN."
        else:
            base_reason = "An toàn: chưa thấy dấu hiệu giả mạo rõ ràng."

        # Gộp reason: 1 reason chính + các reason phụ (loại trùng)
        reasons = [base_reason]
        for r in extra_reasons:
            if r and r not in reasons:
                reasons.append(r)

        # 8) Quyết định có bật cảnh báo mạnh (alert) không
        #    - red luôn alert
        #    - amber + có flag "too_clean" hoặc "robotic_prosody" hoặc "high_zcr" cũng alert
        alert = False
        if level == "red":
            alert = True
        elif level == "amber" and any(flags.get(k) for k in ("too_clean", "robotic_prosody", "high_zcr")):
            alert = True

        # 🔴 In log ra console cho dễ debug
        print(
            f"[ANALYZE] prob_fast={prob_fast:.3f} "
            f"prob_fused={prob_fused:.3f} level={level} snr={snr:.1f} "
            f"flags={list(flags.keys())} alert={alert}"
        )

        # 🔴 Ghi log vào file
        log_event(feats, prob_fused, level, flags, snr, reasons, alert)

        # ✅ CẬP NHẬT SỰ KIỆN MỚI NHẤT CHO DASHBOARD BACKEND
        global LAST_EVENT
        LAST_EVENT = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "prob_fused": float(prob_fused),
            "prob_fast": float(prob_fast),
            "prob_deep": float(prob_deep),
            "prob_embed": float(prob_embed),
            "prob_heur": float(prob_heur),
            "level": level,
            "snr": float(snr),
            "flags": flags or {},
            "reasons": reasons,
            "alert": bool(alert),
        }

        return jsonify({
            "prob_fast": prob_fast,
            "prob_deep": prob_deep,
            "prob_embed": prob_embed,
            "prob_fused": prob_fused,
            "prob_heur": prob_heur,
            "reason": reasons,
            "level": level,
            "snr": snr,
            "flags": flags,
            "alert": alert,
            "version": "dv-1.0.0"
        })

    except Exception as e:
        print("[ANALYZE_ERROR]", e)
        return jsonify({"error": str(e)}), 500


# =========================
# 4️⃣  PHÂN TÍCH NHANH ĐỂ TẠO FLAGS
# =========================
def quick_flags(feats, prob):
    """
    Phân tích nhanh một số đặc trưng để sinh:
      - flags: {too_clean, robotic_prosody, high_zcr, weird_f0, ...}
      - extra_reasons: list[str] mô tả cho overlay
      - snr: số dB (nếu có)
    """
    flags = {}
    extra = []
    snr = 0.0

    # Nếu features là dict (đúng format từ extension)
    if isinstance(feats, dict):
        spec = feats.get("spec", {}) or {}
        pros = feats.get("prosody", {}) or {}
        meta = feats.get("meta", {}) or {}

        zcr = float(spec.get("zcr", 0.0))
        flat = float(spec.get("flat", 0.0))
        entropy = float(spec.get("entropy", 0.0))
        contrast = float(spec.get("contrast", 0.0))

        f0 = float(pros.get("f0", 0.0))
        jitter = float(pros.get("jitter", 0.0))
        shimmer = float(pros.get("shimmer", 0.0))
        cpp = float(pros.get("cpp", 0.0))

        snr = float(meta.get("snr", 0.0))

        # 1) Âm thanh quá "sạch" & phẳng
        if snr > 28 and flat > 0.5 and entropy < 0.5:
            flags["too_clean"] = True
            extra.append("Phổ tần số rất sạch & phẳng (nghi ngờ tổng hợp).")

        # 2) Prosody robot: jitter/shimmer rất thấp, CPP cao
        if jitter < 0.5 and shimmer < 0.5 and cpp > 8:
            flags["robotic_prosody"] = True
            extra.append("Độ run & biên độ giọng rất thấp, formant ổn định bất thường.")

        # 3) ZCR cao
        if zcr > 0.25:
            flags["high_zcr"] = True
            extra.append("Zero-crossing rate cao, có thể là tín hiệu tổng hợp / nhiễu lạ.")

        # 4) F0 lạ
        if 0 < f0 < 60 or f0 > 400:
            flags["weird_f0"] = True
            extra.append("Tần số cơ bản nằm ngoài dải giọng người điển hình.")

        # 5) Nếu prob thấp & không có flag nào → reassure
        if prob < 0.4 and not flags:
            extra.append("Đặc trưng ổn định, phù hợp giọng nói tự nhiên.")
    else:
        # Không phải dict (ví dụ: gửi thẳng vector) → không phân tích được chi tiết
        if prob < 0.4:
            extra.append("Đặc trưng tổng thể ở mức an toàn.")

    return flags, extra, snr


# =========================
# 5️⃣  GHI LOG SỰ KIỆN /analyze
# =========================
def log_event(feats, prob_fused, level, flags, snr, reasons, alert=False):
    """
    Ghi lại mỗi lần /analyze vào file JSONL:
      backend_flask/Logs/events.jsonl

    Mỗi dòng là một JSON:
      {
        "ts": "...",
        "prob_fused": ...,
        "level": "...",
        "snr": ...,
        "flags": {...},
        "reasons": [...],
        "alert": bool,
        "spec": {...},
        "prosody": {...},
        "meta": {...}
      }
    """
    try:
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "prob_fused": float(prob_fused),
            "level": level,
            "snr": float(snr),
            "flags": flags or {},
            "reasons": reasons or [],
            "alert": bool(alert),
        }

        # Nếu feats là dict (đúng format từ extension) thì log gọn phần spec/prosody/meta
        if isinstance(feats, dict):
            event["spec"] = feats.get("spec", {})
            event["prosody"] = feats.get("prosody", {})
            event["meta"] = feats.get("meta", {})
        else:
            # Nếu là vector phẳng thì chỉ log độ dài
            try:
                event["raw_dim"] = int(np.asarray(feats).size)
            except Exception:
                event["raw_dim"] = 0

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        # Không để việc log lỗi làm crash API
        print("[LOG_EVENT_ERR]", e)


# =========================
# 6️⃣  HÀM GHÉP VECTOR ĐẶC TRƯNG
# =========================
def extract_feature_vector(feats):
    """
    Ghép toàn bộ đặc trưng (MFCC, LFCC, PCEN, spec, prosody) thành vector 1D.

    Hỗ trợ 2 dạng:
      - feats là list/ndarray 196 phần tử  → dùng trực tiếp
      - feats là dict:
          {
            "mfcc": [...39],
            "lfcc": [...20],
            "pcen_stats": {"mean":[...64], "std":[...64]},
            "spec": {"zcr":..,"flat":..,"rolloff":..,"entropy":..,"contrast":..},
            "prosody": {"f0":..,"jitter":..,"shimmer":..,"cpp":..}
          }
    """
    # Trường hợp extension gửi sẵn mảng phẳng
    if isinstance(feats, (list, tuple, np.ndarray)):
        arr = np.asarray(feats, dtype=np.float32).ravel()
        # nếu ít hơn N_FEAT thì pad 0, dài hơn thì cắt bớt
        if arr.size < N_FEAT:
            out = np.zeros(N_FEAT, dtype=np.float32)
            out[:arr.size] = arr
            return out
        return arr[:N_FEAT]

    # Trường hợp gửi dạng dict nhiều trường
    if not isinstance(feats, dict):
        # format sai → trả vector 0
        return np.zeros(N_FEAT, dtype=np.float32)

    def safe(arr, n):
        if not isinstance(arr, (list, np.ndarray)):
            return np.zeros(n, dtype=np.float32)
        a = np.asarray(arr, dtype=np.float32).ravel()
        if a.size >= n:
            return a[:n]
        out = np.zeros(n, dtype=np.float32)
        out[:a.size] = a
        return out

    mfcc = safe(feats.get("mfcc"), 39)
    lfcc = safe(feats.get("lfcc"), 20)

    pcen = feats.get("pcen_stats", {}) or {}
    pcen_mean = safe(pcen.get("mean"), 64)
    pcen_std = safe(pcen.get("std"), 64)

    spec = feats.get("spec", {}) or {}
    spec_vec = np.array([
        spec.get("zcr", 0.0),
        spec.get("flat", 0.0),
        spec.get("rolloff", 0.0),
        spec.get("entropy", 0.0),
        spec.get("contrast", 0.0),
    ], dtype=np.float32)

    pros = feats.get("prosody", {}) or {}
    pros_vec = np.array([
        pros.get("f0", 0.0),
        pros.get("jitter", 0.0),
        pros.get("shimmer", 0.0),
        pros.get("cpp", 0.0),
    ], dtype=np.float32)

    full_vec = np.concatenate(
        [mfcc, lfcc, pcen_mean, pcen_std, spec_vec, pros_vec],
        axis=0
    )

    # Đảm bảo đúng N_FEAT
    if full_vec.size < N_FEAT:
        out = np.zeros(N_FEAT, dtype=np.float32)
        out[:full_vec.size] = full_vec
        return out
    return full_vec[:N_FEAT]


# =========================
# 7️⃣  API BACKEND REALTIME: /status
# =========================
@app.route("/status", methods=["GET"])
def status():
    """
    Trả về event phân tích mới nhất để UI backend hiển thị.
    """
    return jsonify(LAST_EVENT)


# =========================
# 8️⃣  API LỊCH SỬ: /events (JSON)
# =========================
@app.route("/events", methods=["GET"])
def events():
    """
    Trả về danh sách các event gần nhất (JSON) để frontend /history dùng.
    Query param: ?limit=100 (default 50)
    """
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit, 1000))

    rows = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        except Exception as e:
            print("[EVENTS_READ_ERR]", e)

    # sắp xếp theo thời gian nếu có trường ts
    rows.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return jsonify(rows)


# =========================
# 9️⃣  DASHBOARD REALTIME: /dashboard
# =========================
@app.route("/dashboard", methods=["GET"])
def dashboard():
    html = """
    <!doctype html>
    <html lang="vi">
    <head>
      <meta charset="utf-8">
      <title>DeepVoice Guard – Backend Monitor</title>
      <style>
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f7;
          margin: 0;
          padding: 24px;
        }
        .card {
          max-width: 520px;
          margin: 0 auto;
          background: #fff;
          border-radius: 16px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.08);
          padding: 20px 24px 24px;
        }
        .title {
          font-size: 20px;
          font-weight: 600;
          margin-bottom: 4px;
        }
        .sub {
          font-size: 13px;
          color: #666;
          margin-bottom: 16px;
        }
        .dot {
          width: 12px;
          height: 12px;
          border-radius: 999px;
          margin-right: 8px;
        }
        .row {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
        }
        .meter {
          position: relative;
          height: 10px;
          border-radius: 999px;
          background: #e5e5ea;
          overflow: hidden;
          margin: 8px 0 4px;
        }
        .meter-fill {
          position: absolute;
          inset: 0;
          width: 0%;
          background: linear-gradient(90deg, #34c759, #ff3b30);
          transition: width 0.25s ease-out;
        }
        .label-row {
          display: flex;
          justify-content: space-between;
          font-size: 12px;
          color: #555;
          margin-bottom: 8px;
        }
        .reason {
          font-size: 13px;
          color: #333;
          margin-top: 8px;
          white-space: pre-wrap;
        }
        .flags {
          font-size: 12px;
          color: #555;
          margin-top: 4px;
        }
        .chip {
          display: inline-flex;
          align-items: center;
          padding: 2px 8px;
          border-radius: 999px;
          background: #f2f2f7;
          font-size: 11px;
          margin-right: 4px;
          margin-top: 4px;
        }
        .chip span {
          font-size: 10px;
          margin-right: 4px;
        }
        .meta {
          font-size: 11px;
          color: #888;
          margin-top: 8px;
        }
        .link-row {
          margin-top: 12px;
          font-size: 12px;
        }
        .link-row a {
          color: #007bff;
          text-decoration: none;
        }
        .link-row a:hover {
          text-decoration: underline;
        }
      </style>
    </head>
    <body>
      <div class="card">
        <div class="row">
          <div id="dot" class="dot" style="background:#34c759;"></div>
          <div>
            <div class="title">DeepVoice Guard – Backend Monitor</div>
            <div class="sub">Theo dõi các lần gọi /analyze từ Chrome Extension</div>
          </div>
        </div>

        <div class="label-row">
          <div>Xác suất giả mạo (prob_fused)</div>
          <div id="prob-label">0.000</div>
        </div>
        <div class="meter">
          <div id="meter-fill" class="meter-fill"></div>
        </div>

        <div class="label-row">
          <div>Level: <span id="level">green</span></div>
          <div>SNR: <span id="snr">0.0</span> dB</div>
        </div>

        <div class="reason" id="reasons">Chưa có dữ liệu. Hãy mở Google Meet và bật extension.</div>
        <div class="flags" id="flags"></div>
        <div class="meta" id="ts"></div>

        <div class="link-row">
          Xem lịch sử chi tiết: <a href="/history" target="_blank">/history</a>
        </div>
      </div>

      <script>
        function updateUI(data) {
          const prob = Number(data.prob_fused || 0);
          const level = data.level || "green";
          const snr = Number(data.snr || 0);
          const flags = data.flags || {};
          const reasons = data.reasons || [];
          const ts = data.ts || "";

          const fill = document.getElementById("meter-fill");
          const probLabel = document.getElementById("prob-label");
          const levelEl = document.getElementById("level");
          const snrEl = document.getElementById("snr");
          const dot = document.getElementById("dot");
          const reasonEl = document.getElementById("reasons");
          const flagsEl = document.getElementById("flags");
          const tsEl = document.getElementById("ts");

          const p = Math.max(0, Math.min(1, prob));
          fill.style.width = (p * 100).toFixed(1) + "%";
          probLabel.textContent = p.toFixed(3);

          levelEl.textContent = level;
          snrEl.textContent = snr.toFixed(1);

          if (level === "red") {
            dot.style.background = "#ff3b30";
          } else if (level === "amber") {
            dot.style.background = "#ff9500";
          } else {
            dot.style.background = "#34c759";
          }

          if (reasons.length > 0) {
            reasonEl.textContent = "• " + reasons.join("\\n• ");
          } else {
            reasonEl.textContent = "Không có lý do chi tiết (reasons trống).";
          }

          const flagKeys = Object.keys(flags).filter(k => flags[k]);
          if (flagKeys.length > 0) {
            flagsEl.innerHTML = flagKeys.map(k =>
              "<span class='chip'><span>⚑</span>" + k + "</span>"
            ).join(" ");
          } else {
            flagsEl.textContent = "";
          }

          tsEl.textContent = ts ? ("Last event: " + ts) : "";
        }

        async function poll() {
          try {
            const res = await fetch("/status");
            if (!res.ok) throw new Error("HTTP " + res.status);
            const data = await res.json();
            updateUI(data);
          } catch (e) {
            console.error(e);
          }
        }

        // Poll mỗi 1 giây
        poll();
        setInterval(poll, 1000);
      </script>
    </body>
    </html>
    """
    return html


# =========================
# 🔟  LỊCH SỬ ĐẸP: /history
# =========================
@app.route("/history", methods=["GET"])
def history_page():
    html = """
    <!doctype html>
    <html lang="vi">
    <head>
      <meta charset="utf-8">
      <title>DeepVoice Guard – History</title>
      <style>
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f7;
          margin: 0;
          padding: 24px;
        }
        .container {
          max-width: 920px;
          margin: 0 auto;
          background: #fff;
          border-radius: 16px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.08);
          padding: 20px 24px 24px;
        }
        h1 {
          font-size: 20px;
          margin-top: 0;
          margin-bottom: 4px;
        }
        .sub {
          font-size: 13px;
          color: #666;
          margin-bottom: 16px;
        }
        table {
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }
        th, td {
          border-bottom: 1px solid #eee;
          padding: 6px 8px;
          text-align: left;
          vertical-align: top;
        }
        th {
          background: #f9f9fb;
          font-weight: 600;
        }
        tr:nth-child(even) td {
          background: #fafafa;
        }
        .badge {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 11px;
          color: #fff;
        }
        .badge.green { background: #34c759; }
        .badge.amber { background: #ff9500; }
        .badge.red { background: #ff3b30; }
        .flags {
          font-size: 11px;
          color: #555;
        }
        .flag-chip {
          display: inline-block;
          padding: 1px 6px;
          border-radius: 999px;
          background: #f2f2f7;
          margin-right: 4px;
          margin-top: 2px;
        }
        .reasons {
          white-space: pre-wrap;
        }
        .toolbar {
          margin-bottom: 12px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 12px;
        }
        select, input {
          font-size: 12px;
          padding: 3px 6px;
          border-radius: 8px;
          border: 1px solid #ccc;
          outline: none;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>DeepVoice Guard – Lịch sử phân tích</h1>
        <div class="sub">Đọc từ Logs/events.jsonl (backend_flask/Logs/events.jsonl)</div>

        <div class="toolbar">
          <div>
            Hiển thị:
            <select id="limit">
              <option value="20">20</option>
              <option value="50" selected>50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
            bản ghi mới nhất
          </div>
          <div>
            Bộ lọc level:
            <select id="filter-level">
              <option value="">Tất cả</option>
              <option value="green">green</option>
              <option value="amber">amber</option>
              <option value="red">red</option>
            </select>
          </div>
        </div>

        <table>
          <thead>
            <tr>
              <th>Thời gian (UTC)</th>
              <th>Prob</th>
              <th>Level</th>
              <th>SNR (dB)</th>
              <th>Flags</th>
              <th>Reasons</th>
            </tr>
          </thead>
          <tbody id="tbody">
            <tr><td colspan="6">Đang tải dữ liệu...</td></tr>
          </tbody>
        </table>
      </div>

      <script>
        async function loadData() {
          const limit = document.getElementById("limit").value;
          const filterLevel = document.getElementById("filter-level").value;
          const tbody = document.getElementById("tbody");
          tbody.innerHTML = "<tr><td colspan='6'>Đang tải dữ liệu...</td></tr>";

          try {
            const res = await fetch("/events?limit=" + encodeURIComponent(limit));
            if (!res.ok) throw new Error("HTTP " + res.status);
            let data = await res.json();

            if (filterLevel) {
              data = data.filter(row => row.level === filterLevel);
            }

            if (!data.length) {
              tbody.innerHTML = "<tr><td colspan='6'>Không có dữ liệu.</td></tr>";
              return;
            }

            const rowsHtml = data.map(ev => {
              const ts = ev.ts || "";
              const prob = Number(ev.prob_fused || 0).toFixed(3);
              const level = ev.level || "green";
              const snr = Number(ev.snr || 0).toFixed(1);
              const reasons = (ev.reasons || []).map(r => "• " + r).join("\\n");
              const flags = ev.flags || {};
              const flagKeys = Object.keys(flags).filter(k => flags[k]);

              let badgeClass = "green";
              if (level === "red") badgeClass = "red";
              else if (level === "amber") badgeClass = "amber";

              const flagsHtml = flagKeys.length
                ? flagKeys.map(k => "<span class='flag-chip'>" + k + "</span>").join(" ")
                : "";

              return `
                <tr>
                  <td>${ts}</td>
                  <td>${prob}</td>
                  <td><span class="badge ${badgeClass}">${level}</span></td>
                  <td>${snr}</td>
                  <td class="flags">${flagsHtml}</td>
                  <td class="reasons">${reasons}</td>
                </tr>
              `;
            }).join("");

            tbody.innerHTML = rowsHtml;
          } catch (e) {
            console.error(e);
            tbody.innerHTML = "<tr><td colspan='6'>Lỗi tải dữ liệu.</td></tr>";
          }
        }

        document.getElementById("limit").addEventListener("change", loadData);
        document.getElementById("filter-level").addEventListener("change", loadData);

        loadData();
      </script>
    </body>
    </html>
    """
    return html


# =========================
# 1️⃣1️⃣  MAIN
# =========================
if __name__ == "__main__":
    # debug=True để tiện dev, khi deploy thật nên để False
    app.run(host="0.0.0.0", port=5000, debug=True)
