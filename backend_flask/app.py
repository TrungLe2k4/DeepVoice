# backend_flask/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =========================
# 1️⃣  NẠP MÔ HÌNH ĐÃ TRAIN
# =========================

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Models", "xgb_model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "Models", "fast_features.json")

has_model = os.path.exists(MODEL_PATH)

if has_model:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded:", MODEL_PATH)
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

        # Chuyển về vector 1D chuẩn (196 phần tử) cho model
        vec = extract_feature_vector(feats)

        if vec.size != N_FEAT:
            return jsonify({
                "error": f"Sai số lượng đặc trưng: cần {N_FEAT}, nhận {vec.size}"
            }), 400

        # Dự đoán xác suất Deepfake
        prob = float(model.predict_proba(vec.reshape(1, -1))[0, 1])

        # Phân tích nhanh spec/prosody để sinh flags + extra reasons + snr
        flags, extra_reasons, snr = quick_flags(feats, prob)

        # Gán mức cảnh báo chính
        if prob >= 0.85:
            level = "red"
            base_reason = "Tín hiệu tổng hợp rõ rệt (MFCC/LFCC/PCEN lệch chuẩn)."
        elif prob >= 0.6:
            level = "amber"
            base_reason = "Có dấu hiệu bất thường trong MFCC/LFCC/PCEN."
        else:
            level = "green"
            base_reason = "An toàn: chưa thấy dấu hiệu giả mạo rõ ràng."

        # Gộp reason: 1 reason chính + các reason phụ (loại trùng)
        reasons = [base_reason]
        for r in extra_reasons:
            if r and r not in reasons:
                reasons.append(r)

        # 🔴 Ghi log mỗi lần gọi /analyze
        log_event(feats, prob, level, flags, snr)

        return jsonify({
            "prob_fast": prob,
            "prob_deep": prob * 0.95,    # tạm thời reuse fast cho demo
            "prob_embed": prob * 0.90,
            "prob_fused": prob,
            "reason": reasons,
            "level": level,
            "snr": snr,
            "flags": flags,
            "version": "dv-1.0.0"
        })

    except Exception as e:
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
def log_event(feats, prob, level, flags, snr):
    """
    Ghi lại mỗi lần /analyze vào file JSONL:
      backend_flask/Logs/events.jsonl

    Mỗi dòng là một JSON:
      {
        "ts": "...",
        "prob": ...,
        "level": "...",
        "snr": ...,
        "flags": {...},
        "spec": {...},
        "prosody": {...},
        "meta": {...}
      }
    """
    try:
        log_dir = os.path.join(BASE_DIR, "Logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "events.jsonl")

        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "prob": float(prob),
            "level": level,
            "snr": float(snr),
            "flags": flags or {},
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

        with open(log_path, "a", encoding="utf-8") as f:
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
    pcen_std  = safe(pcen.get("std"), 64)

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
# 7️⃣  MAIN
# =========================
if __name__ == "__main__":
    # debug=True để tiện dev, khi deploy thật nên để False
    app.run(host="0.0.0.0", port=5000, debug=True)
