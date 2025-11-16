# backend_flask/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import json

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

        # Chuyển về vector 1D chuẩn (196 phần tử)
        vec = extract_feature_vector(feats)

        if vec.size != N_FEAT:
            return jsonify({
                "error": f"Sai số lượng đặc trưng: cần {N_FEAT}, nhận {vec.size}"
            }), 400

        # Dự đoán xác suất Deepfake
        prob = float(model.predict_proba(vec.reshape(1, -1))[0, 1])

        # Gán mức cảnh báo
        if prob >= 0.85:
            level = "red"
            reason = "Tín hiệu tổng hợp rõ rệt (formant drift, PCEN phẳng)."
        elif prob >= 0.6:
            level = "amber"
            reason = "Có dấu hiệu bất thường trong MFCC/LFCC/PCEN."
        else:
            level = "green"
            reason = "An toàn: chưa thấy dấu hiệu giả mạo rõ ràng."

        return jsonify({
            "prob_fast": prob,
            "prob_deep": prob * 0.95,   # tạm thời reuse fast cho demo
            "prob_embed": prob * 0.90,
            "prob_fused": prob,
            "reason": [reason],
            "level": level,
            "version": "dv-1.0.0"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 4️⃣  HÀM PHỤ TRỢ
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
# 5️⃣  MAIN
# =========================
if __name__ == "__main__":
    # debug=True để tiện dev, khi deploy thật nên để False
    app.run(host="0.0.0.0", port=5000, debug=True)
