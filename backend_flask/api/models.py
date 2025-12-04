# backend_flask/api/models.py
import os
import json
import joblib
import numpy as np

# ================== ĐƯỜNG DẪN CỐ ĐỊNH ==================
# __file__ đang là .../backend_flask/api/models.py
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # -> backend_flask

MODEL_DIR = os.path.join(BASE_DIR, "Models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEAT_PATH = os.path.join(MODEL_DIR, "fast_features.json")
SCALER_STATS_PATH = os.path.join(MODEL_DIR, "fast_scaler_stats.json")
RES2NET_METRICS_PATH = os.path.join(MODEL_DIR, "res2net_metrics.json")

LOG_DIR = os.path.join(BASE_DIR, "Logs")
LOG_PATH = os.path.join(LOG_DIR, "events.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ================== METADATA: N_FEAT, SCALER, RES2NET ==================
if os.path.exists(FEAT_PATH):
    with open(FEAT_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
    N_FEAT = len(_cfg.get("features", [])) or 196
else:
    N_FEAT = 196  # fallback

if os.path.exists(SCALER_STATS_PATH):
    with open(SCALER_STATS_PATH, "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)
else:
    scaler_stats = {}

if os.path.exists(RES2NET_METRICS_PATH):
    with open(RES2NET_METRICS_PATH, "r", encoding="utf-8") as f:
        res2net_metrics = json.load(f)
else:
    res2net_metrics = {}

# ================== NẠP MODEL NHANH (PIPELINE) ==================
_model = None
_has_model = None


def load_fast_model():
    """
    Trả về (model, has_model):
      - model: Pipeline (StandardScaler + XGBoost) hoặc DummyModel
      - has_model: True nếu có file thật xgb_model.pkl
    Hàm có cache, chỉ load từ disk 1 lần.
    """
    global _model, _has_model
    if _model is not None:
        return _model, _has_model

    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        _has_model = True
        print("✅ Fast model pipeline loaded:", MODEL_PATH)
    else:
        print("⚠️ Chưa có model thật, dùng DummyModel tạm (chỉ để test API).")

        class DummyModel:
            def predict_proba(self, X):
                mean = float(np.mean(X))
                p = 1.0 / (1.0 + np.exp(-5 * mean))
                return np.array([[1 - p, p]], dtype=np.float32)

        _model = DummyModel()
        _has_model = False

    return _model, _has_model
