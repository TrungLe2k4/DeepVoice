# backend_flask/api/analyze.py
from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from .models import (
    N_FEAT,
    LOG_PATH,
    load_fast_model,
)
from .fusion import decide_level, heuristic_prob, fuse_probs

# ================== EVENT CUỐI CÙNG (cho /status) ==================
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


def get_last_event():
    """Cho /status dùng."""
    return LAST_EVENT


# ================== GHÉP VECTOR ĐẶC TRƯNG ==================
def extract_feature_vector(feats, n_feat: int = N_FEAT) -> np.ndarray:
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
    # Trường hợp gửi thẳng mảng phẳng
    if isinstance(feats, (list, tuple, np.ndarray)):
        arr = np.asarray(feats, dtype=np.float32).ravel()
        out = np.zeros(n_feat, dtype=np.float32)
        out[: min(arr.size, n_feat)] = arr[:n_feat]
        return out

    # Không phải dict → trả vector 0, vẫn cho model chạy được (nhưng không khuyến khích)
    if not isinstance(feats, dict):
        return np.zeros(n_feat, dtype=np.float32)

    def safe(arr, n):
        if not isinstance(arr, (list, np.ndarray)):
            return np.zeros(n, dtype=np.float32)
        a = np.asarray(arr, dtype=np.float32).ravel()
        out = np.zeros(n, dtype=np.float32)
        out[: min(a.size, n)] = a[:n]
        return out

    mfcc = safe(feats.get("mfcc"), 39)
    lfcc = safe(feats.get("lfcc"), 20)

    pcen = feats.get("pcen_stats", {}) or {}
    pcen_mean = safe(pcen.get("mean"), 64)
    pcen_std = safe(pcen.get("std"), 64)

    spec = feats.get("spec", {}) or {}
    spec_vec = np.array(
        [
            spec.get("zcr", 0.0),
            spec.get("flat", 0.0),
            spec.get("rolloff", 0.0),
            spec.get("entropy", 0.0),
            spec.get("contrast", 0.0),
        ],
        dtype=np.float32,
    )

    pros = feats.get("prosody", {}) or {}
    pros_vec = np.array(
        [
            pros.get("f0", 0.0),
            pros.get("jitter", 0.0),
            pros.get("shimmer", 0.0),
            pros.get("cpp", 0.0),
        ],
        dtype=np.float32,
    )

    full_vec = np.concatenate(
        [mfcc, lfcc, pcen_mean, pcen_std, spec_vec, pros_vec],
        axis=0,
    )

    out = np.zeros(n_feat, dtype=np.float32)
    out[: min(full_vec.size, n_feat)] = full_vec[:n_feat]
    return out


# ================== PHÂN TÍCH NHANH -> FLAGS ==================
def quick_flags(feats, prob):
    """
    Phân tích nhanh một số đặc trưng để sinh:
      - flags: {too_clean, robotic_prosody, high_zcr, weird_f0, ...}
      - extra_reasons: list[str] cho overlay
      - snr: số dB (nếu có)
    """
    flags = {}
    extra = []
    snr = 0.0

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
            extra.append(
                "Phổ tần số rất sạch & phẳng (nghi ngờ tổng hợp)."
            )

        # 2) Prosody robot: jitter/shimmer rất thấp, CPP cao
        if jitter < 0.5 and shimmer < 0.5 and cpp > 8:
            flags["robotic_prosody"] = True
            extra.append(
                "Độ run & biên độ giọng rất thấp, formant ổn định bất thường."
            )

        # 3) ZCR cao
        if zcr > 0.25:
            flags["high_zcr"] = True
            extra.append(
                "Zero-crossing rate cao, có thể là tín hiệu tổng hợp / nhiễu lạ."
            )

        # 4) F0 lạ
        if 0 < f0 < 60 or f0 > 400:
            flags["weird_f0"] = True
            extra.append("Tần số cơ bản nằm ngoài dải giọng người điển hình.")

        # 5) Nếu prob thấp & không có flag nào → reassure
        if prob < 0.4 and not flags:
            extra.append("Đặc trưng ổn định, phù hợp giọng nói tự nhiên.")
    else:
        if prob < 0.4:
            extra.append("Đặc trưng tổng thể ở mức an toàn.")

    return flags, extra, snr


# ================== GHI LOG EVENT ==================
def log_event(feats, prob_fused, level, flags, snr, reasons, alert=False):
    """
    Ghi lại mỗi lần /analyze vào file JSONL:
      backend_flask/Logs/events.jsonl
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

        if isinstance(feats, dict):
            event["spec"] = feats.get("spec", {})
            event["prosody"] = feats.get("prosody", {})
            event["meta"] = feats.get("meta", {})
        else:
            try:
                event["raw_dim"] = int(np.asarray(feats).size)
            except Exception:
                event["raw_dim"] = 0

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[LOG_EVENT_ERR]", e)


# ================== HÀM CHÍNH: PHÂN TÍCH 1 REQUEST ==================
def analyze_features(feats):
    """
    Hàm core dùng chung cho Flask & script:
      - nhận dict/list 'feats'
      - trả về dict kết quả (prob, level, flags, reasons, ...)
    """
    # 1) Ghép vector cho fast model
    vec = extract_feature_vector(feats, N_FEAT)

    # 2) Load model nhanh
    model, _ = load_fast_model()

    # 3) Dự đoán xác suất FAKE
    prob_fast = float(model.predict_proba(vec.reshape(1, -1))[0, 1])

    # 4) Flags + snr
    flags, extra_reasons, snr = quick_flags(feats, prob_fast)

    # 5) Heuristic phụ trợ
    prob_heur = heuristic_prob(feats, snr)

    # 6) Deep / embed (tạm thời reuse fast – sau này thay bằng Res2Net)
    prob_deep = prob_fast
    prob_embed = prob_fast

    # 7) Fusion
    prob_fused = fuse_probs(prob_fast, prob_deep, prob_heur)

    # 8) Level + reason
    level = decide_level(prob_fused)
    if level == "red":
        base_reason = (
            "Tín hiệu tổng hợp rõ rệt (MFCC/LFCC/PCEN lệch chuẩn)."
        )
    elif level == "amber":
        base_reason = "Có dấu hiệu bất thường trong MFCC/LFCC/PCEN."
    else:
        base_reason = "An toàn: chưa thấy dấu hiệu giả mạo rõ ràng."

    reasons = [base_reason]
    for r in extra_reasons:
        if r and r not in reasons:
            reasons.append(r)

    # 9) Có bật alert mạnh hay không
    alert = False
    if level == "red":
        alert = True
    elif level == "amber" and any(
        flags.get(k) for k in ("too_clean", "robotic_prosody", "high_zcr")
    ):
        alert = True

    # 10) In log console
    print(
        f"[ANALYZE] prob_fast={prob_fast:.3f} "
        f"prob_fused={prob_fused:.3f} level={level} snr={snr:.1f} "
        f"flags={list(flags.keys())} alert={alert}"
    )

    # 11) Ghi log file
    log_event(feats, prob_fused, level, flags, snr, reasons, alert)

    # 12) Cập nhật LAST_EVENT cho /status
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

    return {
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
        "version": "dv-1.0.0",
    }
