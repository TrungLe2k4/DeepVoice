# backend_flask/api/fusion.py
from __future__ import annotations
import numpy as np


def decide_level(prob_fused: float) -> str:
    """
    Gán level dựa trên xác suất giả mạo đã fusion.
    Giữ nguyên ngưỡng bạn đang dùng.
    """
    if prob_fused >= 0.85:
        return "red"
    if prob_fused >= 0.6:
        return "amber"
    return "green"


def heuristic_prob(feats, snr: float) -> float:
    """
    Heuristic đơn giản dựa trên spec + SNR để hỗ trợ mô hình:
      - flat cao
      - entropy thấp
      - snr cao
    """
    entropy = 0.0
    flat = 0.0

    if isinstance(feats, dict):
        spec = feats.get("spec", {}) or {}
        entropy = float(spec.get("entropy", 0.0))
        flat = float(spec.get("flat", 0.0))

    h = (
        0.5 * flat
        + 0.3 * max(0.0, 1.0 - entropy)
        + 0.2 * max(0.0, (snr - 10.0) / 30.0)
    )
    return float(np.clip(h, 0.0, 1.0))


def fuse_probs(prob_fast: float, prob_deep: float, prob_heur: float) -> float:
    """
    Kết hợp fast model + deep model + heuristic.
    """
    return float(0.6 * prob_fast + 0.3 * prob_deep + 0.1 * prob_heur)
