# backend_flask/api/__init__.py
"""
Package api: chứa logic mô hình & phân tích cho DeepVoice Guard.
"""

from .models import (
    BASE_DIR,
    MODEL_DIR,
    MODEL_PATH,
    FEAT_PATH,
    SCALER_STATS_PATH,
    RES2NET_METRICS_PATH,
    LOG_DIR,
    LOG_PATH,
    N_FEAT,
    scaler_stats,
    res2net_metrics,
    load_fast_model,
)

from .analyze import (
    analyze_features,
    extract_feature_vector,
    quick_flags,
    get_last_event,
)
from .fusion import decide_level, heuristic_prob, fuse_probs

__all__ = [
    # models
    "BASE_DIR",
    "MODEL_DIR",
    "MODEL_PATH",
    "FEAT_PATH",
    "SCALER_STATS_PATH",
    "RES2NET_METRICS_PATH",
    "LOG_DIR",
    "LOG_PATH",
    "N_FEAT",
    "scaler_stats",
    "res2net_metrics",
    "load_fast_model",
    # analyze
    "analyze_features",
    "extract_feature_vector",
    "quick_flags",
    "get_last_event",
    # fusion
    "decide_level",
    "heuristic_prob",
    "fuse_probs",
]
