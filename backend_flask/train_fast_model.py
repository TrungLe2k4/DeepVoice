# backend_flask/train_fast_model.py
import os
import json
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pandas as pd

from scipy.fft import dct  # DCT từ SciPy
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# ======= Cấu hình đường dẫn =======
DATA_CLEANED   = r"D:\DeepVoice\Data\Cleaned"          # chứa real/ và fake/
METADATA_CSV   = r"D:\DeepVoice\Data\metadata_master.csv"

BASE_DIR       = os.path.dirname(__file__)
MODEL_DIR      = os.path.join(BASE_DIR, "Models")
MODEL_PATH     = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEATURES_JSON  = os.path.join(MODEL_DIR, "fast_features.json")
SCALER_JSON    = os.path.join(MODEL_DIR, "fast_scaler_stats.json")

# ======= Tham số DSP =======
SR          = 16000
NFFT        = 1024
HOP         = 256
NMELS       = 64
LFCC_BANDS  = 40
MFCC_CEPS   = 13
LFCC_CEPS   = 20

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Pad / trim về đúng 5s ----------
def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad hoặc cắt tín hiệu audio về đúng độ dài target_len mẫu.
    Dùng thay cho librosa.util.fix_length (tránh lỗi version).
    """
    cur_len = len(y)
    if cur_len > target_len:
        return y[:target_len]
    if cur_len < target_len:
        pad_width = target_len - cur_len
        return np.pad(y, (0, pad_width), mode="constant")
    return y


# ---------- LFCC tuyến tính ----------
def lfcc(y, sr=SR, n_fft=NFFT, hop_length=HOP,
         n_bands=LFCC_BANDS, n_ceps=LFCC_CEPS):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freq_bins = S.shape[0]

    edges = np.linspace(0, freq_bins - 1, n_bands + 2).astype(int)
    fb = np.zeros((n_bands, freq_bins), dtype=np.float32)

    for b in range(n_bands):
        left, center, right = edges[b], edges[b + 1], edges[b + 2]
        if center <= left or right <= center:
            continue

        up = np.linspace(0, 1, max(center - left, 1), endpoint=False)
        down = np.linspace(1, 0, max(right - center, 1), endpoint=True)

        fb[b, left:center] = up[:max(center - left, 0)]
        fb[b, center:right] = down[:max(right - center, 0)]

    E = fb @ (S ** 2)
    E[E <= 1e-12] = 1e-12
    logE = np.log(E)

    ceps = dct(logE, type=2, axis=0, norm="ortho")
    ceps = ceps[:n_ceps, :]
    return ceps.mean(axis=1)


# ---------- PCEN mean/std ----------
def pcen_mean_std(y, sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMELS):
    M = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=50, fmax=8000, power=1.0
    )
    P = librosa.pcen(M, sr=sr, hop_length=hop_length)
    return P.mean(axis=1), P.std(axis=1)


# ---------- Spectral stats ----------
def spectral_stats(y, sr=SR, n_fft=NFFT, hop_length=HOP):
    zcr = float(librosa.feature.zero_crossing_rate(y)[0].mean())

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    flat = float(librosa.feature.spectral_flatness(S=S)[0].mean())
    roll = float(
        librosa.feature.spectral_rolloff(
            S=S, sr=sr, roll_percent=0.85
        )[0].mean() / (sr / 2)
    )

    power = (S ** 2).mean(axis=1)
    P = power / (power.sum() + 1e-12)

    blocks = np.array_split(P, 10)
    ent = 0.0
    for b in blocks:
        pb = float(b.sum())
        if pb > 0:
            ent += -pb * np.log2(pb)
    ent /= np.log2(10)

    bands = np.array_split(power, 6)
    contr = float(
        np.mean([(b.max() - b.min()) / (b.mean() + 1e-9) for b in bands])
    )

    return zcr, flat, roll, ent, contr


# ---------- Prosody ----------
def prosody_feats(y, sr=SR):
    try:
        f0 = librosa.yin(
            y, fmin=60, fmax=400, sr=sr,
            frame_length=2048, hop_length=256
        )
        f0 = f0[np.isfinite(f0)]
        f0_val = float(np.median(f0)) if f0.size else 0.0
        jitter = float(
            np.std(np.diff(f0)) / (np.mean(f0) + 1e-9) * 100
        ) if f0.size > 5 else 0.0
    except Exception:
        f0_val, jitter = 0.0, 0.0

    env = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
    shimmer = float(
        np.std(np.diff(env)) / (np.mean(env) + 1e-9) * 100
    ) if env.size > 5 else 0.0

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    flat = float(librosa.feature.spectral_flatness(S=S)[0].mean())

    power = (S ** 2).mean(axis=1)
    P = power / (power.sum() + 1e-12)
    blocks = np.array_split(P, 10)
    ent = 0.0
    for b in blocks:
        pb = float(b.sum())
        if pb > 0:
            ent += -pb * np.log2(pb)
    ent /= np.log2(10)

    cpp = float(max(0.0, 20.0 - 10.0 * ent - 5.0 * flat))
    return f0_val, jitter, shimmer, cpp


# ---------- MFCC 39 chiều ----------
def mfcc39(y, sr=SR, n_fft=NFFT, hop_length=HOP, n_mfcc=MFCC_CEPS):
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length
    )
    d = librosa.feature.delta(mfcc)
    dd = librosa.feature.delta(mfcc, order=2)

    vec = np.concatenate(
        [mfcc.mean(axis=1), d.mean(axis=1), dd.mean(axis=1)],
        axis=0
    )
    return vec.astype(np.float32)


# ---------- Ghép full vector 196D ----------
def extract_vector(y, sr=SR):
    """
    Vector 196 chiều:
      39 MFCC (mean+delta+delta2)
      20 LFCC
      64 PCEN mean
      64 PCEN std
      5 spec: zcr, flat, rolloff, entropy, contrast
      4 prosody: f0, jitter, shimmer, cpp
    """
    m39 = mfcc39(y, sr)
    l20 = lfcc(y, sr)
    mu, sd = pcen_mean_std(y, sr)
    zcr, flat, roll, ent, contr = spectral_stats(y, sr)
    f0, jitter, shimmer, cpp = prosody_feats(y, sr)

    tail = np.array(
        [zcr, flat, roll, ent, contr, f0, jitter, shimmer, cpp],
        dtype=np.float32
    )

    vec = np.concatenate([m39, l20, mu, sd, tail], axis=0)
    return vec


# ---------- Load dataset từ metadata_master.csv ----------
def load_dataset_from_metadata(clean_root: str, meta_csv: str):
    """
    Đọc metadata_master.csv:
      - file_path: "real/xxx.wav" hoặc "fake/yyy.wav"
      - label: "real"/"fake"
      - set: "train"/"val"/"test"
    Trả về:
      X: (N, 196)
      y: (N,)
      set_arr: mảng set tương ứng từng mẫu
    """
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"Không tìm thấy metadata CSV: {meta_csv}")

    df = pd.read_csv(meta_csv)
    if not {"file_path", "label", "set"}.issubset(df.columns):
        raise ValueError("metadata_master.csv phải có cột: file_path, label, set")

    X_list, y_list, set_list = [], [], []

    for _, row in df.iterrows():
        rel_path = str(row["file_path"])
        label_str = str(row["label"]).lower()
        set_str = str(row["set"]).lower()

        full_path = os.path.join(clean_root, rel_path)
        if not os.path.isfile(full_path):
            print(f"⚠️ Mất file audio, bỏ qua: {full_path}")
            continue

        yi = 1 if label_str == "fake" else 0

        try:
            sig, sr = sf.read(full_path, dtype="float32")
            if sig.ndim > 1:
                sig = sig[:, 0]

            if sr != SR:
                sig = librosa.resample(sig, orig_sr=sr, target_sr=SR)

            target_len = SR * 5
            sig = pad_or_trim(sig, target_len)

            feat = extract_vector(sig, SR)
            X_list.append(feat)
            y_list.append(yi)
            set_list.append(set_str)

        except Exception as e:
            print(f"⚠️ lỗi đọc/feature {full_path}: {e}")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    set_arr = np.array(set_list)

    return X, y, set_arr


# ---------- MAIN ----------
def main():
    print("📂 Đang đọc dữ liệu từ:", DATA_CLEANED)
    print("📄 Metadata:", METADATA_CSV)
    X_all, y_all, set_all = load_dataset_from_metadata(DATA_CLEANED, METADATA_CSV)
    print("✅ Tổng mẫu:", X_all.shape, " | Tỷ lệ FAKE:", float(y_all.mean()))

    # Tách theo set: train+val để train/CV, test để đánh giá cuối
    train_mask = np.isin(set_all, ["train", "val"])
    test_mask = (set_all == "test")

    X_tr = X_all[train_mask]
    y_tr = y_all[train_mask]
    X_te = X_all[test_mask]
    y_te = y_all[test_mask]

    print(f"📊 Train+Val: {X_tr.shape[0]} mẫu | Test: {X_te.shape[0]} mẫu")

    # Ưu tiên XGBoost; nếu không có thì dùng RF
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=1.0,
            min_child_weight=5,
            gamma=0.5,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=4,
            scale_pos_weight=(y_tr == 0).sum()
                             / max(1, (y_tr == 1).sum()),
        )
        print("🧠 Sử dụng XGBoost (có regularization)")
    except Exception:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            n_jobs=4,
            random_state=42,
            class_weight="balanced",
        )
        print("🧠 XGBoost không có, dùng RandomForest")

    # 🔹 Pipeline có bước chuẩn hóa StandardScaler
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])

    # 🔁 Cross-validation trên train+val (không đụng test)
    print("🔁 Đang chạy 5-fold cross-validation (AUC) trên train+val...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipe, X_tr, y_tr, cv=skf, scoring="roc_auc"
    )
    print("📊 AUC từng fold:", cv_scores)
    print("📊 AUC trung bình:", cv_scores.mean(), "±", cv_scores.std())

    # 🚀 Train final model trên toàn bộ train+val
    print("🚀 Đang train model trên train+val...")
    pipe.fit(X_tr, y_tr)

    # 🎯 Đánh giá trên tập test hold-out
    if X_te.shape[0] > 0:
        prob = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, prob)
        pred = (prob >= 0.5).astype(int)

        print(f"🎯 AUC (test): {auc:.4f}")
        print("📄 Classification report (test):")
        print(classification_report(y_te, pred, digits=4))
    else:
        print("⚠️ Không có mẫu 'test' trong metadata, bỏ qua đánh giá hold-out.")

    # 💾 Lưu pipeline (gồm cả scaler + model)
    joblib.dump(pipe, MODEL_PATH)
    print("💾 Đã lưu model (Pipeline) tại:", MODEL_PATH)

    # 🧾 Lưu danh sách đặc trưng
    feat_names = (
        [f"mfcc_{i}" for i in range(39)] +
        [f"lfcc_{i}" for i in range(20)] +
        [f"pcen_mu_{i}" for i in range(64)] +
        [f"pcen_sd_{i}" for i in range(64)] +
        ["zcr", "flatness", "rolloff", "entropy",
         "contrast", "f0", "jitter", "shimmer", "cpp"]
    )
    with open(FEATURES_JSON, "w", encoding="utf-8") as f:
        json.dump({"features": feat_names}, f, ensure_ascii=False, indent=2)
    print("🧾 Lưu danh sách đặc trưng:", FEATURES_JSON)

    # 📐 Lưu thống kê StandardScaler để dùng trong báo cáo / kiểm tra
    try:
        scaler = pipe.named_steps["scaler"]
        scaler_info = {
            "n_features_in_": int(
                getattr(scaler, "n_features_in_", len(feat_names))
            ),
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "var_": scaler.var_.tolist() if hasattr(scaler, "var_") else None,
            "feature_names": feat_names,
        }
        with open(SCALER_JSON, "w", encoding="utf-8") as f:
            json.dump(scaler_info, f, ensure_ascii=False, indent=2)
        print("📐 Đã lưu thống kê StandardScaler:", SCALER_JSON)
    except Exception as e:
        print("⚠️ Không lưu được scaler stats:", e)


if __name__ == "__main__":
    main()
