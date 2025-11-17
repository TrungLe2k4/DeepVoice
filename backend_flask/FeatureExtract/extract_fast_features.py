# FeatureExtract/extract_fast_features.py
# """
# Trích xuất đặc trưng "fast" 196 chiều cho DeepVoice Guard:

# - 39 MFCC (13 static + 13 delta + 13 delta-delta)
# - 20 LFCC (linear filterbank cepstra)
# - 64 PCEN mean + 64 PCEN std (trên cửa sổ ~2s, giống worklet)
# - 5 spectral stats: zcr, flatness, rolloff, entropy, contrast
# - 4 prosody: f0, jitter, shimmer, cpp

# Từ các file WAV trong:

#   {data_root}/real/*.wav  --> label 0
#   {data_root}/fake/*.wav  --> label 1

# Usage (từ thư mục backend_flask):

#   python -m FeatureExtract.extract_fast_features \
#       --data-root 1_data \
#       --out Models/fast_dataset.npz \
#       --sr 16000
# """

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ====================== DSP HELPERS (port từ worklet) ======================

def hanning(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(nfft: int,
                   sr: int,
                   n_mels: int = 64,
                   fmin: float = 50.0,
                   fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    hz = np.array([mel_to_hz(m) for m in m_pts], dtype=np.float32)
    bins = np.floor(((nfft + 1) * hz) / sr).astype(int)

    fb = np.zeros((n_mels, nfft // 2), dtype=np.float32)
    for m in range(1, n_mels + 1):
        for k in range(bins[m - 1], bins[m]):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - bins[m - 1]) / max(1, bins[m] - bins[m - 1])
        for k in range(bins[m], bins[m + 1]):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (bins[m + 1] - k) / max(1, bins[m + 1] - bins[m])
    return fb


def linear_filterbank(nfft: int,
                      sr: int,
                      n_bands: int = 40,
                      fmin: float = 50.0,
                      fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    hz = np.linspace(fmin, fmax, n_bands + 2, dtype=np.float32)
    bins = np.floor(((nfft + 1) * hz) / sr).astype(int)

    fb = np.zeros((n_bands, nfft // 2), dtype=np.float32)
    for b in range(1, n_bands + 1):
        for k in range(bins[b - 1], bins[b]):
            if 0 <= k < fb.shape[1]:
                fb[b - 1, k] = (k - bins[b - 1]) / max(1, bins[b] - bins[b - 1])
        for k in range(bins[b], bins[b + 1]):
            if 0 <= k < fb.shape[1]:
                fb[b - 1, k] = (bins[b + 1] - k) / max(1, bins[b + 1] - bins[b])
    return fb


def apply_fb(mag: np.ndarray, fb: np.ndarray) -> np.ndarray:
    # mag: (n_fft/2,), fb: (n_bands, n_fft/2)
    return np.maximum(1e-12, fb @ mag[: fb.shape[1]])


def dct_ii(x: np.ndarray, k_count: int) -> np.ndarray:
    N = x.shape[0]
    K = min(k_count, N)
    out = np.zeros(K, dtype=np.float32)
    factor = math.pi / N
    for k in range(K):
        n = np.arange(N, dtype=np.float32)
        out[k] = float(np.sum(x * np.cos((n + 0.5) * k * factor)))
    return out


def pcen_create_state(n_bands: int,
                      alpha: float = 0.98,
                      delta: float = 2.0,
                      r: float = 0.5,
                      eps: float = 1e-6,
                      ema_beta: float = 0.1):
    return {
        "alpha": alpha,
        "delta": delta,
        "r": r,
        "eps": eps,
        "ema_beta": ema_beta,
        "ema": np.zeros(n_bands, dtype=np.float32),
    }


def pcen_apply(band_pow: np.ndarray, state) -> np.ndarray:
    ema = state["ema"]
    alpha = state["alpha"]
    delta = state["delta"]
    r = state["r"]
    eps = state["eps"]
    beta = state["ema_beta"]

    ema[:] = (1.0 - beta) * ema + beta * band_pow
    norm = band_pow / np.power(eps + ema, alpha)
    out = np.power(norm + delta, r) - np.power(delta, r)
    return out.astype(np.float32)


def zcr(frame: np.ndarray) -> float:
    s = np.sign(frame)
    s[s == 0] = 1
    return float(np.mean(s[:-1] * s[1:] < 0))


def spectral_flatness(mag: np.ndarray) -> float:
    eps = 1e-12
    p = mag * mag + eps
    geo = float(np.exp(np.mean(np.log(p))))
    arith = float(np.mean(p) + eps)
    return geo / arith


def spectral_rolloff(mag: np.ndarray, roll: float = 0.85) -> float:
    total = float(np.sum(mag))
    if total <= 0:
        return 1.0
    threshold = total * roll
    acc = np.cumsum(mag)
    idx = int(np.searchsorted(acc, threshold))
    return min(1.0, idx / max(1, mag.shape[0]))


def spectral_entropy(mag: np.ndarray, n_blocks: int = 10) -> float:
    N = mag.shape[0]
    eps = 1e-12
    total = float(np.sum(mag))
    if total <= 0:
        return 0.0
    block = max(1, N // n_blocks)
    H = 0.0
    for b in range(n_blocks):
        st = b * block
        en = N if b == n_blocks - 1 else min(N, st + block)
        s = float(np.sum(mag[st:en]))
        p = s / total + eps
        H += -p * math.log2(p)
    return H / math.log2(n_blocks)


def spectral_contrast(mag: np.ndarray, n_bands: int = 6) -> float:
    N = mag.shape[0]
    band_size = max(1, N // n_bands)
    acc = 0.0
    for b in range(n_bands):
        st = b * band_size
        en = N if b == n_bands - 1 else min(N, st + band_size)
        band = mag[st:en]
        if band.size == 0:
            continue
        minv = float(np.min(band))
        maxv = float(np.max(band))
        mean = float(np.mean(band))
        if mean > 0:
            c = (maxv - minv) / mean
            acc += c
    return acc / n_bands


def f0_autocorr(frame: np.ndarray,
                sr: int,
                fmin: float = 60.0,
                fmax: float = 400.0) -> float:
    pre = np.empty_like(frame)
    pre[0] = frame[0]
    pre[1:] = frame[1:] - 0.97 * frame[:-1]

    N = pre.shape[0]
    tau_min = max(1, int(sr / fmax))
    tau_max = min(N - 1, int(sr / fmin))
    best_tau = -1
    best_val = -1.0
    for tau in range(tau_min, tau_max + 1):
        corr = float(np.dot(pre[: N - tau], pre[tau:]))
        if corr > best_val:
            best_val = corr
            best_tau = tau
    if best_tau <= 0:
        return 0.0
    f0 = sr / best_tau
    if fmin <= f0 <= fmax:
        return f0
    return 0.0


# ====================== RING BUFFER (PCEN history) ======================

class RingBuffer:
    def __init__(self, cap: int, dim: int):
        self.cap = cap
        self.dim = dim
        self.buf = np.zeros((cap, dim), dtype=np.float32)
        self.size = 0
        self.head = 0

    def push(self, vec: np.ndarray):
        idx = self.head % self.cap
        self.buf[idx, :] = vec
        self.head = (self.head + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            return (np.zeros(self.dim, dtype=np.float32),
                    np.zeros(self.dim, dtype=np.float32))
        data = self.buf[: self.size, :]
        return data.mean(axis=0), data.std(axis=0)


# ====================== CORE EXTRACTOR (mirror worklet) ======================

class FastFeatureExtractor:
    def __init__(self,
                 sr: int = 16000,
                 frame_sec: float = 0.5,
                 nfft: int = 1024,
                 n_mels: int = 64,
                 n_lfcc_bands: int = 40,
                 pcen_hist_len: int = 4):
        self.sr = sr
        self.frame_sec = frame_sec
        self.frame_len = int(sr * frame_sec)
        self.nfft = nfft
        self.win = hanning(nfft)
        self.mel_fb = mel_filterbank(nfft, sr, n_mels, 50.0, min(8000.0, sr / 2))
        self.lin_fb = linear_filterbank(nfft, sr, n_lfcc_bands, 50.0, sr / 2)
        self.pcen_state = pcen_create_state(n_mels, 0.98, 2.0, 0.5, 1e-6, 0.1)
        self.pcen_hist = RingBuffer(pcen_hist_len, n_mels)
        self.mfcc_hist = RingBuffer(pcen_hist_len, 13)

        self.smooth_rms = 0.0
        self.smooth_zcr = 0.0
        self.smooth_alpha = 0.2

        self.noise_ema = 1e-6
        self.noise_beta = 0.05

        self.last_mfcc = None
        self.last_delta = None
        self.last_entropy = 0.0
        self.last_flat = 0.0
        self.last_f0 = 0.0
        self.last_rms = 0.0

        # VAD giống worklet
        self.vad_rms_thresh = 0.005
        self.vad_snr_thresh = 3.0

    def _mag_from_frame(self, frame: np.ndarray) -> np.ndarray:
        buf = np.zeros(self.nfft, dtype=np.float32)
        L = min(frame.shape[0], self.nfft)
        buf[:L] = frame[:L] * self.win[:L]
        spec = np.fft.rfft(buf)
        mag = np.abs(spec).astype(np.float32)
        # rfft cho nfft//2 + 1; worklet dùng nfft//2 -> bỏ bin cuối
        return mag[: self.nfft // 2]

    def _snr_estimate(self, rms: float) -> float:
        p = rms * rms
        self.noise_ema = (1.0 - self.noise_beta) * self.noise_ema + self.noise_beta * max(1e-9, p)
        snr_lin = max(1e-6, p / max(1e-9, self.noise_ema))
        snr_db = 10.0 * math.log10(snr_lin)
        return max(0.0, min(40.0, snr_db))

    def extract_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Trả về vector 196 chiều cho 1 frame,
        hoặc None nếu frame bị coi là im lặng/noise (VAD).
        """
        frame = frame.astype(np.float32)

        mag = self._mag_from_frame(frame)

        # Mel + PCEN + history
        mel_pow = apply_fb(mag, self.mel_fb)
        pcen_vec = pcen_apply(mel_pow, self.pcen_state)
        self.pcen_hist.push(pcen_vec)
        pcen_mean, pcen_std = self.pcen_hist.mean_std()

        # LFCC
        lin_pow = apply_fb(mag, self.lin_fb)
        lin_log = np.log(lin_pow)
        lfcc20 = dct_ii(lin_log, 20)

        # MFCC 13 + delta + deltadelta
        mel_log = np.log(mel_pow)
        mfcc13 = dct_ii(mel_log, 13)

        if self.mfcc_hist.size > 0 and self.last_mfcc is not None:
            delta13 = mfcc13 - self.last_mfcc
        else:
            delta13 = np.zeros_like(mfcc13)

        if self.last_delta is not None:
            deltadelta13 = delta13 - self.last_delta
        else:
            deltadelta13 = np.zeros_like(delta13)

        self.mfcc_hist.push(mfcc13)
        self.last_mfcc = mfcc13.copy()
        self.last_delta = delta13.copy()

        # Spectral stats
        zcr_val = zcr(frame)
        flat = spectral_flatness(mag)
        roll = spectral_rolloff(mag, 0.85)
        ent = spectral_entropy(mag, 10)
        contr = spectral_contrast(mag, 6)
        self.last_entropy = ent
        self.last_flat = flat

        # Prosody
        f0 = f0_autocorr(frame, self.sr, 60.0, 400.0)
        rms = float(np.sqrt(np.mean(frame * frame)))

        jitter = min(5.0, abs((f0 - (self.last_f0 or f0)) /
                              max(60.0, f0 or 60.0))) * 100.0
        shimmer = min(5.0, abs((rms - (self.last_rms or rms)) /
                               max(1e-6, rms))) * 100.0
        self.last_f0 = f0
        self.last_rms = rms
        cpp = max(0.0, 20.0 - 10.0 * self.last_entropy - 5.0 * self.last_flat)

        # SNR + VAD
        snr_db = self._snr_estimate(rms)
        self.smooth_rms = self.smooth_alpha * rms + (1 - self.smooth_alpha) * self.smooth_rms
        self.smooth_zcr = self.smooth_alpha * zcr_val + (1 - self.smooth_alpha) * self.smooth_zcr

        is_silent_like = (self.smooth_rms < self.vad_rms_thresh) and (snr_db < self.vad_snr_thresh)
        if is_silent_like:
            return None  # bỏ frame im lặng / noise

        # Ghép thành vector 196 chiều
        mfcc39 = np.concatenate([mfcc13, delta13, deltadelta13], axis=0)  # 39
        spec_vec = np.array([zcr_val, flat, roll, ent, contr], dtype=np.float32)  # 5
        pros_vec = np.array([f0, jitter, shimmer, cpp], dtype=np.float32)  # 4

        full_vec = np.concatenate(
            [mfcc39, lfcc20, pcen_mean, pcen_std, spec_vec, pros_vec],
            axis=0,
        ).astype(np.float32)

        assert full_vec.shape[0] == 196, f"Expected 196 features, got {full_vec.shape[0]}"
        return full_vec

    def process_file(self, path: Path) -> np.ndarray:
        y, sr = load_audio(path, self.sr)
        if y.size < self.frame_len:
            return np.empty((0, 196), dtype=np.float32)

        feats: List[np.ndarray] = []
        for start in range(0, y.size - self.frame_len + 1, self.frame_len):
            frame = y[start: start + self.frame_len]
            vec = self.extract_frame(frame)
            if vec is not None:
                feats.append(vec)

        if not feats:
            return np.empty((0, 196), dtype=np.float32)
        return np.stack(feats, axis=0)


# ====================== IO & DATASET BUILD ======================

def load_audio(path: Path, sr_target: int) -> Tuple[np.ndarray, int]:
    """Đọc wav và resample về sr_target (mono)."""
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    if sr != sr_target:
        g = math.gcd(sr, sr_target)
        up = sr_target // g
        down = sr // g
        y = resample_poly(y, up, down).astype(np.float32)
        sr = sr_target
    return y, sr


def build_dataset(data_root: Path,
                  out_path: Path,
                  sr: int = 16000) -> None:
    extractor = FastFeatureExtractor(sr=sr)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for label_name, label in [("real", 0), ("fake", 1)]:
        class_dir = data_root / label_name
        if not class_dir.is_dir():
            print(f"[WARN] Không tìm thấy thư mục: {class_dir}")
            continue

        wav_files = sorted(class_dir.rglob("*.wav"))
        print(f"[INFO] {label_name}: tìm thấy {len(wav_files)} file WAV")

        for wav in wav_files:
            feats = extractor.process_file(wav)
            if feats.size == 0:
                continue
            X_list.append(feats)
            y_list.append(np.full(feats.shape[0], label, dtype=np.int32))

    if not X_list:
        raise RuntimeError("Không thu được frame nào — kiểm tra lại data_root / cấu trúc thư mục.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)
    print(f"[OK] Saved dataset: {out_path}  (X: {X.shape}, y: {y.shape})")


# ====================== MAIN ======================

def main():
    parser = argparse.ArgumentParser(description="Extract 196-dim fast features for DeepVoiceGuard.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="1_data",
        help="Thư mục chứa real/ và fake/ (mặc định: Data)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Models/fast_dataset.npz",
        help="Đường dẫn file output .npz (mặc định: Models/fast_dataset.npz)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate chuẩn để resample (gợi ý: 16000 hoặc 48000).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent  # backend_flask/
    data_root = (base_dir / args.data_root).resolve()
    out_path = (base_dir / args.out).resolve()

    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Output   : {out_path}")
    print(f"[INFO] SR       : {args.sr}")

    build_dataset(data_root, out_path, sr=args.sr)


if __name__ == "__main__":
    main()
