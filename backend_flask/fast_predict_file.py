# backend_flask/fast_predict_file.py
import os
import csv
import argparse
import numpy as np
import soundfile as sf
import librosa
import joblib

from train_fast_model import extract_vector, SR, MODEL_PATH, MODEL_DIR

BASE_DIR = os.path.dirname(__file__)
DEFAULT_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CSV_OUT_PATH = os.path.join(MODEL_DIR, "fast_uploads_predictions.csv")


def load_model():
    """Load pipeline (StandardScaler + XGBoost) chỉ 1 lần."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    print(f"✅ Đang load model từ: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)
    return pipe


def preprocess_audio(path_wav: str):
    """Đọc file âm thanh và chuẩn hoá giống lúc train (mono, SR=16000, 5s)."""
    sig, sr = sf.read(path_wav, dtype="float32")

    # Nếu file nhiều kênh (stereo) → lấy kênh đầu
    if sig.ndim > 1:
        sig = sig[:, 0]

    if sr != SR:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=SR)

    target_len = SR * 5  # 5 giây
    if len(sig) < target_len:
        sig = librosa.util.fix_length(sig, target_len)
    elif len(sig) > target_len:
        sig = sig[:target_len]

    return sig


def predict_one(pipe, path_wav: str):
    """Dự đoán 1 file .wav và in kết quả. Trả về (label, prob)."""
    sig = preprocess_audio(path_wav)

    # Trích vector đặc trưng giống train_fast_model
    vec = extract_vector(sig, SR).reshape(1, -1)  # (1, 196)

    # Dự đoán xác suất FAKE (class 1)
    prob = float(pipe.predict_proba(vec)[0, 1])
    label = "FAKE" if prob >= 0.5 else "REAL"

    print(f"\n🎧 File: {path_wav}")
    print(f"   → Xác suất FAKE: {prob:.4f}")
    print(f"   → Dự đoán:       {label}")

    return label, prob


def iter_audio_files(root_dir):
    """Duyệt tất cả file audio (wav/mp3/flac) trong 1 thư mục (đệ quy)."""
    exts = (".wav", ".mp3", ".flac")
    for r, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(exts):
                yield os.path.join(r, name)


def main():
    ap = argparse.ArgumentParser(
        description="Dự đoán DeepVoice (REAL/FAKE) cho 1 file hoặc cả thư mục uploads (XGBoost)."
    )
    ap.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_UPLOAD_DIR,
        help=(
            "Đường dẫn file audio hoặc thư mục chứa audio. "
            "Nếu bỏ trống sẽ dùng thư mục 'uploads' trong backend_flask."
        ),
    )
    args = ap.parse_args()
    target = args.path

    pipe = load_model()

    # ===== Trường hợp là thư mục: quét hết file + lưu CSV =====
    if os.path.isdir(target):
        print(f"📂 Đang quét thư mục: {target}")
        files = list(iter_audio_files(target))
        if not files:
            print("⚠️ Không tìm thấy file audio (.wav/.mp3/.flac) trong thư mục.")
            return

        n_real = n_fake = 0
        rows = []

        for f in sorted(files):
            label, prob = predict_one(pipe, f)
            if label == "FAKE":
                n_fake += 1
            else:
                n_real += 1

            rows.append({
                "filename": os.path.basename(f),
                "path": f,
                "prob_fast": f"{prob:.6f}",
                "label_fast": label,
            })

        total = n_real + n_fake
        print("\n===== TỔNG KẾT (XGBoost) =====")
        print(f"  Tổng file: {total}")
        print(f"  REAL: {n_real}")
        print(f"  FAKE: {n_fake}")

        # 💾 Lưu CSV phục vụ báo cáo
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(CSV_OUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "path", "prob_fast", "label_fast"]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print("\n📄 Đã lưu kết quả dự đoán XGBoost tại:", CSV_OUT_PATH)

    # ===== Trường hợp là 1 file lẻ =====
    else:
        if not os.path.isfile(target):
            print("❌ Không tìm thấy file hoặc thư mục:", target)
            return
        label, prob = predict_one(pipe, target)

        # Option: vẫn lưu CSV 1 dòng cho tiện chèn vào báo cáo nếu muốn
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(CSV_OUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "path", "prob_fast", "label_fast"]
            )
            writer.writeheader()
            writer.writerow({
                "filename": os.path.basename(target),
                "path": os.path.abspath(target),
                "prob_fast": f"{prob:.6f}",
                "label_fast": label,
            })
        print("\n📄 Đã lưu kết quả dự đoán XGBoost tại:", CSV_OUT_PATH)


if __name__ == "__main__":
    main()
