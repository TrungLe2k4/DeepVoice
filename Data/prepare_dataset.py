import os
import glob
import random
import librosa
import soundfile as sf
from pydub import AudioSegment, effects
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ========== Cấu hình ==========
input_root  = r"D:\DeepVoice\Data\Raw"
output_root = r"D:\DeepVoice\Data\Cleaned"
metadata_csv = r"D:\DeepVoice\Data\metadata_master.csv"
error_log    = r"D:\DeepVoice\Data\convert_audio_error_log.txt"

target_sr        = 16000
target_duration  = 5.0                 # độ dài chuẩn hoá cuối cùng (s)
target_length    = int(target_sr * target_duration)
labels           = ["real", "fake"]

# Chỉ yêu cầu file sau TRIM có độ dài >= 2s
min_duration     = 2.0                 # tối thiểu 2 giây

min_rms          = 0.01
trim_top_db      = 30
allowed_exts     = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}


# ========== Tiện ích ==========
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pick_set():
    r = random.random()
    for k, p in SPLIT.items():
        if r <= p:
            return k
        r -= p
    return "train"


def normalize_audiosegment(seg: AudioSegment) -> AudioSegment:
    seg = seg.set_channels(1)
    seg = effects.normalize(seg)
    return seg


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad hoặc cắt tín hiệu audio về đúng độ dài target_len mẫu.
    Thay thế cho librosa.util.fix_length để tránh lỗi version.
    """
    cur_len = len(y)
    if cur_len > target_len:
        return y[:target_len]
    if cur_len < target_len:
        pad_width = target_len - cur_len
        return np.pad(y, (0, pad_width), mode="constant")
    return y


# ========== Hàm xử lý từng file ==========
def process_one(args):
    ip, label, input_root, output_root = args
    try:
        audio = AudioSegment.from_file(ip)
        audio = audio.set_channels(1)

        # Độ dài gốc (chỉ dùng để log / bỏ cực kỳ ngắn)
        dur = audio.duration_seconds
        if dur < 0.5:
            return None, f"{ip} | Quá ngắn gốc ({dur:.2f}s)"

        audio = normalize_audiosegment(audio)

        # Lấy samples dạng float32 [-1, 1]
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        # Lọc theo RMS (âm lượng)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < min_rms:
            return None, f"{ip} | Âm lượng thấp (RMS={rms:.4f})"

        y = samples

        # Resample về target_sr
        if audio.frame_rate != target_sr:
            y = librosa.resample(y=y, orig_sr=audio.frame_rate, target_sr=target_sr)

        # Trim silence
        y_trim, _ = librosa.effects.trim(y, top_db=trim_top_db)
        trim_len_sec = len(y_trim) / target_sr

        # ❗ Chỉ loại file quá ngắn sau trim (< 2s)
        if trim_len_sec < min_duration:
            return None, f"{ip} | Quá ngắn sau trim (<2s: {trim_len_sec:.2f}s)"

        # Chuẩn hoá độ dài cuối cùng = 5s:
        # - Nếu > 5s: CẮT xuống 5s
        # - Nếu 2–5s: PAD thêm im lặng cho đủ 5s
        y_final = pad_or_trim(y_trim, target_length)

        # Chuẩn hoá biên độ [-1, 1]
        y_final = y_final / (np.max(np.abs(y_final)) + 1e-9)

        # ===== TÍNH ĐẶC TRƯNG ĐỂ LỌC TRÙNG =====
        # MFCC 13 chiều, lấy trung bình theo thời gian → vector (13,)
        mfcc = librosa.feature.mfcc(y=y_final, sr=target_sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        # Lượng tử hoá 4 chữ số thập phân để ổn định
        feat_key = "|".join(f"{v:.4f}" for v in mfcc_mean)

        # ===== LƯU FILE WAV =====
        base_name = os.path.splitext(os.path.basename(ip))[0]
        out_label_dir = os.path.join(output_root, label)
        ensure_dir(out_label_dir)
        op = os.path.join(out_label_dir, f"{base_name}.wav")
        sf.write(op, y_final, target_sr)

        return {
            "file_path": f"{label}/{base_name}.wav",
            "label": label,
            "duration": round(len(y_final) / target_sr, 3),
            "rms": round(float(np.sqrt(np.mean(y_final ** 2))), 5),
            "set": pick_set(),
            "feat_key": feat_key,
        }, None

    except Exception as e:
        return None, f"{ip} | {str(e)}"


# ========== Main ==========
def main():
    random.seed(42)
    ensure_dir(output_root)

    # Thu thập danh sách file đầu vào
    all_files = []
    for label in labels:
        in_label_dir = os.path.join(input_root, label)
        if not os.path.isdir(in_label_dir):
            print(f"⚠️ Không tìm thấy thư mục: {in_label_dir}")
            continue
        for ext in allowed_exts:
            all_files.extend(
                [(fp, label, input_root, output_root)
                 for fp in glob.glob(os.path.join(in_label_dir, f"*{ext}"))]
            )

    print(f"🔍 Tổng số file cần xử lý: {len(all_files)}")
    if not all_files:
        return

    n_jobs = min(cpu_count(), 8)
    print(f"⚙️ Đang sử dụng {n_jobs} CPU lõi song song...")

    rows, errors = [], []
    with Pool(processes=n_jobs) as pool:
        for result, err in tqdm(pool.imap_unordered(process_one, all_files), total=len(all_files)):
            if result:
                rows.append(result)
            if err:
                errors.append(err)

    # ===== Ghi metadata & loại bỏ file trùng đặc trưng =====
    if rows:
        df = pd.DataFrame(rows)

        # Nếu có cột feat_key → lọc trùng
        if "feat_key" in df.columns:
            # Giữ lại 1 bản ghi đầu tiên cho mỗi feat_key
            df_unique = df.drop_duplicates(subset=["feat_key"], keep="first").copy()

            # Xác định các file bị loại (trùng đặc trưng)
            dup_mask = ~df["file_path"].isin(df_unique["file_path"])
            dup_files = df.loc[dup_mask, "file_path"].tolist()

            removed_count = 0
            for relpath in dup_files:
                full_path = os.path.join(output_root, relpath)
                if os.path.isfile(full_path):
                    try:
                        os.remove(full_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"⚠️ Không xoá được file trùng {full_path}: {e}")

            if removed_count:
                print(f"🧹 Đã loại bỏ {removed_count} file có đặc trưng trùng nhau.")

            # Sử dụng df_unique cho metadata cuối
            df = df_unique

        # Ghi metadata CSV
        df.to_csv(metadata_csv, index=False, encoding="utf-8")
        print(f"✅ Đã lưu metadata: {metadata_csv} ({len(df)} file)")
        print(df.groupby(["label", "set"]).size())

    # Ghi log lỗi
    if errors:
        with open(error_log, "w", encoding="utf-8") as f:
            for msg in errors:
                f.write(msg + "\n")
        print(f"⚠️ Đã ghi log lỗi: {error_log} ({len(errors)} lỗi)")

    print("🎉 Hoàn tất chuẩn hoá dataset .")


if __name__ == "__main__":
    main()
