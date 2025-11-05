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
target_duration  = 5.0
target_length    = int(target_sr * target_duration)
labels           = ["real", "fake"]
min_duration     = 2.0
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

# ========== Hàm xử lý từng file ==========
def process_one(args):
    ip, label, input_root, output_root = args
    try:
        audio = AudioSegment.from_file(ip)
        audio = audio.set_channels(1)
        dur = audio.duration_seconds
        if dur < min_duration:
            return None, f"Quá ngắn ({dur:.2f}s)"

        audio = normalize_audiosegment(audio)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < min_rms:
            return None, f"Âm lượng thấp (RMS={rms:.4f})"

        y = samples
        if audio.frame_rate != target_sr:
            y = librosa.resample(y=y, orig_sr=audio.frame_rate, target_sr=target_sr)
        y_trim, _ = librosa.effects.trim(y, top_db=trim_top_db)
        if len(y_trim) < min_duration * target_sr:
            return None, f"Quá ngắn sau trim ({len(y_trim)/target_sr:.2f}s)"

        y_final = librosa.util.fix_length(y_trim[:target_length], size=target_length)
        y_final = y_final / (np.max(np.abs(y_final)) + 1e-9)

        # Tạo đường dẫn output
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
            "set": pick_set()
        }, None
    except Exception as e:
        return None, str(e)

# ========== Main ==========
def main():
    random.seed(42)
    ensure_dir(output_root)

    all_files = []
    for label in labels:
        in_label_dir = os.path.join(input_root, label)
        if not os.path.isdir(in_label_dir):
            print(f"⚠️ Không tìm thấy thư mục: {in_label_dir}")
            continue
        for ext in allowed_exts:
            all_files.extend([(fp, label, input_root, output_root)
                              for fp in glob.glob(os.path.join(in_label_dir, f"*{ext}"))])

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

    # Ghi metadata
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(metadata_csv, index=False, encoding="utf-8")
        print(f"✅ Đã lưu metadata: {metadata_csv} ({len(df)} file)")
        print(df.groupby(["label", "set"]).size())

    # Ghi log lỗi
    if errors:
        with open(error_log, "w", encoding="utf-8") as f:
            for msg in errors:
                f.write(msg + "\n")
        print(f"⚠️ Đã ghi log lỗi: {error_log} ({len(errors)} lỗi)")

    print("🎉 Hoàn tất chuẩn hoá dataset (song song & tối giản).")

if __name__ == "__main__":
    main()
