# backend_flask/res2net_predict_uploads.py
import os
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 🔁 TÁI DÙNG KIẾN TRÚC + HÀM XỬ LÝ AUDIO TỪ train_res2net.py
from train_res2net import (
    Res2NetClassifier,
    load_wav_fixed,
    wav_to_logmel,
    SR,
)

# ================== CẤU HÌNH ĐƯỜNG DẪN ==================
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "Models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

RES2NET_CKPT   = os.path.join(MODEL_DIR, "res2net_best.pt")
CSV_OUT_PATH   = os.path.join(MODEL_DIR, "res2net_uploads_predictions.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== DỰ ĐOÁN 1 FILE ==================
def predict_res2net(model: nn.Module, path_wav: str, threshold: float = 0.5):
    """
    - Load file wav
    - Chuẩn hóa độ dài giống train_res2net (load_wav_fixed)
    - Chuyển waveform -> log-mel (wav_to_logmel)
    - Chuẩn hóa mean/std trên spectrogram (giống DeepfakeSpecDataset)
    - Đưa vào Res2NetClassifier → prob FAKE
    """
    # 1) Load và chuẩn hóa length
    sig = load_wav_fixed(path_wav, sr=SR)          # giống train_res2net

    # 2) Waveform -> log-mel (F, T)
    spec = wav_to_logmel(sig, sr=SR)               # giống train_res2net

    # 3) Chuẩn hóa theo mean/std trên mỗi mẫu (y chang DeepfakeSpecDataset)
    m = np.mean(spec)
    s = np.std(spec) + 1e-6
    spec = (spec - m) / s

    # 4) Đưa vào model: (1,1,F,T)
    spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (B=1, C=1, F, T)
    spec_t = spec_t.to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(spec_t)          # (1,)
        prob = torch.sigmoid(logits)[0].item()

    label = "FAKE" if prob >= threshold else "REAL"
    return prob, label


# ================== MAIN: QUÉT uploads/ + LƯU CSV ==================
def main():
    print("✅ Đang load Res2Net checkpoint từ:", RES2NET_CKPT)
    if not os.path.isfile(RES2NET_CKPT):
        print("❌ Không tìm thấy checkpoint:", RES2NET_CKPT)
        return

    # Checkpoint do train_res2net.py lưu:
    #   {"epoch": ..., "model_state": ..., "optimizer_state": ...}
    ckpt = torch.load(RES2NET_CKPT, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        # fallback nếu sau này bạn save kiểu khác
        state_dict = ckpt

    # Tạo model giống hệt lúc train
    model = Res2NetClassifier(num_classes=1, base_channels=32, scales=4).to(DEVICE)

    # Nạp weight đúng kiến trúc
    print("🔎 Một vài key trong checkpoint:", list(state_dict.keys())[:8])
    model.load_state_dict(state_dict, strict=True)
    print("✅ Đã load state_dict vào Res2NetClassifier.")

    # Quét thư mục uploads
    root = Path(UPLOAD_DIR)
    print("📂 Đang quét thư mục:", root)
    if not root.exists():
        print("❌ Thư mục uploads không tồn tại!")
        return

    wavs = sorted(list(root.glob("*.wav")))
    if not wavs:
        print("⚠️ Không có file .wav nào trong uploads/")
        return

    rows = []
    n_real = n_fake = 0

    for p in wavs:
        path_str = str(p)
        prob, label = predict_res2net(model, path_str)

        if label == "FAKE":
            n_fake += 1
        else:
            n_real += 1

        print(f"\n🎧 File: {path_str}")
        print(f"   → Xác suất FAKE (Res2Net): {prob:.4f}")
        print(f"   → Dự đoán:                 {label}")

        rows.append({
            "filename": p.name,
            "path": path_str,
            "prob_res2net": f"{prob:.6f}",
            "label_res2net": label,
        })

    # Ghi CSV phục vụ báo cáo / so sánh với XGBoost
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CSV_OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "path", "prob_res2net", "label_res2net"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\n===== TỔNG KẾT RES2NET (uploads/) =====")
    print(f"  Tổng file: {len(wavs)}")
    print(f"  REAL: {n_real}")
    print(f"  FAKE: {n_fake}")
    print("\n📄 Đã lưu kết quả CSV tại:", CSV_OUT_PATH)


if __name__ == "__main__":
    main()
