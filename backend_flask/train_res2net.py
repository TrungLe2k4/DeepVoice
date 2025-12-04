# backend_flask/train_res2net.py
import os
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

import pandas as pd

# ================== CẤU HÌNH ĐƯỜNG DẪN ==================
# Thư mục audio đã clean: chứa real/ và fake/
DATA_ROOT = r"D:\DeepVoice\Data\Cleaned"
METADATA_CSV = r"D:\DeepVoice\Data\metadata_master.csv"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

RES2NET_WEIGHTS = os.path.join(MODEL_DIR, "res2net_best.pt")
RES2NET_HISTORY = os.path.join(MODEL_DIR, "res2net_train_history.csv")
RES2NET_METRICS = os.path.join(MODEL_DIR, "res2net_metrics.json")

# ================== CẤU HÌNH AUDIO / SPEC ==================
SR = 16000
DURATION = 5.0  # giây (cắt / pad về 5s cho đồng bộ với fast model)
TARGET_LEN = int(SR * DURATION)

N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 50
FMAX = 8000

# ================== HYPERPARAMS ==================
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42


# ================== UTILS ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad hoặc cắt tín hiệu audio về đúng độ dài target_len mẫu.
    Thay cho librosa.util.fix_length để tránh lỗi version.
    """
    cur_len = len(y)
    if cur_len > target_len:
        return y[:target_len]
    if cur_len < target_len:
        pad_width = target_len - cur_len
        return np.pad(y, (0, pad_width), mode="constant")
    return y


def load_wav_fixed(path, sr=SR, duration=DURATION):
    """Đọc wav, resample, cắt/pad về độ dài cố định (duration giây)."""
    sig, orig_sr = sf.read(str(path), dtype="float32")

    # mixdown nếu stereo
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)

    if orig_sr != sr:
        sig = librosa.resample(sig, orig_sr=orig_sr, target_sr=sr)

    target_len = int(sr * duration)
    sig = pad_or_trim(sig, target_len)

    return sig


def wav_to_logmel(y, sr=SR):
    """Chuyển waveform -> log-mel spectrogram (N_MELS x T)."""
    M = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # dùng log10(1 + M) để tránh -inf
    logM = np.log10(1.0 + M).astype(np.float32)
    return logM


# ================== DATASET ==================
class DeepfakeSpecDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = list(paths)
        self.labels = list(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]

        wav = load_wav_fixed(path, sr=SR, duration=DURATION)
        spec = wav_to_logmel(wav, sr=SR)  # (F, T)

        # Chuẩn hóa theo mean/std trên mỗi mẫu (tùy chọn)
        m = np.mean(spec)
        s = np.std(spec) + 1e-6
        spec = (spec - m) / s

        spec = np.expand_dims(spec, axis=0)  # (1, F, T)
        spec = torch.from_numpy(spec)        # float32
        y = torch.tensor(float(y), dtype=torch.float32)

        return spec, y


def build_splits_from_metadata(clean_root: str, meta_csv: str):
    """
    Dùng metadata_master.csv để tách train / val / test giống hệt fast model.

    metadata_master.csv cần có cột:
      - file_path: "real/xxxx.wav" hoặc "fake/yyyy.wav"
      - label: "real" / "fake"
      - set: "train" / "val" / "test"
    """
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"Không tìm thấy metadata CSV: {meta_csv}")

    df = pd.read_csv(meta_csv)
    required_cols = {"file_path", "label", "set"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"metadata_master.csv phải chứa các cột: {required_cols}"
        )

    paths_tr, y_tr = [], []
    paths_val, y_val = [], []
    paths_te, y_te = [], []

    for _, row in df.iterrows():
        rel_path = str(row["file_path"])
        label_str = str(row["label"]).lower()
        set_str = str(row["set"]).lower()

        full_path = os.path.join(clean_root, rel_path)
        if not os.path.isfile(full_path):
            print(f"⚠️ Mất file audio, bỏ qua: {full_path}")
            continue

        yi = 1 if label_str == "fake" else 0

        if set_str == "train":
            paths_tr.append(full_path)
            y_tr.append(yi)
        elif set_str == "val":
            paths_val.append(full_path)
            y_val.append(yi)
        elif set_str == "test":
            paths_te.append(full_path)
            y_te.append(yi)
        else:
            # nếu set khác (ví dụ lỗi nhập), bỏ qua
            print(f"⚠️ set không hợp lệ '{set_str}' cho {full_path}, bỏ qua.")

    paths_tr = np.array(paths_tr)
    paths_val = np.array(paths_val)
    paths_te = np.array(paths_te)
    y_tr = np.array(y_tr, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_te = np.array(y_te, dtype=np.int64)

    return paths_tr, y_tr, paths_val, y_val, paths_te, y_te


# ================== RES2NET BLOCK ==================
class Res2Block(nn.Module):
    """
    Res2Net block đơn giản:
      - conv1x1 -> chia thành nhiều scale
      - mỗi scale có conv3x3 riêng, dùng skip nội bộ
      - conv1x1 out + shortcut
    """

    def __init__(self, in_ch, out_ch, scales=4, stride=(1, 1)):
        super().__init__()
        assert out_ch % scales == 0, "out_ch phải chia hết cho scales"
        self.scales = scales
        width = out_ch // scales

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        # Một conv3x3 cho mỗi scale
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    width, width, kernel_size=3, padding=1, bias=False
                )
                for _ in range(scales)
            ]
        )
        self.bns2 = nn.ModuleList(
            [nn.BatchNorm2d(width) for _ in range(scales)]
        )

        self.conv3 = nn.Conv2d(
            out_ch, out_ch, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != (1, 1) or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # chia channel thành scales nhánh
        splits = torch.chunk(out, self.scales, dim=1)
        out_scales = []
        prev = None
        for s in range(self.scales):
            z = splits[s]
            if s > 0 and prev is not None:
                z = z + prev
            z = self.convs[s](z)
            z = self.bns2[s](z)
            z = self.relu(z)
            prev = z
            out_scales.append(z)

        out = torch.cat(out_scales, dim=1)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


# ================== RES2NET CLASSIFIER ==================
class Res2NetClassifier(nn.Module):
    def __init__(self, num_classes=1, base_channels=32, scales=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                base_channels,
                kernel_size=3,
                stride=(1, 2),  # giảm T một nửa
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = Res2Block(
            base_channels,
            base_channels * 2,
            scales=scales,
            stride=(1, 2),
        )  # Cx2, T/2
        self.layer2 = Res2Block(
            base_channels * 2,
            base_channels * 4,
            scales=scales,
            stride=(1, 2),
        )  # Cx4, T/2
        self.layer3 = Res2Block(
            base_channels * 4,
            base_channels * 4,
            scales=scales,
            stride=(1, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)  # (B, C, 1, 1)
        x = torch.flatten(x, 1)  # (B, C)
        logits = self.fc(x).squeeze(-1)  # (B,)
        return logits


# ================== TRAIN / EVAL LOOP ==================
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_correct = 0
    n_samples = 0

    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)  # float (0/1)

        optimizer.zero_grad()
        logits = model(specs)  # (B,)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        n_samples += labels.size(0)

    avg_loss = running_loss / max(1, n_samples)
    avg_acc = running_correct / max(1, n_samples)

    print(
        f"[Train] Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"
    )
    return avg_loss, avg_acc


def eval_epoch(model, loader, device, epoch, split_name="Val"):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_correct = 0
    n_samples = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            n_samples += labels.size(0)

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss = running_loss / max(1, n_samples)
    avg_acc = running_correct / max(1, n_samples)

    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    if all_probs.size > 0 and all_labels.size > 0:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")

    print(
        f"[{split_name}] Epoch {epoch:02d} | Loss: {avg_loss:.4f} | "
        f"Acc: {avg_acc:.4f} | AUC: {auc:.4f}"
    )
    return avg_loss, avg_acc, auc, all_probs, all_labels


# ================== MAIN ==================
def main():
    set_seed(SEED)

    print("📂 Đang đọc splits từ metadata:", METADATA_CSV)
    paths_tr, y_tr, paths_val, y_val, paths_te, y_te = build_splits_from_metadata(
        DATA_ROOT, METADATA_CSV
    )

    print(
        f"✅ Số mẫu: train={len(paths_tr)}, val={len(paths_val)}, test={len(paths_te)}"
    )
    if len(paths_tr) == 0 or len(paths_val) == 0 or len(paths_te) == 0:
        print("⚠️ Thiếu một trong các split (train/val/test), kiểm tra lại metadata.")
    print(
        f"   Tỷ lệ fake (train): {float(y_tr.mean()):.4f} | "
        f"(val): {float(y_val.mean()):.4f} | (test): {float(y_te.mean()):.4f}"
    )

    train_ds = DeepfakeSpecDataset(paths_tr, y_tr)
    val_ds = DeepfakeSpecDataset(paths_val, y_val)
    test_ds = DeepfakeSpecDataset(paths_te, y_te)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥  Device:", device)

    model = Res2NetClassifier(
        num_classes=1, base_channels=32, scales=4
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    history = []
    best_val_auc = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc, val_auc, _, _ = eval_epoch(
            model, val_loader, device, epoch, split_name="Val"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_auc": float(val_auc),
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(best_state, RES2NET_WEIGHTS)
            print(
                f"💾 Lưu model tốt nhất (val AUC={val_auc:.4f}) tại {RES2NET_WEIGHTS}"
            )

    # Sau khi train xong: load best model để đánh giá trên test
    if best_state is None:
        print("❌ Không có best_state, có lỗi trong training?")
        return

    model.load_state_dict(best_state["model_state"])
    test_loss, test_acc, test_auc, test_probs, test_labels = eval_epoch(
        model, test_loader, device, epoch=0, split_name="Test"
    )

    # ================== LƯU HISTORY & METRICS ==================
    try:
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(RES2NET_HISTORY, index=False)
        print("📈 Đã lưu lịch sử train/val tại:", RES2NET_HISTORY)
    except Exception as e:
        print("⚠️ Lỗi lưu train history:", e)

    n_params = sum(p.numel() for p in model.parameters())

    metrics = {
        "best_val_auc": float(best_val_auc),
        "final_val_acc": float(history[-1]["val_acc"]),
        "final_val_auc": float(history[-1]["val_auc"]),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_auc": float(test_auc),
        "n_params": int(n_params),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
    }

    with open(RES2NET_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("📊 Đã lưu metric Res2Net tại:", RES2NET_METRICS)


if __name__ == "__main__":
    main()
