import os
from tqdm import tqdm

# ========== Cấu hình ==========
data_root = r"D:\DeepVoice\Data\Raw"   # thư mục chứa real/ và fake/
labels = ["real", "fake"]              # hai nhãn cần xử lý
allowed_exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

# ========== Đổi tên ==========

def rename_files():
    for label in labels:
        folder = os.path.join(data_root, label)
        if not os.path.isdir(folder):
            print(f"⚠️ Không tìm thấy thư mục: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(allowed_exts) and os.path.isfile(os.path.join(folder, f))]

        print(f"📁 {label}: tìm thấy {len(files)} file")

        for i, old_name in enumerate(tqdm(sorted(files), desc=f"Đang đổi tên {label}")):
            ext = os.path.splitext(old_name)[1].lower()
            new_name = f"{label}_{i+1:04d}{ext}"  # -> real_0001.wav, fake_0001.wav
            old_path = os.path.join(folder, old_name)
            new_path = os.path.join(folder, new_name)

            # Nếu tên mới đã tồn tại (rất hiếm), thêm hậu tố _dup
            if os.path.exists(new_path):
                base, ext2 = os.path.splitext(new_name)
                new_path = os.path.join(folder, f"{base}_dup{ext2}")

            os.rename(old_path, new_path)

        print(f"✅ Đã đổi tên toàn bộ file trong {label}.")

if __name__ == "__main__":
    print("🔄 Bắt đầu đổi tên tất cả file trong Raw/real và Raw/fake ...")
    rename_files()
    print("🎉 Hoàn tất đổi tên!")
