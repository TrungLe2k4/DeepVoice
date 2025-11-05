# DeepVoice 

Dữ liệu giọng nói sử dụng trong hệ thống phát hiện tấn công DeepVoice chỉ gồm tiếng Việt.

Cấu trúc thư mục:
- raw/: dữ liệu âm thanh gốc
  - real/: giọng người thật (ghi âm tự nhiên)
  - fake/: giọng tổng hợp từ mô hình AI (VITS, Tacotron2, ElevenLabs,...)
- cleaned/: âm thanh đã xử lý (mono, 16kHz, ≤5 giây)
- augment/: dữ liệu tăng cường (noise, pitch, reverb), tách riêng real/fake
- metadata_master.csv: bảng thông tin toàn bộ dữ liệu tiếng Việt

Lưu ý:
- Tất cả file .wav nên là mono, 16kHz, PCM 16-bit.
- Thời lượng: 5 giây.
- Tỷ lệ real/fake cân bằng (khoảng 1:1).
