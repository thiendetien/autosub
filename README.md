# AutoSub Tool6 - Batch Video Subtitle Processing

Công cụ xử lý phụ đề video hàng loạt (batch) với giao diện PyQt5, hỗ trợ:

- **OCR Subtitle Extraction** – Trích xuất phụ đề từ video bằng PaddleOCR / RapidOCR
- **Speech-to-Text (STT)** – Tạo phụ đề từ giọng nói bằng WhisperX
- **Auto Translation** – Dịch phụ đề sang tiếng Việt qua Gemini API
- **TTS (Text-to-Speech)** – Lồng tiếng tự động (Edge TTS, Kokoro, Piper, ChumTTS, MiniMax)
- **Subtitle Removal (Inpainting)** – Xóa phụ đề cứng khỏi video bằng AI (STTN, LaMa, OpenCV)
- **Merge / Burn Subtitles** – Ghép phụ đề mềm hoặc đốt phụ đề cứng vào video bằng FFmpeg

## 📁 Cấu trúc thư mục

```
autosub2/
├── tool6_batch.py             # 🚀 File chính - GUI batch processing
├── tool6.py                   # OCRWorker, InpaintWorker (core workers)
├── vse_runner.py              # Single-thread PaddleOCR subtitle extractor
├── srt_to_video.py            # MergeWorker - ghép sub/audio vào video (FFmpeg)
├── subtitle_settings_dialog.py # Dialog tùy chỉnh style subtitle
├── download8movie_gui.py      # Module tải video
│
├── trans_api.py               # Dịch SRT qua API (Gemini)
├── trans_gemini.py            # Gemini translation backend
├── trans_local.py             # Local translation fallback
│
├── edge_tts_bridge.py         # Edge TTS engine bridge
├── kokoro_bridge.py           # Kokoro TTS engine bridge
├── chumtts_bridge.py          # ChumTTS engine bridge
├── minimax_bridge.py          # MiniMax TTS engine bridge
│
├── video_inpainting.py        # Video inpainting (xóa sub bằng AI)
├── sttn_inpainting.py         # STTN model inpainting
├── vsr_bridge.py              # Video Subtitle Remover bridge
│
├── core/                      # STTN core utilities
│   ├── __init__.py
│   ├── spectral_norm.py
│   └── utils.py
│
├── model/                     # AI model definitions & configs
│   ├── __init__.py
│   ├── sttn.py                # STTN InpaintGenerator model
│   ├── download_checks.json
│   ├── mdx_model_data.json
│   └── vr_model_data.json
│
├── fonts/                     # Font files cho subtitle rendering
│   ├── UTM-IMPACT.TTF
│   ├── NotoSansArabic-*.ttf
│   ├── NotoSansDevanagari-*.ttf
│   ├── NotoSansTamil-*.ttf
│   └── NotoNaskhArabic-*.ttf
│
├── checkpoints/               # ⚠️ Model weights (tải riêng, xem bên dưới)
│
├── UTM-IMPACT.TTF             # Font Impact (root copy)
├── impact.ttf                 # Font Impact (alias)
├── icon_autosub.png           # App icon
│
├── requirements.txt           # Danh sách pip packages
├── .gitignore
└── README.md
```

## ⚡ Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| **OS** | Ubuntu 20.04+ / Windows 10+ |
| **Python** | 3.10+ |
| **GPU** | NVIDIA GPU với CUDA 12.x (khuyến nghị) |
| **FFmpeg** | Cần cài sẵn (`sudo apt install ffmpeg`) |
| **RAM** | Tối thiểu 8GB, khuyến nghị 16GB+ |

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/thiendetien/autosub
cd autosub
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# hoặc: venv\Scripts\activate   # Windows
```

### 3. Cài đặt dependencies

```bash
# Cài PyTorch có CUDA trước (chọn đúng CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Cài PaddlePaddle GPU
pip install paddlepaddle-gpu

# Cài các package còn lại
pip install -r requirements.txt
```

### 4. Cài FFmpeg (nếu chưa có)

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Kiểm tra
ffmpeg -version
```

### 5. Tải model weights (nếu dùng STTN Inpainting)

Model STTN (~63MB) cần được tải về thư mục `checkpoints/`:

```bash
mkdir -p checkpoints
# Tải sttn.pth từ link bên dưới và đặt vào checkpoints/
# Link: https://drive.google.com/file/d/<FILE_ID>/view
```

> ⚠️ Nếu không dùng tính năng "Xóa sub bằng STTN", bạn có thể bỏ qua bước này.

## ▶️ Chạy

```bash
python tool6_batch.py
```

Giao diện PyQt5 sẽ mở ra, cho phép bạn:
1. Kéo thả video vào
2. Chọn chế độ xử lý (OCR, STT, Translate, TTS, Remove Sub, Merge...)
3. Bấm Start để chạy batch

## 🔑 API Keys (tùy chọn)

| Tính năng | API Key cần thiết |
|---|---|
| **Gemini Translation** | Google Gemini API Key (miễn phí) |
| **MiniMax TTS** | MiniMax API Key |
| **Edge TTS** | Không cần (miễn phí) |
| **Kokoro TTS** | Không cần (chạy local) |

Các API key được nhập trực tiếp trong giao diện GUI.

## 📋 Lưu ý

- **GPU**: Nhiều tính năng (OCR, Inpainting, TTS) chạy nhanh hơn nhiều lần khi có NVIDIA GPU với CUDA.
- **CPU-only**: Vẫn chạy được nhưng chậm hơn. Thay `paddlepaddle-gpu` bằng `paddlepaddle`, thay `onnxruntime-gpu` bằng `onnxruntime`.
- **FFmpeg**: Bắt buộc phải có để merge/burn subtitle vào video. Cần build với `--enable-libass` để hỗ trợ subtitle rendering.

## 📜 License

MIT License
