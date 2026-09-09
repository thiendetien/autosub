# srt_to_video.py
# -*- coding: utf-8 -*-
"""
SRT → TTS API → WAV Timeline (NO DELAY, NO DRIFT) → Sync → Burn Sub.
Module dùng lại trong tool5.py (KHÔNG GUI).
"""

import os
import requests
import subprocess
import tempfile
import time
from PyQt5 import QtCore
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
import unicodedata
import json

# ============================================================
# TTS PERFORMANCE CONFIG - Adjust this to test speed
# ============================================================
# Number of concurrent TTS API requests
# Default: 1 (safe, sequential)
# Try: 2 (faster, requires API MAX_CONCURRENT >= 2)
# Max: 4 (fastest, but may overload API)
MAX_TTS_WORKERS = 2  # <-- CHANGE THIS VALUE (1, 2, 3, or 4)
# ============================================================


def load_subtitle_settings():
    """Load settings from ~/.subtitle_settings.json"""
    settings_file = os.path.expanduser('~/.subtitle_settings.json')
    default_settings = {
        'font_name': 'UTM-IMPACT',
        'font_size': 24,
        'font_color': '#FFFFFF',
        'border_width': 2,
        'border_color': '#000000',
        'bg_enabled': True,
        'bg_color': '#000000',
        'bg_opacity': 0.7,
        'position_v': 60,
        'position_h': 'center',
        'bold': True,
        'italic': False,
        'shadow_enabled': True,
        'shadow_offset': 2,
        'shadow_color': '#000000',
    }
    
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                saved = json.load(f)
                default_settings.update(saved)
        except:
            pass
    return default_settings

def hex_to_ass_color(hex_color, opacity=1.0):
    """Convert #RRGGBB to &HAABBGGRR"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
    else:
        return "&H00FFFFFF"
        
    alpha = int((1.0 - opacity) * 255)
    return f"&H{alpha:02X}{b}{g}{r}"

def build_ass_style(video_height=None, overrides=None, target_lang=None):
    """Build FFmpeg force_style string from settings
    
    Args:
        video_height: Actual video height in pixels. If provided, position_v will be
                     scaled from real pixels to ASS coordinate system (PlayResY=288)
        overrides: Optional dict of per-video subtitle settings to override global settings
        target_lang: Optional target language name (e.g. 'Tiếng Ả Rập') to auto-select font
    """
    s = load_subtitle_settings()
    if overrides:
        s.update(overrides)  # Per-video settings override global
    
    styles = []
    
    # Font - auto-select based on target language for non-Latin scripts
    font_name = s['font_name']
    if target_lang:
        # Map language to appropriate font that supports the script
        lang_font_map = {
            "Tiếng Ả Rập": "Noto Naskh Arabic",
            "Tiếng Hindi": "Noto Sans Devanagari",
            "Tiếng Tamil": "Noto Sans Tamil",
            "Tiếng Trung": "Noto Sans CJK SC",
            "Tiếng Hàn": "Noto Sans CJK KR",
            "Tiếng Nhật": "Noto Sans CJK JP",
        }
        if target_lang in lang_font_map:
            font_name = lang_font_map[target_lang]
    
    styles.append(f"Fontname={font_name}")
    
    font_size_real = s.get('font_size', 24)  # Real pixels
    if video_height:
        font_size_ass = int(font_size_real * (PLAY_RES_Y / video_height))
    else:
        # Fallback: assume 1080p
        font_size_ass = int(font_size_real * (PLAY_RES_Y / 1080))
    
    styles.append(f"Fontsize={font_size_ass}")
    
    # Colors
    styles.append(f"PrimaryColour={hex_to_ass_color(s['font_color'])}")
    styles.append(f"OutlineColour={hex_to_ass_color(s['border_color'])}")
    
    # Background
    # BorderStyle=4: Outline + opaque box (supports alpha in BackColour better than BorderStyle=3)
    if s.get('bg_enabled'):
        styles.append(f"BorderStyle=4")
        styles.append(f"BackColour={hex_to_ass_color(s['bg_color'], s['bg_opacity'])}")
    else:
        styles.append(f"BorderStyle=1")
        styles.append(f"BackColour={hex_to_ass_color('#000000', 0)}")
        
    # Border/Outline width - Scale to ASS coordinates
    # NOTE: BorderStyle=4 handles background padding automatically, so only use border_width
    border_width_real = s.get('border_width', 2)
    
    if video_height:
        outline_ass = int(border_width_real * (PLAY_RES_Y / video_height))
    else:
        outline_ass = int(border_width_real * (PLAY_RES_Y / 1080))
    
    # Allow 0 for no border, but ensure at least some visibility if user wants border
    if border_width_real > 0 and outline_ass == 0:
        outline_ass = 1  # Minimum 1 if user wants border
    
    styles.append(f"Outline={outline_ass}")
    
    # Shadow - Scale offset to ASS coordinates (minimum 1 for visibility)
    if s.get('shadow_enabled') and not s.get('bg_enabled'):
        shadow_offset_real = s.get('shadow_offset', 2)
        if video_height:
            shadow_offset_ass = int(shadow_offset_real * (PLAY_RES_Y / video_height))
        else:
            shadow_offset_ass = int(shadow_offset_real * (PLAY_RES_Y / 1080))
        # Ensure minimum value of 1 for visibility
        shadow_offset_ass = max(1, shadow_offset_ass)
        styles.append(f"Shadow={shadow_offset_ass}")
    else:
        styles.append(f"Shadow=0")

    # Alignment
    pos_map = {'left': 1, 'center': 2, 'right': 3}
    align = pos_map.get(s.get('position_h', 'center'), 2)
    styles.append(f"Alignment={align}")
    
    # MarginV (Bottom margin) - Scale from real pixels to ASS coordinates
    position_v_real = s.get('position_v', 60)  # Real pixels from bottom
    
    if video_height:
        # Convert real pixels to ASS coordinate system
        # ASS uses PlayResY=288, so we need to scale:
        # margin_ass = position_real * (PLAY_RES_Y / video_height)
        margin_v = int(position_v_real * (PLAY_RES_Y / video_height))
    else:
        # Fallback: assume 1080p video
        margin_v = int(position_v_real * (PLAY_RES_Y / 1080))
    
    styles.append(f"MarginV={margin_v}")
    
    # Bold/Italic
    styles.append(f"Bold={-1 if s.get('bold', True) else 0}")
    styles.append(f"Italic={-1 if s.get('italic', False) else 0}")
    
    # MaxLines - Auto wrap long subtitles to 2 lines
    styles.append(f"MaxLines=2")
    
    return ",".join(styles)


def clean_text_for_tts(text: str) -> str:
    """
    Làm sạch text cho TTS:
    - Giữ: chữ cái tiếng Việt, số, dấu cách, dấu chấm (.), dấu phẩy (,)
    - Bỏ: ngoặc đơn (), ngoặc vuông [], HTML tags, và các ký tự đặc biệt khác
    - Thay các ký tự đặc biệt còn lại thành dấu chấm hoặc phẩy
    """
    text = unicodedata.normalize("NFC", text)

    # Bỏ <i>...</i>, <b>...</b> HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Bỏ ngoặc vuông [Music], [SFX], [xxx]
    text = re.sub(r"\[[^\]]*\]", " ", text)
    
    # Bỏ dấu ngoặc đơn () nhưng GIỮ LẠI nội dung bên trong
    text = text.replace("(", " ").replace(")", " ")
    
    # Thay các ký tự đặc biệt thành dấu câu cơ bản
    # Giữ lại: chữ cái (a-zA-Z + tiếng Việt), số (0-9), dấu cách, dấu chấm (.), dấu phẩy (,)
    # Thay các ký tự khác như: ?, !, :, ;, -, —, ..., v.v. thành dấu chấm
    special_chars = {
        '?': '.',
        '!': '.',
        ':': ',',
        ';': ',',
        '—': ',',
        '–': ',',
        '-': ' ',
        '…': '.',
        '"': '',
        '"': '',
        '"': '',
        ''': '',
        ''': '',
        '\'': '',
    }
    
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)

    # Gộp nhiều dòng thành một
    text = text.replace("\r", " ").replace("\n", " ")

    # Gộp nhiều khoảng trắng
    text = re.sub(r"\s+", " ", text)
    
    # Gộp nhiều dấu chấm/phẩy liên tiếp
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)

    return text.strip()


# ===================================
# CONFIG
# ===================================

TTS_API_URL = "http://192.168.1.87:8002/api2/convert-tts"  # sửa IP tuỳ máy của mày

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cross-platform FFmpeg detection (Windows + Linux/Ubuntu)
import sys
import shutil
if sys.platform == "win32":
    FFMPEG = os.path.join(BASE_DIR, "ffmpeg.exe")
    FFPROBE = os.path.join(BASE_DIR, "ffprobe.exe")
else:
    # Linux/Ubuntu: PRIORITIZE system ffmpeg (/usr/bin/ffmpeg) which has libass!
    # Conda ffmpeg lacks subtitles filter, so system ffmpeg is required
    if os.path.exists("/usr/bin/ffmpeg"):
        FFMPEG = "/usr/bin/ffmpeg"
        FFPROBE = "/usr/bin/ffprobe"
        print(f"✅ Using system FFmpeg with libass: {FFMPEG}")
    else:
        FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
        FFPROBE = shutil.which("ffprobe") or "ffprobe"
        print(f"⚠️ Using fallback FFmpeg: {FFMPEG}")

# Verify FFmpeg exists (Windows only)
if sys.platform == "win32" and not os.path.isfile(FFMPEG):
    raise FileNotFoundError("Không tìm thấy ffmpeg.exe trong thư mục!")

PLAY_RES_Y = 288

DEBUG_AUDIO_DIR = os.path.join(BASE_DIR, "_debug_tts_audio")
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)

TAIL_TEXT = "Đây là một câu ngắn cần chính xác."
SHORT_WORDS_THRESHOLD = 5
TAIL_CUT_BUFFER_SEC = 0.01

# Font file - allow missing on Linux (will use system fonts)
IMPACT_TTF = os.path.join(BASE_DIR, "UTM-IMPACT.TTF")
if sys.platform == "win32" and not os.path.isfile(IMPACT_TTF):
    raise FileNotFoundError("Không tìm thấy impact.ttf cùng cấp với file python!")




# ===================================
# UTILS
# ===================================

def ffprobe_duration(path):
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


_TAIL_DUR_CACHE = {}  # key = round(speed,3) -> duration_sec

def tts_api_get_mp3(text: str, speed: float, out_mp3_path: str, api_url: str = None, retries=3):
    url = api_url or TTS_API_URL
    
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                params={"input_text": text, "speed": float(speed)},
                timeout=60
            )
            resp.raise_for_status()
            download_url = resp.json()["download_link"]
            data = requests.get(download_url, timeout=60).content
            
            # Check for minimum file size to ensure it's not a corrupted/empty MP3
            if len(data) < 500:
                print(f"⚠️ Warning: Downloaded MP3 is too small ({len(data)} bytes). Retry {attempt + 1}/{retries}...")
                if os.path.exists(out_mp3_path): os.remove(out_mp3_path)
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                else:
                    # All retries exhausted, file still too small
                    raise Exception(f"TTS API returned file too small ({len(data)} bytes) after {retries} attempts. Text: '{text[:50]}'")

            with open(out_mp3_path, "wb") as f:
                f.write(data)
            
            # Success
            return
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1}/{retries} failed: {e}")
            if os.path.exists(out_mp3_path): os.remove(out_mp3_path)
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e


def mp3_to_wav(mp3_path: str, wav_path: str):
    subprocess.run([
        FFMPEG, "-y",
        "-i", mp3_path,
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "44100",
        wav_path
    ], check=True)

def get_tail_duration_sec(speed: float, tmp_dir: str, api_url: str = None) -> float:
    key = round(float(speed), 3)
    if key in _TAIL_DUR_CACHE:
        return _TAIL_DUR_CACHE[key]

    mp3_path = os.path.join(tmp_dir, f"__tail_speed_{key}.mp3")
    wav_path = os.path.join(tmp_dir, f"__tail_speed_{key}.wav")

    url_to_check = api_url or TTS_API_URL
    if url_to_check.strip().lower() == "native" or url_to_check.strip().lower() in ["piper", "chumtts"]:
        from chumtts_bridge import chumtts_generate_wav
        chumtts_generate_wav(TAIL_TEXT, speed, wav_path)
        # Resample to 44100Hz to match standard timeline
        seg = AudioSegment.from_wav(wav_path)
        seg.set_frame_rate(44100).set_channels(1).export(wav_path, format="wav")
    elif url_to_check.strip().lower().startswith("kokoro"):
        from kokoro_bridge import kokoro_generate_wav
        v_name = url_to_check.split(":", 1)[1].strip() if ":" in url_to_check else "af_bella"
        kokoro_generate_wav(TAIL_TEXT, speed, wav_path, voice=v_name)
        seg = AudioSegment.from_wav(wav_path)
        seg.set_frame_rate(44100).set_channels(1).export(wav_path, format="wav")
    elif url_to_check.strip().lower().startswith("minimax"):
        from minimax_bridge import minimax_generate_wav
        v_name = url_to_check.split(":", 1)[1].strip() if ":" in url_to_check else "Vietnamese_Serene_Man"
        minimax_generate_wav(TAIL_TEXT, speed, wav_path, voice_id=v_name, emotion="neutral")
        seg = AudioSegment.from_wav(wav_path)
        seg.set_frame_rate(44100).set_channels(1).export(wav_path, format="wav")
    elif url_to_check.strip().lower().startswith("edge"):
        from edge_tts_bridge import edge_tts_generate_wav
        v_name = url_to_check.split(":", 1)[1].strip() if ":" in url_to_check else "en-US-AriaNeural"
        edge_tts_generate_wav(TAIL_TEXT, speed, wav_path, voice=v_name)
        seg = AudioSegment.from_wav(wav_path)
        seg.set_frame_rate(44100).set_channels(1).export(wav_path, format="wav")
    else:
        tts_api_get_mp3(TAIL_TEXT, speed, mp3_path, api_url=url_to_check)
        mp3_to_wav(mp3_path, wav_path)
        
    dur = ffprobe_duration(wav_path)

    _TAIL_DUR_CACHE[key] = dur
    return dur





def srt_time_to_seconds(t):
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    blocks, cur = [], []
    for line in lines:
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)

    segs = []
    for block in blocks:
        if len(block) < 2:
            continue

        idx = None
        p = 0
        if block[0].isdigit():
            idx = int(block[0])
            p = 1

        if "-->" not in block[p]:
            continue

        t1, t2 = [x.strip() for x in block[p].split("-->")]

        segs.append({
            "index": idx,
            "start": srt_time_to_seconds(t1),
            "end": srt_time_to_seconds(t2),
            "text": " ".join(x.strip() for x in block[p + 1:] if x.strip())
        })

    return segs

def ffprobe_resolution(path):
    """
    Trả về (width, height) của video bằng ffprobe.
    """
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    w_str, h_str = result.stdout.strip().split("x")
    return int(w_str), int(h_str)




# ===================================
# WORKER
# ===================================
class MergeWorker(QtCore.QThread):
    """
    QThread worker: nhận (video, srt, out, speed, margin_v) rồi:
    - Gọi TTS API
    - Làm timeline WAV
    - Sync với video gốc
    - Burn subtitle lên (BOTTOM-CENTER, dùng ROI nếu có)
    """
    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, video, srt, out,
                 speed=1.2,
                 orig_video_speed=1.0,
                 uvr_enabled=False,
                 uvr_strength=100,
                 vol_bgm=2.0,
                 vol_tts=2.0,
                 margin_v=60,
                 roi_top=None,
                 roi_bottom=None,
                 tts_api_url=None,
                 kokoro_voice="af_bella",
                 sync_mode="stretch_video",
                 sub_overrides=None,
                 burn_sub_only=False,
                 bgm_files=None,
                 bgm_random=True,
                 vol_bgm_import=1.0,
                 speaker_mapping=None,
                 speaker_json_path=None,
                 line_mapping=None,
                 minimax_voice="Vietnamese_Serene_Man",
                 minimax_model="speech-2.8-hd",
                 line_emotion=None,
                 target_lang=None,
                 parent=None):
        super().__init__(parent)
        self.speaker_mapping = speaker_mapping or {}
        self.line_mapping = line_mapping or {}
        self.minimax_voice = minimax_voice or "Vietnamese_Serene_Man"
        self.minimax_model = minimax_model or "speech-2.8-hd"
        self.line_emotion = line_emotion or {}
        self.speaker_json_path = speaker_json_path
        self.target_lang = target_lang
        
        # Load speaker JSON if provided
        self.speaker_idx_map = {}
        if self.speaker_json_path and os.path.exists(self.speaker_json_path):
            try:
                import json
                with open(self.speaker_json_path, 'r', encoding='utf-8') as f:
                    speaker_data = json.load(f)
                for item in speaker_data:
                    idx = int(item.get("index", -1))
                    spk_id = item.get("speaker_id", "")
                    if idx > 0 and spk_id:
                        self.speaker_idx_map[idx] = spk_id
            except Exception as e:
                print(f"Error loading speaker JSON: {e}")
                
        self.burn_sub_only = burn_sub_only
        self.sync_mode = sync_mode
        self.video = video
        self.srt = srt
        self.out = out
        self.sub_overrides = sub_overrides  # Per-video subtitle settings
        self.speed = float(speed)
        self.orig_video_speed = float(orig_video_speed)
        self.uvr_enabled = uvr_enabled
        self.uvr_strength = max(0, min(100, int(uvr_strength)))
        self.vol_bgm = float(vol_bgm)
        self.vol_tts = float(vol_tts)
        self.margin_v = int(margin_v)

        # ROI theo pixel trên video gốc (y từ trên xuống)
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        
        # BGM files list
        self.bgm_files = bgm_files or []
        self.bgm_random = bgm_random
        self.vol_bgm_import = float(vol_bgm_import)
        
        # TTS API
        self.tts_api_url = tts_api_url or "http://127.0.0.1:8002/api2/convert-tts"
        self.kokoro_voice = kokoro_voice or "af_bella"
        # Khởi tạo file log tĩnh để chẩn đoán
        self.debug_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_debug.log")
        with open(self.debug_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- RUN SCRIPT AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Vocal Remove: {uvr_enabled} at {uvr_strength}%\n")
            f.write(f"Volume BGM: {self.vol_bgm}\n")
            f.flush()

    def emit_log(self, msg):
        self.log.emit(msg)
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} | {msg}\n")
                f.flush()
        except:
            pass

    def run(self):
        try:
            if self.burn_sub_only:
                self.emit_log("🔗 Bắt đầu ráp Video & Subtitle (Không chạy TTS)...")
                _, video_h = ffprobe_resolution(self.video)
                sub_style = build_ass_style(video_height=video_h, overrides=self.sub_overrides, target_lang=self.target_lang)
                font_dir = os.path.join(BASE_DIR, "fonts").replace("\\", "/")
                os.makedirs(font_dir, exist_ok=True)
                impact_font_src = os.path.join(BASE_DIR, "UTM-IMPACT.TTF")
                if os.path.exists(impact_font_src):
                    import shutil
                    shutil.copy2(impact_font_src, os.path.join(font_dir, "UTM-IMPACT.TTF"))
                
                safe_srt = self.srt.replace("\\", "/").replace(":", "\\:")
                sub_filter = f"[0:v]subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'[vout]"
                
                cmd = [FFMPEG, "-y", "-i", self.video,
                    "-filter_complex", sub_filter,
                    "-map", "[vout]", "-map", "0:a?",
                    "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "22", "-r", "30",
                    "-c:a", "copy",
                    self.out
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError:
                    if "-hwaccel" in cmd:
                        hw_idx = cmd.index("-hwaccel")
                        del cmd[hw_idx:hw_idx+2]
                    v_idx = cmd.index("h264_nvenc")
                    cmd[v_idx] = "libx264"
                    cmd[cmd.index("-preset") + 1] = "superfast"
                    q_idx = cmd.index("-qp")
                    cmd[q_idx] = "-crf"
                    subprocess.run(cmd, check=True)
                
                self.emit_log("🎉 DONE! Burn Subtitle thành công.")
                self.finished.emit(self.out)
                return

            self.emit_log(f"🎬 Khởi chạy MergeWorker cho: {os.path.basename(self.video)}")
            self._run_impl()
            self.finished.emit(self.out)
        except Exception as e:
            self.error.emit(str(e))

    def _run_impl(self):
        if self.sync_mode == "stretch_video":
            self._run_impl_chunking()
        else:
            self._run_impl_monolithic()

    def _run_impl_chunking(self):
        segs = parse_srt(self.srt)
        self.emit_log(f"📑 Có {len(segs)} câu SRT")

        # Use local temp dir instead of /tmp to avoid sudo issues
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_base = os.path.join(script_dir, '.temp_wav_sync')
        os.makedirs(temp_base, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="wav_sync_", dir=temp_base)
        self.emit_log(f"📂 TEMP WAV FOLDER: {tmp}")
        self.emit_log(f"📂 DEBUG WAV FOLDER: {DEBUG_AUDIO_DIR}")

        try:
            tail_dur_cached = get_tail_duration_sec(self.speed, tmp, api_url=self.tts_api_url)
            self.emit_log(f"🧷 Tail duration cached @speed={self.speed}: {tail_dur_cached:.3f}s")
        except Exception as e:
            tail_dur_cached = None
            self.emit_log(f"⚠️ Không đo được tail duration (sẽ bỏ short-fix): {e}")

        # UVR Extraction Block
        audio_stream_idx = "0:a"
        bg_audio_file = self.video
        bg_volume = self.vol_bgm
        self.is_random_bgm = False
        
        import random
        if self.bgm_files:
            valid_bgm = [f for f in self.bgm_files if os.path.exists(f)]
            if valid_bgm:
                if self.bgm_random:
                    bg_audio_file = random.choice(valid_bgm)
                else:
                    bg_audio_file = valid_bgm[0]
                self.emit_log(f"🎵 BGM: {os.path.basename(bg_audio_file)}")
                self.uvr_enabled = False
                self.is_random_bgm = True
                bg_volume = self.vol_bgm_import
        
        if self.uvr_enabled:
            self.emit_log("🎙 Bật Khử Giọng UVR5. Trích xuất âm thanh gốc để tách...")
            orig_wav_path = os.path.join(tmp, "orig_extract.wav")
            subprocess.run([FFMPEG, "-y", "-i", self.video, "-q:a", "0", "-map", "a", orig_wav_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(orig_wav_path):
                self.emit_log("🎙 audio-separator đang lột giọng bằng MDX-Net... (Vui lòng đợi 1-3 phút)")
                uvr_model_dir = os.path.join(BASE_DIR, "model")
                os.makedirs(uvr_model_dir, exist_ok=True)
                try:
                    import sys
                    audio_sep_bin = os.path.join(os.path.dirname(sys.executable), "audio-separator")
                    cmd_uvr = [
                        audio_sep_bin, orig_wav_path,
                        "--model_filename", "UVR-MDX-NET-Inst_HQ_3.onnx",
                        "--output_dir", tmp,
                        "--output_format", "wav",
                        "--model_file_dir", uvr_model_dir
                    ]
                    
                    
                    # Bỏ DEVNULL để cho hiển thị tiến trình Separation ra màn hình Terminal
                    uvr_process = subprocess.run(cmd_uvr, check=True, capture_output=True, text=True)
                    self.emit_log("✅ Tách giọng UVR xong! Output log:")
                    self.emit_log(uvr_process.stdout)
                    
                    import glob
                    instr_files = glob.glob(os.path.join(tmp, "*_(Instrumental)_*.wav"))
                    vocal_files = glob.glob(os.path.join(tmp, "*_(Vocals)_*.wav"))
                    if instr_files:
                        import math
                        raw_ratio = (100 - self.uvr_strength) / 100.0
                        vocal_volume = math.sqrt(raw_ratio) if raw_ratio > 0 else 0.0
                        if vocal_volume > 0.01 and vocal_files:
                            mixed_bg_path = os.path.join(tmp, "mixed_bg.wav")
                            mix_cmd = [
                                FFMPEG, "-y",
                                "-i", instr_files[0], "-i", vocal_files[0],
                                "-filter_complex",
                                f"[0:a]volume=1.0[inst];[1:a]volume={vocal_volume:.2f}[voc];[inst][voc]amix=inputs=2:dropout_transition=0:normalize=0[out]",
                                "-map", "[out]", mixed_bg_path
                            ]
                            subprocess.run(mix_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if os.path.exists(mixed_bg_path):
                                bg_audio_file = mixed_bg_path
                                self.emit_log(f"✅ Khử giọng {self.uvr_strength}%! Giữ lại {100-self.uvr_strength}% giọng gốc.")
                            else:
                                bg_audio_file = instr_files[0]
                                self.emit_log("✅ Khử giọng 100% (mix lỗi, fallback).")
                        else:
                            bg_audio_file = instr_files[0]
                            self.emit_log("✅ Khử giọng 100%! BGM sạch tinh.")
                    else:
                        self.emit_log("⚠️ Không tìm thấy file Instrumental sau khi tách.")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    self.emit_log(f"⚠️ UVR5 lỗi: {e}\nTraceback:\n{tb}\nDùng lại âm thanh nền kèm giọng gốc.")
            else:
                self.emit_log("⚠️ Không xuất được âm thanh từ video gốc (Video bị câm?).")

        # Create chunks
        self.emit_log(f"🔪 Đang cắt Frame và Lồng Tiếng (Phương án Chunking Delay)... Có thể mất một lát...")
        chunk_files = []
        time_cursor = 0.0
        
        _, video_h = ffprobe_resolution(self.video)
        sub_style = build_ass_style(video_height=video_h, overrides=self.sub_overrides, target_lang=self.target_lang)
        font_dir = os.path.join(BASE_DIR, "fonts").replace("\\", "/")
        os.makedirs(font_dir, exist_ok=True)
        
        # Sort segments to avoid overlap
        segs = sorted(segs, key=lambda x: x["start"])

        # ═══════════════════════════════════════════════
        # PHA 1: INFER TTS TẤT CẢ SEGMENTS (tuần tự, nhanh)
        # ═══════════════════════════════════════════════
        self.emit_log(f"🎙 Pha 1: Đang infer TTS cho {len(segs)} segments...")
        tts_results = []  # [(i, seg, tts_wav_path, audio_seg, stretch_factor, srt_dur, text)]
        
        for i, seg in enumerate(segs):
            raw_text = seg["text"]
            text = clean_text_for_tts(raw_text)
            srt_start = seg["start"]
            srt_end = seg["end"]
            srt_dur = srt_end - srt_start

            if not text or not text.strip():
                tts_results.append((i, seg, None, None, 1.0, srt_dur, text))
                continue
            
            tts_wav_path = os.path.join(tmp, f"tts_{i:04}.wav")
            mp3_path = os.path.join(tmp, f"tts_{i:04}.mp3")
            
            words = [w for w in text.split() if w.strip()]
            is_short = (len(words) < SHORT_WORDS_THRESHOLD) and (tail_dur_cached is not None)
            text_for_tts = text
            tail_dur = 0.0

            if is_short and not self.tts_api_url.strip().lower().startswith(("native", "piper", "chumtts", "kokoro", "minimax", "edge")):
                # Đổi dấu chấm/chấm hỏi/chấm than thành dấu phẩy để ép engine API đọc liền tù tì không ngắt câu, tránh rớt đuôi.
                text_clean = text.replace('.', ',').replace('?', ',').replace('!', ',')
                text_for_tts = f"{text_clean}, {TAIL_TEXT}"
                tail_dur = float(tail_dur_cached)
            
            try:
                if self.tts_api_url.strip().lower() in ["native", "piper", "chumtts"]:
                    from chumtts_bridge import chumtts_generate_wav
                    chumtts_generate_wav(text_for_tts, 1.0, tts_wav_path)
                    audio_seg = AudioSegment.from_wav(tts_wav_path).set_frame_rate(44100).set_channels(1)
                    audio_seg.export(tts_wav_path, format="wav")
                elif self.tts_api_url.strip().lower().startswith("kokoro"):
                    # Determine voice
                    v_name = self.tts_api_url.split(":", 1)[1].strip() if ":" in self.tts_api_url else getattr(self, "kokoro_voice", "af_bella")
                    srt_index = i + 1
                    if hasattr(self, 'line_mapping') and srt_index in self.line_mapping:
                        mapped = self.line_mapping[srt_index]
                        if not mapped.startswith("mm:"):
                            v_name = mapped
                    elif hasattr(self, 'speaker_idx_map') and srt_index in self.speaker_idx_map:
                        spk_id = self.speaker_idx_map[srt_index]
                        if spk_id in self.speaker_mapping:
                            v_name = self.speaker_mapping[spk_id]
                    
                    from kokoro_bridge import kokoro_generate_wav
                    kokoro_generate_wav(text_for_tts, 1.0, tts_wav_path, voice=v_name)
                    audio_seg = AudioSegment.from_wav(tts_wav_path).set_frame_rate(44100).set_channels(1)
                    audio_seg.export(tts_wav_path, format="wav")
                elif self.tts_api_url.strip().lower().startswith("minimax"):
                    # Determine MiniMax voice
                    mm_voice = getattr(self, "minimax_voice", "Vietnamese_Serene_Man")
                    mm_model = getattr(self, "minimax_model", "speech-2.8-hd")
                    mm_emotion = self.line_emotion.get(i + 1, "neutral") if hasattr(self, 'line_emotion') else "neutral"
                    srt_index = i + 1
                    if hasattr(self, 'line_mapping') and srt_index in self.line_mapping:
                        mapped = self.line_mapping[srt_index]
                        if mapped.startswith("mm:"):
                            mm_voice = mapped[3:]  # strip "mm:" prefix
                    elif hasattr(self, 'speaker_idx_map') and srt_index in self.speaker_idx_map:
                        spk_id = self.speaker_idx_map[srt_index]
                        if spk_id in self.speaker_mapping:
                            mapped = self.speaker_mapping[spk_id]
                            if mapped.startswith("mm:"):
                                mm_voice = mapped[3:]
                    
                    from minimax_bridge import minimax_generate_wav
                    minimax_generate_wav(text_for_tts, self.speed, tts_wav_path, voice_id=mm_voice, emotion=mm_emotion, model=mm_model)
                    audio_seg = AudioSegment.from_wav(tts_wav_path)
                elif self.tts_api_url.strip().lower().startswith("edge"):
                    from edge_tts_bridge import edge_tts_generate_wav, get_default_voice_for_lang
                    edge_voice = self.tts_api_url.split(":", 1)[1].strip() if ":" in self.tts_api_url else None
                    if not edge_voice and hasattr(self, 'target_lang') and self.target_lang:
                        edge_voice = get_default_voice_for_lang(self.target_lang)
                    if not edge_voice:
                        edge_voice = "en-US-AriaNeural"
                    edge_tts_generate_wav(text_for_tts, self.speed, tts_wav_path, voice=edge_voice)
                    audio_seg = AudioSegment.from_wav(tts_wav_path)
                else:
                    tts_api_get_mp3(text_for_tts, 1.0, mp3_path, api_url=self.tts_api_url)
                    mp3_to_wav(mp3_path, tts_wav_path)
                    audio_seg = AudioSegment.from_wav(tts_wav_path)
                    
                if self.speed != 1.0:
                    atempo_wav = tts_wav_path.replace(".wav", "_atempo.wav")
                    subprocess.run([FFMPEG, "-y", "-i", tts_wav_path, "-filter:a", f"atempo={self.speed}", atempo_wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    import shutil
                    shutil.move(atempo_wav, tts_wav_path)
                    audio_seg = AudioSegment.from_wav(tts_wav_path)
            except Exception as tts_error:
                self.emit_log(f"⚠️ TTS failed for segment #{i+1}: {tts_error}")
                audio_seg = AudioSegment.silent(duration=srt_dur*1000)
                audio_seg.export(tts_wav_path, format="wav")

            if is_short and tail_dur > 0:
                cut_ms = int(max(0.0, (tail_dur + TAIL_CUT_BUFFER_SEC)) * 1000.0)
                keep_ms = max(0, len(audio_seg) - cut_ms)
                audio_seg = audio_seg[:keep_ms].fade_out(30)
                audio_seg.export(tts_wav_path, format="wav")

            tts_dur = len(audio_seg) / 1000.0
            stretch_factor = max(1.0, tts_dur / srt_dur)
            
            tts_results.append((i, seg, tts_wav_path, audio_seg, stretch_factor, srt_dur, text))
            self.emit_log(f"🎙 TTS #{i+1}/{len(segs)} done | stretch={stretch_factor:.2f}x")
            
            # Update progress (0-50%)
            self.progress.emit(int((i + 1) / len(segs) * 50))
        
        self.emit_log(f"✅ Pha 1 hoàn tất: {len(segs)} segments TTS đã sẵn sàng!")

        # ═══════════════════════════════════════════════
        # PHA 2: ENCODE FFmpeg SONG SONG 3 LUỒNG (GPU NVENC)
        # ═══════════════════════════════════════════════
        self.emit_log(f"🔪 Pha 2: Encode FFmpeg song song 3 luồng GPU...")
        
        # Tính trước time_cursor và danh sách tasks
        encode_tasks = []  # [(task_type, task_args, output_path, order_key)]
        time_cursor = 0.0
        
        for idx, (i, seg, tts_wav_path, audio_seg, stretch_factor, srt_dur, text) in enumerate(tts_results):
            srt_start = seg["start"]
            srt_end = seg["end"]
            
            if not text or not text.strip():
                continue
            
            # 1. GAP CHUNK
            if srt_start > time_cursor + 0.05:
                gap_dur = srt_start - time_cursor
                chunk_gap = os.path.join(tmp, f"chunk_{i:04}a_gap.mkv")
                encode_tasks.append(("gap", {
                    "time_cursor": time_cursor, "gap_dur": gap_dur, "chunk_out": chunk_gap
                }, chunk_gap, f"{i:04}a"))
            
            # 2. SUBTITLE CHUNK
            import pysrt
            mini_srt_path = os.path.join(tmp, f"chunk_sub_{i:04}.srt")
            
            is_fast = abs(stretch_factor - 1.0) < 0.01 and abs(self.orig_video_speed - 1.0) < 0.01
            chunk_play_dur = srt_dur if is_fast else srt_dur * stretch_factor / self.orig_video_speed
            
            mini_sub = pysrt.SubRipItem(
                index=1,
                start=pysrt.SubRipTime(0,0,0,0),
                end=pysrt.SubRipTime(milliseconds=int(chunk_play_dur * 1000)),
                text=seg["text"]
            )
            mini_srt = pysrt.SubRipFile()
            mini_srt.append(mini_sub)
            mini_srt.save(mini_srt_path, encoding='utf-8')
            
            chunk_sub_out = os.path.join(tmp, f"chunk_{i:04}b_sub.mkv")
            encode_tasks.append(("sub", {
                "srt_start": srt_start, "srt_dur": srt_dur, "tts_wav_path": tts_wav_path,
                "mini_srt_path": mini_srt_path, "chunk_out": chunk_sub_out,
                "stretch_factor": stretch_factor, "is_fast": is_fast
            }, chunk_sub_out, f"{i:04}b"))
            
            time_cursor = srt_end
        
        def _encode_one(task):
            task_type, args, output_path, order_key = task
            
            if task_type == "gap":
                cmd = [FFMPEG, "-y", "-hwaccel", "cuda",
                       "-ss", str(args["time_cursor"]), "-t", str(args["gap_dur"]), "-i", self.video]
                if bg_audio_file != self.video:
                    if getattr(self, 'is_random_bgm', False):
                        cmd.extend(["-stream_loop", "-1", "-ss", str(args["time_cursor"]), "-t", str(args["gap_dur"]), "-i", bg_audio_file])
                    else:
                        cmd.extend(["-ss", str(args["time_cursor"]), "-t", str(args["gap_dur"]), "-i", bg_audio_file])
                    ff_vol = "1:a"
                else:
                    ff_vol = "0:a"
                fc = (f"[0:v]setpts={(1.0/self.orig_video_speed):.5f}*PTS[v]; "
                      f"[{ff_vol}]atempo={self.orig_video_speed},volume={bg_volume}[aout]")
                cmd.extend(["-filter_complex", fc, "-map", "[v]", "-map", "[aout]",
                           "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "22", "-r", "30",
                           "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", "-shortest", args["chunk_out"]])
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    if "-hwaccel" in cmd:
                        hw_idx = cmd.index("-hwaccel")
                        del cmd[hw_idx:hw_idx+2]
                    v_idx = cmd.index("h264_nvenc")
                    cmd[v_idx] = "libx264"
                    cmd[cmd.index("-preset") + 1] = "superfast"
                    q_idx = cmd.index("-qp")
                    cmd[q_idx] = "-crf"
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            elif task_type == "sub":
                safe_mini_srt = args["mini_srt_path"].replace("\\", "/").replace(":", "\\\\:")
                cmd = [FFMPEG, "-y", "-hwaccel", "cuda",
                       "-ss", str(args["srt_start"]), "-t", str(args["srt_dur"]), "-i", self.video]
                if bg_audio_file != self.video:
                    if getattr(self, 'is_random_bgm', False):
                        cmd.extend(["-stream_loop", "-1", "-ss", str(args["srt_start"]), "-t", str(args["srt_dur"]), "-i", bg_audio_file])
                    else:
                        cmd.extend(["-ss", str(args["srt_start"]), "-t", str(args["srt_dur"]), "-i", bg_audio_file])
                    ff_vol = "1:a"
                    tts_idx = "2:a"
                else:
                    ff_vol = "0:a"
                    tts_idx = "1:a"
                cmd.extend(["-i", args["tts_wav_path"]])
                
                if args["is_fast"]:
                    fc = (f"[0:v]subtitles=filename='{safe_mini_srt}':fontsdir='{font_dir}':force_style='{sub_style}'[v]; "
                          f"[{ff_vol}]volume={bg_volume}[a_bg]; "
                          f"[{tts_idx}]volume={self.vol_tts}[a_tts]; "
                          f"[a_bg][a_tts]amix=inputs=2:dropout_transition=0:duration=longest:normalize=0[aout]")
                else:
                    sf = args["stretch_factor"]
                    v_pts = sf * (1.0 / self.orig_video_speed)
                    a_tempo = (1.0 / sf) * self.orig_video_speed
                    
                    a_tempo_filters = []
                    tmp_tempo = max(0.001, a_tempo)
                    while tmp_tempo < 0.5:
                        a_tempo_filters.append("atempo=0.5")
                        tmp_tempo /= 0.5
                    while tmp_tempo > 2.0:
                        a_tempo_filters.append("atempo=2.0")
                        tmp_tempo /= 2.0
                    if abs(tmp_tempo - 1.0) > 0.001:
                        a_tempo_filters.append(f"atempo={tmp_tempo:.5f}")
                    a_tempo_str = ",".join(a_tempo_filters) if a_tempo_filters else "atempo=1.0"
                    
                    fc = (f"[0:v]setpts={v_pts:.5f}*PTS,"
                          f"subtitles=filename='{safe_mini_srt}':fontsdir='{font_dir}':force_style='{sub_style}'[v]; "
                          f"[{ff_vol}]{a_tempo_str},volume={bg_volume}[a_bg]; "
                          f"[{tts_idx}]volume={self.vol_tts}[a_tts]; "
                          f"[a_bg][a_tts]amix=inputs=2:dropout_transition=0:duration=longest:normalize=0[aout]")
                
                cmd.extend(["-filter_complex", fc, "-map", "[v]", "-map", "[aout]",
                           "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "22", "-r", "30",
                           "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", "-shortest", args["chunk_out"]])
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    if "-hwaccel" in cmd:
                        hw_idx = cmd.index("-hwaccel")
                        del cmd[hw_idx:hw_idx+2]
                    v_idx = cmd.index("h264_nvenc")
                    cmd[v_idx] = "libx264"
                    cmd[cmd.index("-preset") + 1] = "superfast"
                    q_idx = cmd.index("-qp")
                    cmd[q_idx] = "-crf"
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return order_key, output_path
        
        # Chạy song song 3 luồng FFmpeg encode!
        from concurrent.futures import ThreadPoolExecutor, as_completed
        chunk_files = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_encode_one, task): task for task in encode_tasks}
            results = {}
            for future in as_completed(futures):
                try:
                    order_key, output_path = future.result()
                    results[order_key] = output_path
                    completed_count += 1
                    task = futures[future]
                    task_type = task[0]
                    sf_info = ""
                    if task_type == "sub":
                        sf = task[1].get("stretch_factor", 1.0)
                        is_fast = task[1].get("is_fast", False)
                        sf_info = " | 1.0x → Fast" if is_fast else f" | stretch={sf:.2f}x"
                    self.emit_log(f"⚡ [{completed_count}/{len(encode_tasks)}] Chunk {order_key}{sf_info}")
                    
                    # Update progress (50-95%)
                    self.progress.emit(50 + int((completed_count / len(encode_tasks)) * 45))
                except Exception as e:
                    self.emit_log(f"❌ Encode error: {e}")
        
        # Sắp xếp chunk_files theo thứ tự đúng
        for key in sorted(results.keys()):
            chunk_files.append(results[key])
            
        # 3. FINAL GAP CHUNK
        video_dur = ffprobe_duration(self.video)
        if video_dur > time_cursor + 0.05:
            gap_dur = video_dur - time_cursor
            chunk_9999_gap = os.path.join(tmp, "chunk_9999_gap.mkv")
            cmd_gap = [FFMPEG, "-y", "-hwaccel", "cuda", "-ss", str(time_cursor), "-t", str(gap_dur), "-i", self.video]
            if bg_audio_file != self.video:
                if getattr(self, 'is_random_bgm', False):
                    cmd_gap.extend(["-stream_loop", "-1", "-ss", str(time_cursor), "-t", str(gap_dur), "-i", bg_audio_file])
                else:
                    cmd_gap.extend(["-ss", str(time_cursor), "-t", str(gap_dur), "-i", bg_audio_file])
                ff_vol_bgm = "1:a"
            else:
                ff_vol_bgm = "0:a"
                
            filter_complex_gap = (
                f"[0:v]setpts={(1.0/self.orig_video_speed):.5f}*PTS[v]; "
                f"[{ff_vol_bgm}]atempo={self.orig_video_speed},volume={bg_volume}[aout]"
            )
            cmd_gap.extend([
                "-filter_complex", filter_complex_gap,
                "-map", "[v]", "-map", "[aout]",
                "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "22", "-r", "30",
                "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", "-shortest", chunk_9999_gap
            ])
            try:
                subprocess.run(cmd_gap, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                if "-hwaccel" in cmd_gap:
                    hw_idx = cmd_gap.index("-hwaccel")
                    del cmd_gap[hw_idx:hw_idx+2]
                v_idx = cmd_gap.index("h264_nvenc")
                cmd_gap[v_idx] = "libx264"
                cmd_gap[cmd_gap.index("-preset") + 1] = "superfast"
                q_idx = cmd_gap.index("-qp")
                cmd_gap[q_idx] = "-crf"
                subprocess.run(cmd_gap, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            chunk_files.append(chunk_9999_gap)

        self.emit_log(f"🔗 Bắt đầu ráp nối {len(chunk_files)} khối video bằng phương pháp siêu tốc Concat Demuxer...")
        
        # Concat all chunks
        concat_txt_path = os.path.join(tmp, "concat.txt")
        with open(concat_txt_path, "w", encoding="utf-8") as f:
            for c in chunk_files:
                safe_path = c.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
                
        cmd_concat = [
            FFMPEG, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_txt_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            self.out
        ]
        
        try:
            subprocess.run(cmd_concat, check=True)
            self.emit_log(f"🎉 DONE! Quá trình Render Stretch Frame hoàn mỹ 100%.")
        except subprocess.CalledProcessError as e:
            self.emit_log(f"⚠️ Process Render Concat LỖI: {e}")

        # CRITICAL: Clean up temp folder
        try:
            # import shutil
            # shutil.rmtree(tmp)
            self.emit_log(f"🧹 TẠM DỪNG XÓA TEMP để Debug: {tmp}")
        except Exception as e:
            self.emit_log(f"⚠️ Failed to clean temp folder: {e}")

    def _run_impl_monolithic(self):
        segs = parse_srt(self.srt)
        self.emit_log(f"📑 Có {len(segs)} câu SRT")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_base = os.path.join(script_dir, '.temp_wav_sync')
        os.makedirs(temp_base, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="wav_sync_mono_", dir=temp_base)
        self.emit_log(f"📂 TEMP WAV FOLDER (Monolithic): {tmp}")

        try:
            tail_dur_cached = get_tail_duration_sec(1.0, tmp, api_url=self.tts_api_url)
            self.emit_log(f"🧷 Tail duration cached @speed=1.0: {tail_dur_cached:.3f}s")
        except Exception as e:
            tail_dur_cached = None
            self.emit_log(f"⚠️ Không đo được tail duration (sẽ bỏ short-fix): {e}")

        bg_audio_file = self.video
        self.is_random_bgm = False
        import random
        if self.bgm_files:
            valid_bgm = [f for f in self.bgm_files if os.path.exists(f)]
            if valid_bgm:
                if self.bgm_random:
                    bg_audio_file = random.choice(valid_bgm)
                else:
                    bg_audio_file = valid_bgm[0]
                self.emit_log(f"🎵 BGM (Monolithic): {os.path.basename(bg_audio_file)}")
                self.uvr_enabled = False
                self.is_random_bgm = True
        if self.uvr_enabled and self.uvr_strength > 0:
            self.emit_log("🎙 Bật Khử Giọng UVR5. Trích xuất âm thanh gốc để tách...")
            orig_wav = os.path.join(tmp, "original_audio.wav")
            subprocess.run([FFMPEG, "-y", "-i", self.video, "-q:a", "0", "-map", "a", orig_wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(orig_wav):
                try:
                    self.emit_log("🎙 audio-separator đang lột giọng bằng MDX-Net...")
                    cmd_uvr = [
                        sys.executable, "-c",
                        f"import logging; logging.basicConfig(level=logging.INFO); from audio_separator.separator import Separator; s = Separator(output_dir='{tmp}', output_format='WAV'); s.load_model('UVR-MDX-NET-Inst_HQ_3.onnx'); s.separate('{orig_wav}')"
                    ]
                    # Bỏ DEVNULL để hiển thị tiến trình ra Terminal
                    uvr_process = subprocess.run(cmd_uvr, check=True, capture_output=True, text=True)
                    self.emit_log("✅ Tách giọng UVR xong (Monolithic)! Output log:")
                    self.emit_log(uvr_process.stdout)
                    instr_files = [f for f in os.listdir(tmp) if "Instrumental" in f and f.endswith(".wav")]
                    if instr_files:
                        self.emit_log(f"✅ Khử giọng {self.uvr_strength}%! Giữ lại {100-self.uvr_strength}% giọng gốc.")
                        inst_path = os.path.join(tmp, instr_files[0])
                        if self.uvr_strength < 100:
                            vocal_files = [f for f in os.listdir(tmp) if "Vocals" in f and f.endswith(".wav")]
                            vocal_path = os.path.join(tmp, vocal_files[0]) if vocal_files else None
                            if vocal_path and os.path.exists(vocal_path):
                                mixed_bg = os.path.join(tmp, "mixed_bg.wav")
                                import math
                                raw_ratio = (100 - self.uvr_strength) / 100.0
                                vol_vocal = math.sqrt(raw_ratio) if raw_ratio > 0 else 0.0
                                mix_cmd = [FFMPEG, "-y", "-i", inst_path, "-i", vocal_path, "-filter_complex", f"[0:a]volume=1.0[a1]; [1:a]volume={vol_vocal}[a2]; [a1][a2]amix=inputs=2:duration=longest:normalize=0", mixed_bg]
                                subprocess.run(mix_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                bg_audio_file = mixed_bg
                            else:
                                bg_audio_file = inst_path
                        else:
                            bg_audio_file = inst_path
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    self.emit_log(f"⚠️ UVR5 Lỗi: {e}\nTraceback:\n{tb}\nSẽ dùng audio gốc.")

        self.emit_log("🔪 Đang dán đè âm thanh vào 1 trục Timeline bằng NumPy (Siêu tốc độ, Không ngốn RAM)...")
        import numpy as np
        
        video_dur = ffprobe_duration(self.video)
        sample_rate = 44100
        total_dur_ms = int(video_dur * 1000 + 120000) # Thêm dư dả 120 giây
        total_samples = int((total_dur_ms / 1000.0) * sample_rate)
        
        timeline_float = np.zeros(total_samples, dtype=np.float32)

        last_audio_end_ms = 0  # Dùng cho queue_audio

        for i, seg in enumerate(segs):
            raw_text = seg["text"]
            text_for_tts = clean_text_for_tts(raw_text)
            words = [w for w in text_for_tts.split() if w.strip()]
            is_short = (len(words) < SHORT_WORDS_THRESHOLD) and (tail_dur_cached is not None)
            tail_dur = 0.0

            if is_short and not self.tts_api_url.strip().lower().startswith(("native", "piper", "chumtts", "kokoro", "minimax", "edge")):
                text_clean = text_for_tts.replace('.', ',').replace('?', ',').replace('!', ',')
                text_for_tts = f"{text_clean}, {TAIL_TEXT}"
                tail_dur = float(tail_dur_cached)
            
            srt_start = seg["start"]
            tts_wav_path = os.path.join(tmp, f"tts_{i:04d}.wav")
            
            try:
                if self.tts_api_url.strip().lower() in ["native", "piper", "chumtts"]:
                    from chumtts_bridge import chumtts_generate_wav
                    chumtts_generate_wav(text_for_tts, 1.0, tts_wav_path)
                elif self.tts_api_url.strip().lower().startswith("kokoro"):
                    # Determine voice
                    v_name = self.tts_api_url.split(":", 1)[1].strip() if ":" in self.tts_api_url else getattr(self, "kokoro_voice", "af_bella")
                    srt_index = i + 1
                    if hasattr(self, 'line_mapping') and srt_index in self.line_mapping:
                        mapped = self.line_mapping[srt_index]
                        if not mapped.startswith("mm:"):
                            v_name = mapped
                    elif hasattr(self, 'speaker_idx_map') and srt_index in self.speaker_idx_map:
                        spk_id = self.speaker_idx_map[srt_index]
                        if spk_id in self.speaker_mapping:
                            v_name = self.speaker_mapping[spk_id]
                    
                    from kokoro_bridge import kokoro_generate_wav
                    kokoro_generate_wav(text_for_tts, 1.0, tts_wav_path, voice=v_name)
                elif self.tts_api_url.strip().lower().startswith("minimax"):
                    # Determine MiniMax voice
                    mm_voice = getattr(self, "minimax_voice", "Vietnamese_Serene_Man")
                    mm_model = getattr(self, "minimax_model", "speech-2.8-hd")
                    mm_emotion = self.line_emotion.get(i + 1, "neutral") if hasattr(self, 'line_emotion') else "neutral"
                    srt_index = i + 1
                    if hasattr(self, 'line_mapping') and srt_index in self.line_mapping:
                        mapped = self.line_mapping[srt_index]
                        if mapped.startswith("mm:"):
                            mm_voice = mapped[3:]
                    elif hasattr(self, 'speaker_idx_map') and srt_index in self.speaker_idx_map:
                        spk_id = self.speaker_idx_map[srt_index]
                        if spk_id in self.speaker_mapping:
                            mapped = self.speaker_mapping[spk_id]
                            if mapped.startswith("mm:"):
                                mm_voice = mapped[3:]
                    
                    from minimax_bridge import minimax_generate_wav
                    minimax_generate_wav(text_for_tts, self.speed, tts_wav_path, voice_id=mm_voice, emotion=mm_emotion, model=mm_model)
                elif self.tts_api_url.strip().lower().startswith("edge"):
                    from edge_tts_bridge import edge_tts_generate_wav, get_default_voice_for_lang
                    edge_voice = self.tts_api_url.split(":", 1)[1].strip() if ":" in self.tts_api_url else None
                    if not edge_voice and hasattr(self, 'target_lang') and self.target_lang:
                        edge_voice = get_default_voice_for_lang(self.target_lang)
                    if not edge_voice:
                        edge_voice = "en-US-AriaNeural"
                    edge_tts_generate_wav(text_for_tts, self.speed, tts_wav_path, voice=edge_voice)
                else:
                    mp3_path = os.path.join(tmp, f"tts_{i:04d}.mp3")
                    tts_api_get_mp3(text_for_tts, 1.0, mp3_path, api_url=self.tts_api_url)
                    mp3_to_wav(mp3_path, tts_wav_path)
            except Exception as e:
                self.emit_log(f"⚠️ Lỗi TTS Segment #{i+1}: {e}")
                continue
                
            if os.path.exists(tts_wav_path):
                if is_short and tail_dur > 0:
                    try:
                        audio_seg_temp = AudioSegment.from_wav(tts_wav_path)
                        cut_ms = int(max(0.0, (tail_dur + TAIL_CUT_BUFFER_SEC)) * 1000.0)
                        keep_ms = max(0, len(audio_seg_temp) - cut_ms)
                        audio_seg_temp = audio_seg_temp[:keep_ms].fade_out(30)
                        audio_seg_temp.export(tts_wav_path, format="wav")
                    except Exception as e:
                        self.emit_log(f"⚠️ Lỗi cắt padding Segment #{i+1}: {e}")

                # Tăng tốc audio theo TTS Speed setting (ví dụ: 1.5x)
                if self.speed and self.speed != 1.0:
                    sped_path = tts_wav_path.replace(".wav", "_sped.wav")
                    atempo_val = float(self.speed)
                    # FFmpeg atempo chỉ chấp nhận 0.5-2.0, chain nếu cần
                    atempo_filters = []
                    remaining = atempo_val
                    while remaining > 2.0:
                        atempo_filters.append("atempo=2.0")
                        remaining /= 2.0
                    while remaining < 0.5:
                        atempo_filters.append("atempo=0.5")
                        remaining *= 2.0
                    atempo_filters.append(f"atempo={remaining:.4f}")
                    atempo_chain = ",".join(atempo_filters)
                    
                    subprocess.run(
                        [FFMPEG, "-y", "-i", tts_wav_path, "-filter:a", atempo_chain, sped_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    if os.path.exists(sped_path):
                        os.replace(sped_path, tts_wav_path)

                seg_tts = AudioSegment.from_wav(tts_wav_path).set_frame_rate(44100).set_channels(1)
                
                if self.vol_tts != 1.0:
                    import math
                    db_change = 20 * math.log10(self.vol_tts) if self.vol_tts > 0 else -100
                    seg_tts = seg_tts + db_change
                    
                position_ms = int(srt_start * 1000)
                
                # Nếu là mode Xếp Hàng, đẩy audio lùi lại nếu thời gian hiện màu < end của clip trước
                if self.sync_mode == "queue_audio":
                    if position_ms < last_audio_end_ms:
                        position_ms = last_audio_end_ms
                        
                # --- CHÈN ARRAY NumPy Siêu Nhanh ---
                samples = np.array(seg_tts.get_array_of_samples(), dtype=np.float32)
                start_sample = int((position_ms / 1000.0) * sample_rate)
                end_sample = start_sample + len(samples)
                
                if end_sample > total_samples:
                    samples = samples[:total_samples - start_sample]
                    end_sample = total_samples
                    
                timeline_float[start_sample:end_sample] += samples
                last_audio_end_ms = position_ms + len(seg_tts)
                
            self.emit_log(f"⚡ Đã dán audio Segment #{i+1} vào Timeline!")
            
            # Thả lỏng cho GPU và CPU thở 10ms để tránh treo đơ toàn bộ máy tính (mượt GUI)
            import time
            time.sleep(0.01)
            
        final_tts_audio = os.path.join(tmp, "final_tts_layer.wav")
        # Chuẩn hóa về int16 an toàn
        np.clip(timeline_float, -32768, 32767, out=timeline_float)
        final_data = timeline_float.astype(np.int16).tobytes()
        
        final_seg = AudioSegment(
            data=final_data,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )
        final_seg.export(final_tts_audio, format="wav")
        
        self.emit_log("🔗 Bắt đầu ráp Video & Âm thanh bằng FFmpeg...")
        vol_bgm = self.vol_bgm_import if getattr(self, 'is_random_bgm', False) else self.vol_bgm
        
        # Load subtitle style từ Settings của user (KHÔNG hardcode!)
        _, video_h = ffprobe_resolution(self.video)
        sub_style = build_ass_style(video_height=video_h, overrides=self.sub_overrides, target_lang=self.target_lang)
        # Trỏ fontsdir vào đúng thư mục chứa font, KHÔNG trỏ vào project dir
        font_dir = os.path.join(BASE_DIR, "fonts").replace("\\", "/")
        os.makedirs(font_dir, exist_ok=True)
        # Copy font file vào thư mục fonts nếu chưa có
        impact_font_src = os.path.join(BASE_DIR, "UTM-IMPACT.TTF")
        if os.path.exists(impact_font_src):
            import shutil
            shutil.copy2(impact_font_src, os.path.join(font_dir, "UTM-IMPACT.TTF"))
        
        safe_srt = self.srt.replace("\\", "/").replace(":", "\\:")
        sub_filter = f"[0:v]subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'[vout]"
            
        if bg_audio_file != self.video:
            ff_bg = "1:a"
            if getattr(self, 'is_random_bgm', False):
                ff_in = ["-i", self.video, "-stream_loop", "-1", "-i", bg_audio_file, "-i", final_tts_audio]
            else:
                ff_in = ["-i", self.video, "-i", bg_audio_file, "-i", final_tts_audio]
        else:
            ff_bg = "0:a"
            ff_in = ["-i", self.video, "-i", final_tts_audio]
            
        ff_tts = f"{len(ff_in) // 2 - 1}:a"
        
        filter_complex = f"{sub_filter}; [{ff_bg}]volume={vol_bgm}[bg]; [bg][{ff_tts}]amix=inputs=2:duration=longest:normalize=0[aout]"
        
        cmd = [FFMPEG, "-y"] + ff_in + [
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "h264_nvenc", "-preset", "p1", "-qp", "22", "-r", "30",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", self.out
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            if "-hwaccel" in cmd:
                hw_idx = cmd.index("-hwaccel")
                del cmd[hw_idx:hw_idx+2]
            v_idx = cmd.index("h264_nvenc")
            cmd[v_idx] = "libx264"
            cmd[cmd.index("-preset") + 1] = "superfast"
            q_idx = cmd.index("-qp")
            cmd[q_idx] = "-crf"
            subprocess.run(cmd, check=True)
        self.emit_log("🎉 DONE! Quá trình Render Monolithic hoàn mỹ 100%.")

        try:
            # import shutil
            # shutil.rmtree(tmp)
            pass
        except:
            pass
