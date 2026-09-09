"""
minimax_bridge.py — Bridge module cho MiniMax TTS API.
Tương tự kokoro_bridge.py nhưng gọi API MiniMax cloud.
"""
import os
import json
import base64
import requests
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MINIMAX_DIR = os.path.join(BASE_DIR, "minimax")
CONFIG_PATH = os.path.join(MINIMAX_DIR, "config.json")
VOICES_PATH = os.path.join(MINIMAX_DIR, "voices.json")

FFMPEG = None
# Detect ffmpeg
for p in [os.path.join(BASE_DIR, "ffmpeg"), "/usr/bin/ffmpeg", "ffmpeg"]:
    if os.path.isfile(p):
        FFMPEG = p
        break
if FFMPEG is None:
    FFMPEG = "ffmpeg"


def _decode_key(enc_key: str) -> str:
    """Decode API key from config.json (XOR + base64 encoding)."""
    if not enc_key:
        return ""
    if not enc_key.startswith("enc_"):
        return enc_key  # Legacy plain text
    try:
        enc_key = enc_key[4:]
        decoded = base64.b64decode(enc_key).decode("utf-8")
        salt = "minimax_secret_salt"
        res = []
        for i in range(len(decoded)):
            res.append(chr(ord(decoded[i]) ^ ord(salt[i % len(salt)])))
        return "".join(res)
    except Exception:
        return ""


def get_api_key() -> str:
    """Load and decode the MiniMax API key from config.json."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Không tìm thấy config MiniMax tại {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    key = _decode_key(config.get("api_key", ""))
    if not key:
        raise ValueError("API key MiniMax trống hoặc không decode được!")
    return key


def get_minimax_voices() -> dict:
    """Load voice list from voices.json. Returns dict grouped by language."""
    if not os.path.exists(VOICES_PATH):
        return {}
    with open(VOICES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_minimax_voice_list_flat() -> list:
    """Return flat list of all voice IDs for UI combobox."""
    voices = get_minimax_voices()
    result = []
    for lang, voice_list in voices.items():
        for v in voice_list:
            if isinstance(v, dict) and "id" in v:
                result.append(v["id"])
    return result


def minimax_generate_wav(text: str, speed: float, out_wav_path: str,
                         voice_id: str = "Vietnamese_Serene_Man",
                         emotion: str = "neutral",
                         model: str = "speech-2.8-hd"):
    """
    Call MiniMax TTS API and save output as WAV (44100Hz, mono).
    
    Args:
        text: Text to synthesize
        speed: TTS speed (0.1 - 3.0)
        out_wav_path: Output WAV file path
        voice_id: MiniMax voice ID (e.g. "Vietnamese_Serene_Man")
        emotion: Emotion tag (neutral, happy, sad, angry, fearful, disgusted, surprised, fluent)
        model: MiniMax model name
    """
    api_key = get_api_key()
    
    url = "https://api.minimax.io/v1/t2a_v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "text": text,
        "stream": False,
        "language_boost": "auto",
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "vol": 1.0,
            "pitch": 0,
            "emotion": emotion
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        raise RuntimeError(f"MiniMax API HTTP Error: {response.status_code} - {response.text[:200]}")
    
    data = response.json()
    if "data" not in data or "audio" not in data["data"]:
        error_msg = data.get("base_resp", {}).get("status_msg", response.text[:200])
        raise RuntimeError(f"MiniMax API Error: {error_msg}")
    
    audio_hex = data["data"]["audio"]
    audio_bytes = bytes.fromhex(audio_hex)
    
    # Save as temp MP3 first
    mp3_path = out_wav_path.replace(".wav", "_mm_tmp.mp3")
    with open(mp3_path, "wb") as f:
        f.write(audio_bytes)
    
    # Convert MP3 -> WAV (44100Hz, mono) using ffmpeg
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", mp3_path, "-ar", "44100", "-ac", "1", out_wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    finally:
        # Cleanup temp MP3
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
    
    if not os.path.exists(out_wav_path):
        raise RuntimeError(f"MiniMax TTS: Không tạo được file WAV tại {out_wav_path}")
    
    return out_wav_path
