"""
edge_tts_bridge.py — Bridge module cho Microsoft Edge TTS (miễn phí).
Tương tự minimax_bridge.py nhưng dùng edge-tts package.
"""
import os
import asyncio
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FFMPEG = None
for p in [os.path.join(BASE_DIR, "ffmpeg"), "/usr/bin/ffmpeg", "ffmpeg"]:
    if os.path.isfile(p):
        FFMPEG = p
        break
if FFMPEG is None:
    FFMPEG = "ffmpeg"

# Default voices per language (best quality neural voices)
LANG_DEFAULT_VOICES = {
    "Tiếng Việt":           "vi-VN-HoaiMyNeural",
    "English":              "en-US-AriaNeural",
    "Tiếng Ả Rập":         "ar-SA-ZariyahNeural",
    "Tiếng Hindi":          "hi-IN-SwaraNeural",
    "Tiếng Indonesia":      "id-ID-GadisNeural",
    "Tiếng Ý":              "it-IT-IsabellaNeural",
    "Tiếng Tamil":          "ta-IN-PallaviNeural",
    "Tiếng Trung":          "zh-CN-XiaoxiaoNeural",
    "Tiếng Hàn":            "ko-KR-SunHiNeural",
    "Tiếng Nhật":           "ja-JP-NanamiNeural",
    "Tiếng Tây Ban Nha":    "es-ES-ElviraNeural",
    "Tiếng Pháp":           "fr-FR-DeniseNeural",
}

# Cache for voice list
_VOICES_CACHE = None


def get_edge_tts_voices() -> dict:
    """
    Return voice list grouped by language name (matching suffix_to_lang).
    Returns: {"Tiếng Việt": [{"id": "vi-VN-HoaiMyNeural", "name": "Hoài My (Nữ)"},...], ...}
    """
    global _VOICES_CACHE
    if _VOICES_CACHE is not None:
        return _VOICES_CACHE

    try:
        import edge_tts
        voices = asyncio.run(edge_tts.list_voices())
    except Exception as e:
        print(f"⚠️ edge-tts: Không thể load danh sách voices: {e}")
        return {}

    # Map locale prefix to our language names
    locale_to_lang = {
        "vi-VN": "Tiếng Việt",
        "en-US": "English",
        "en-GB": "English",
        "ar-SA": "Tiếng Ả Rập",
        "ar-EG": "Tiếng Ả Rập",
        "hi-IN": "Tiếng Hindi",
        "id-ID": "Tiếng Indonesia",
        "it-IT": "Tiếng Ý",
        "ta-IN": "Tiếng Tamil",
        "ta-SG": "Tiếng Tamil",
        "zh-CN": "Tiếng Trung",
        "ko-KR": "Tiếng Hàn",
        "ja-JP": "Tiếng Nhật",
        "es-ES": "Tiếng Tây Ban Nha",
        "fr-FR": "Tiếng Pháp",
    }

    result = {}
    for v in voices:
        locale = v.get("Locale", "")
        lang_name = locale_to_lang.get(locale)
        if not lang_name:
            continue

        short_name = v["ShortName"]
        # Extract friendly name: "en-US-AriaNeural" -> "Aria"
        parts = short_name.split("-")
        friendly = parts[-1].replace("Neural", "").replace("Multilingual", "")
        gender = "Nữ" if v.get("Gender") == "Female" else "Nam"

        if lang_name not in result:
            result[lang_name] = []
        result[lang_name].append({
            "id": short_name,
            "name": f"{friendly} ({gender})",
        })

    _VOICES_CACHE = result
    return result


def get_edge_tts_voice_list_flat() -> list:
    """Return flat list of all voice IDs for UI combobox."""
    voices = get_edge_tts_voices()
    result = []
    for lang, voice_list in voices.items():
        for v in voice_list:
            if isinstance(v, dict) and "id" in v:
                result.append(v["id"])
    return result


def get_default_voice_for_lang(target_lang: str) -> str:
    """Get the best default voice for a given target language name."""
    return LANG_DEFAULT_VOICES.get(target_lang, "en-US-AriaNeural")


def edge_tts_generate_wav(text: str, speed: float, out_wav_path: str,
                          voice: str = "vi-VN-HoaiMyNeural",
                          retries: int = 3):
    """
    Generate speech using Edge TTS and save as WAV (44100Hz, mono).

    Args:
        text: Text to synthesize
        speed: TTS speed multiplier (1.0 = normal)
        out_wav_path: Output WAV file path
        voice: Edge TTS voice ID (e.g. "vi-VN-HoaiMyNeural")
        retries: Number of retry attempts
    """
    import edge_tts
    import re
    import time

    # Clean text for edge-tts
    clean = text.strip()
    if not clean:
        clean = "..."  # Edge TTS needs at least something

    # Remove problematic characters that cause "No audio received"
    # Keep letters, numbers, spaces, basic punctuation
    clean = re.sub(r'[♪♫★☆▶◀●○■□▲△▼▽⬛⬜🔴🔵🟢🟡⭐🎵🎶💀☠️👿😈]', '', clean)
    # Remove excessive whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Remove lines that are just "..." or similar
    if re.match(r'^[\.\,\!\?\-\s]+$', clean):
        clean = "hmm"  # Replace with a short utterance

    if not clean or len(clean) < 2:
        clean = "hmm"

    # Convert speed to Edge TTS rate format: +50% or -20%
    rate_percent = int((speed - 1.0) * 100)
    if rate_percent >= 0:
        rate_str = f"+{rate_percent}%"
    else:
        rate_str = f"{rate_percent}%"

    # Edge TTS outputs MP3, so save to temp first
    mp3_path = out_wav_path.replace(".wav", "_edge_tmp.mp3")

    async def _generate():
        communicate = edge_tts.Communicate(clean, voice, rate=rate_str)
        await communicate.save(mp3_path)

    last_error = None
    for attempt in range(retries):
        try:
            # Always run in a separate thread to avoid Qt event loop conflicts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_generate())).result(timeout=30)

            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 100:
                break  # Success
            else:
                last_error = "Empty MP3 output"
                if attempt < retries - 1:
                    time.sleep(0.5)
        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                time.sleep(0.5)
            # Clean up failed attempt
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 100:
        # Generate 0.3s silence as fallback instead of crashing
        print(f"⚠️ Edge TTS fallback silence (text='{clean[:30]}...'): {last_error}")
        subprocess.run(
            [FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", "0.3", out_wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return out_wav_path

    # Convert MP3 -> WAV (44100Hz, mono)
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", mp3_path, "-ar", "44100", "-ac", "1", out_wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

    if not os.path.exists(out_wav_path):
        raise RuntimeError(f"Edge TTS: Không tạo được file WAV tại {out_wav_path}")

    return out_wav_path
