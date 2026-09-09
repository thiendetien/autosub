import os
import soundfile as sf

try:
    from kokoro_onnx import Kokoro
except ImportError:
    Kokoro = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KOKORO_DIR = os.path.join(BASE_DIR, "chumtts", "kokoro")

class KokoroNativeBridge:
    _instance = None
    
    def __init__(self):
        if Kokoro is None:
            raise ImportError("Vui lòng cài đặt kokoro-onnx bằng lệnh: pip install kokoro-onnx soundfile")
            
        self.model_path = os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx")
        self.voices_path = os.path.join(KOKORO_DIR, "voices-v1.0.bin")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
            raise FileNotFoundError(f"Không tìm thấy model Kokoro tại {KOKORO_DIR}. Vui lòng tải model và đặt vào thư mục này.")
            
        print(f"⏳ Tải mô hình Kokoro TTS (ONNX) vào bộ nhớ: {self.model_path}")
        self.kokoro = Kokoro(self.model_path, self.voices_path)
        print("✅ Kokoro TTS Onnx đã tải xong!")
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def generate_wav(self, text, speed, out_wav_path, voice="af_bella"):
        actual_voice = voice
        lang = "en-gb" if actual_voice.startswith("b") else "en-us"
        
        # kokoro.create returns samples and sample_rate
        samples, sample_rate = self.kokoro.create(
            text, voice=actual_voice, speed=speed, lang=lang
        )
        
        # Save WAV using soundfile
        sf.write(out_wav_path, samples, sample_rate)
        return out_wav_path

def kokoro_generate_wav(text: str, speed: float, out_wav_path: str, voice: str = "af_bella"):
    bridge = KokoroNativeBridge.get_instance()
    bridge.generate_wav(text, float(speed), out_wav_path, voice=voice)
