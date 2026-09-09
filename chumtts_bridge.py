import os
import wave
import json
import csv
import re
import numpy as np
import onnxruntime
from piper_phonemize import phonemize_espeak

# Lấy đường dẫn base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUMTTS_DIR = os.path.join(BASE_DIR, "chumtts")

class ChumTTSNativeBridge:
    _instance = None
    
    def __init__(self):
        # Paths hardcode theo thư mục chumtts của anh
        self.model_path = os.path.join(CHUMTTS_DIR, "ngochuyen.onnx")
        self.config_path = os.path.join(CHUMTTS_DIR, "ngochuyen.onnx.json")
        self.dict_path = os.path.join(CHUMTTS_DIR, "non-vietnamese-words-20k.csv")
        
        # 1. Load the Voice Configuration 
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.sample_rate = self.config["audio"]["sample_rate"]
        self.phoneme_id_map = self.config["phoneme_id_map"]
        
        # Load dictionary
        self.word_dict = {}
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    next(reader)
                except StopIteration:
                    pass
                for row in reader:
                    if len(row) >= 2:
                        self.word_dict[row[0].lower()] = row[1]
                        
        # 2. Setup ONNX Runtime
        print(f"⏳ Tải mô hình ChumTTS (ONNX) vào 메모리: {self.model_path}")
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Giới hạn thread CPU để không nghẽn
        # VITS model nhỏ, chạy CPU nhanh như chớp, không cần GPU
        # Để GPU rảnh cho UVR5/Inpainting (việc nặng thật sự)
        self.session = onnxruntime.InferenceSession(
            self.model_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        print("✅ ChumTTS Onnx đã tải xong (CPU mode - nhẹ nhàng)!")
        
        # Optimize for maximum phonetic clarity (monotone / robotic but highly accurate)
        self.noise_scale = 0.333  # Reduced from 0.667 for zero vocal cracks/variance
        self.length_scale = 1.15  # Slower base length for distinct pronunciation
        self.noise_w_scale = 0.8  # Keep phoneme rhythm steady

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def normalize_text(self, text):
        from vinorm import TTSnorm
        
        # Normalization chuẩn Vinorm
        try:
            text = TTSnorm(text)
        except Exception:
            pass

        # Sửa lỗi đọc chữ "Chấm" do TTSnorm sinh ra dấu chấm có khoảng trắng
        # Thay thế tất cả dấu chấm bằng dấu phẩy để tạo khoảng nghỉ (pause) tự nhiên
        text = text.replace(".", ",")

        def replacer(match):
            word = match.group(0)
            return self.word_dict.get(word.lower(), word)
            
        return re.sub(r'[\w\-]+', replacer, text)

    def generate_wav(self, text, speed, out_wav_path):
        """
        Sinh audio giống hệt hàm infer.py của ChumTTS
        """
        text_to_read = self.normalize_text(text)
        
        # Scale tốc độ
        local_length_scale = self.length_scale / speed
        scales = np.array([self.noise_scale, local_length_scale, self.noise_w_scale], dtype=np.float32)
        sid = np.array([0], dtype=np.int64) if self.config["num_speakers"] > 1 else None

        # Phonemize
        sentence_phonemes = phonemize_espeak(text_to_read, self.config["espeak"]["voice"])
        
        audio_arrays = []
        def infer_phonemes(phonemes, use_scales):
            ids = []
            ids.extend(self.phoneme_id_map.get('^', [1])) 
            ids.extend(self.phoneme_id_map.get('_', [0])) 
            
            for p in phonemes:
                if p in self.phoneme_id_map:
                    ids.extend(self.phoneme_id_map[p])
                    ids.extend(self.phoneme_id_map.get('_', [0]))
                    
            ids.extend(self.phoneme_id_map.get('$', [2]))
            
            input_ids = np.array([ids], dtype=np.int64)
            input_lengths = np.array([input_ids.shape[1]], dtype=np.int64)

            inputs = {
                "input": input_ids,
                "input_lengths": input_lengths,
                "scales": use_scales
            }
            if sid is not None:
                inputs["sid"] = sid

            outputs = self.session.run(["output"], inputs)
            return outputs[0].squeeze()

        word_count = len(text_to_read.replace(',', '').replace('.', '').split())
        is_short_text = word_count <= 4 and word_count > 0

        if is_short_text and len(sentence_phonemes) == 1:
            # Thêm dấu phẩy [,] vào ngay đầu chuỗi dummy để ép mô hình tạo ra một nhịp ngắt (acoustic pause).
            # Điều này giúp ngăn chặn hiện tượng "co-articulation" (nuốt âm/dính âm) giữa từ cuối của anh (ví dụ chữ "c" trong "nước")
            # và chữ cái đầu của dummy text (chữ "x" trong "xa"), giải quyết triệt để lỗi "nước xờ".
            dummy_text = ", xa ba la ơi hò khoan"
            full_text_for_phonemes = text_to_read.replace(',', '').replace('.', '') + dummy_text
            
            det_scales = np.array([0.001, local_length_scale, 0.001], dtype=np.float32)
            
            try:
                full_b_phonemes = phonemize_espeak(full_text_for_phonemes, self.config["espeak"]["voice"])[0]
                dummy_phonemes = phonemize_espeak(dummy_text, self.config["espeak"]["voice"])[0]
                
                w_full = infer_phonemes(full_b_phonemes, det_scales)
                w_dummy = infer_phonemes(dummy_phonemes, det_scales)
                # Vì đã có dấu phẩy tạo khoảng nghỉ (silence) hoàn hảo để kết thúc tự nhiên chữ "nước",
                # ta KHÔNG CẦN PADDING NỮA. Cắt bằng 0ms sẽ chém trúng đúng khe hở im lặng tuyệt đối trước chữ "x".
                pad_frames = 0
                slice_idx = len(w_full) - len(w_dummy) + pad_frames
                slice_idx = min(max(slice_idx, 0), len(w_full))
                
                audio_arrays.append(w_full[:slice_idx])
            except Exception as e:
                print(f"⚠️ [ChumTTS] Short text splice failed: {e}")
                is_short_text = False
                
        if not is_short_text or len(sentence_phonemes) > 1:
            for phonemes in sentence_phonemes:
                if not phonemes:
                    continue
                audio = infer_phonemes(phonemes, scales)
                audio_arrays.append(audio)
                
                silence = np.zeros(int(0.15 * self.sample_rate), dtype=np.float32)
                audio_arrays.append(silence)

        if not audio_arrays:
            # Fallback nếu câu trống
            final_audio = np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
        else:
            final_audio = np.concatenate(audio_arrays)
            max_val = np.max(np.abs(final_audio))
            if max_val > 0.0:
                final_audio = final_audio / max_val

        # Lưu WAV
        with wave.open(out_wav_path, "wb") as wav_file:
            wav_file.setnchannels(1)  
            wav_file.setsampwidth(2)   
            wav_file.setframerate(self.sample_rate)
            audio_int16 = (final_audio * 32767.0).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
            
        return out_wav_path

def chumtts_generate_wav(text: str, speed: float, out_wav_path: str):
    bridge = ChumTTSNativeBridge.get_instance()
    bridge.generate_wav(text, float(speed), out_wav_path)
