# trans_local.py
# Dịch SRT zh → vi bằng model offline hirashiba-mt-tiny-zh-vi (CPU)
# CHỈ dịch dòng thoại, giữ nguyên STT và timecode.

from typing import List
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "chi-vi/hirashiba-mt-tiny-zh-vi"

print("⏳ Load model dịch offline zh→vi (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to("cpu")
model.eval()
print("✅ Model offline loaded!")


# ============================
# SRT LINE HELPERS
# ============================

def is_index_line(line: str) -> bool:
    """
    Dòng chỉ có số (STT của block SRT)
    Ví dụ: '1', '23'
    """
    s = line.strip()
    return s.isdigit() if s else False


def is_time_line(line: str) -> bool:
    """
    Dòng timecode: chỉ cần chứa '-->' là đủ.
    Ví dụ: '00:00:01,000 --> 00:00:02,000'
    Không cố format lại, giữ nguyên y chang.
    """
    return "-->" in line


def translate_line_zh_vi(text: str) -> str:
    """
    Dịch 1 câu thoại ngắn (1 dòng).
    """
    text = text.strip()
    if not text:
        return text

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out_tokens = model.generate(
            **inputs,
            max_length=80,
            num_beams=4,
        )
    return tokenizer.decode(out_tokens[0], skip_special_tokens=True)


def translate_srt_lines(full_text: str, progress_cb=None) -> str:
    """
    Dịch SRT theo từng dòng:
    - STT (chỉ số): giữ nguyên
    - Timecode (có '-->'): giữ nguyên
    - Dòng trống: giữ nguyên
    - Còn lại: coi là text -> dịch
    """
    # tách dòng nhưng bỏ \n; ta sẽ ghép lại sau
    lines: List[str] = full_text.splitlines()

    # Đếm tổng số dòng cần dịch để báo progress
    text_indices = [
        i for i, ln in enumerate(lines)
        if ln.strip()                      # không trắng
        and not is_index_line(ln)         # không phải STT
        and not is_time_line(ln)          # không phải timecode
    ]
    total_text = len(text_indices) or 1
    done = 0

    out_lines: List[str] = []

    for i, ln in enumerate(lines):
        # Giữ nguyên nếu là STT, time hoặc dòng trống
        if not ln.strip() or is_index_line(ln) or is_time_line(ln):
            out_lines.append(ln)
        else:
            # Dịch dòng thoại
            translated = translate_line_zh_vi(ln)
            out_lines.append(translated)
            done += 1
            if progress_cb:
                progress_cb(int(done * 100 / total_text))

    result = "\n".join(out_lines)
    # đảm bảo file kết thúc bằng newline cho chắc
    if not result.endswith("\n"):
        result += "\n"
    return result


# ============================
# PyQt Worker
# ============================

class LocalSrtWorker(QObject):
    finished = pyqtSignal(str)  # out_path
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, in_path: str, out_path: str, target_lang: str = "vi"):
        """
        Giữ target_lang cho tương thích với trans_gemini.SrtTranslateWorker,
        hiện tại chỉ hỗ trợ zh -> vi nên target_lang bị bỏ qua (hoặc phải là 'vi').
        """
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.target_lang = target_lang  # để đó cho khỏi lỗi

    @pyqtSlot()
    def run(self):
        try:
            with open(self.in_path, "r", encoding="utf-8") as f:
                text = f.read()

            result = translate_srt_lines(text, progress_cb=self.progress.emit)

            with open(self.out_path, "w", encoding="utf-8") as f:
                f.write(result)

            self.progress.emit(100)
            self.finished.emit(self.out_path)

        except Exception as e:
            self.error.emit(str(e))
