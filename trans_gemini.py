

# trans_gemini.py
# pip install google-generativeai

import re
from typing import List, Tuple
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import google.genai as genai

# ====== CẤU HÌNH ======
API_KEY = "AIzaSyCRHTvCoOQAIU78SBmniiJoOEQnn4-ERtk"
MODEL_NAME = "gemini-2.5-flash-lite"   # hoặc "gemini-3.5-pro" nếu muốn
MAX_LINES_PER_CHUNK = 40          # ~30–40 dòng / lần như mày nói
# ======================

def _read_text_guess_encoding(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r") as f:
        return f.read()

def _write_text_utf8(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# --------- SRT helpers ----------

SRT_BLOCK_RE = re.compile(
    r"(?P<idx>\d+)\s*\r?\n"
    r"(?P<time>\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\s*\r?\n"
    r"(?P<text>(?:.*(?:\r?\n)?)*?)(?=\r?\n\r?\n|\Z)",
    re.MULTILINE
)

def parse_srt_blocks(srt_text: str) -> List[Tuple[str, str, List[str]]]:
    """
    Trả về list (idx, time, [text_lines])
    - text_lines: list từng dòng thoại (có thể 1 hoặc 2 dòng)
    """
    blocks = []
    for m in SRT_BLOCK_RE.finditer(srt_text.strip()):
        idx = m.group("idx").strip()
        time = m.group("time").strip()
        text = m.group("text").strip()
        # tách từng dòng thoại
        if text:
            text_lines = [ln.strip() for ln in re.split(r"\r?\n", text)]
        else:
            text_lines = []
        blocks.append((idx, time, text_lines))
    return blocks

def join_srt_blocks(blocks: List[Tuple[str, str, List[str]]]) -> str:
    """
    Ghép lại SRT từ (idx, time, [text_lines])
    """
    out_lines = []
    for idx, time, text_lines in blocks:
        out_lines.append(str(idx))
        out_lines.append(time)
        if text_lines:
            out_lines.extend(text_lines)
        out_lines.append("")  # dòng trống giữa block
    return "\n".join(out_lines).strip() + "\n"

def flatten_text_lines(blocks: List[Tuple[str, str, List[str]]]):
    """
    Từ list block -> list tất cả các dòng thoại + map để ghép ngược lại.

    return:
      all_lines: List[str]   (line_0, line_1, ...)
      layout: List[Tuple[int, int]]
          layout[i] = (start_idx, count)
          nghĩa là block i dùng all_lines[start_idx : start_idx+count]
    """
    all_lines: List[str] = []
    layout: List[Tuple[int, int]] = []
    cur = 0
    for _, _, text_lines in blocks:
        cnt = len(text_lines)
        layout.append((cur, cnt))
        all_lines.extend(text_lines)
        cur += cnt
    return all_lines, layout

def rebuild_blocks_from_lines(
    blocks_meta: List[Tuple[str, str]],
    new_lines: List[str],
    layout: List[Tuple[int, int]],
) -> List[Tuple[str, str, List[str]]]:
    """
    Ghép lại block sau khi đã dịch:
      - blocks_meta: list (idx, time)
      - new_lines: toàn bộ dòng thoại đã dịch
      - layout: mapping từng block -> slice trong new_lines
    """
    out = []
    for i, (idx, time) in enumerate(blocks_meta):
        start, cnt = layout[i]
        text_lines = new_lines[start : start + cnt] if cnt > 0 else []
        out.append((idx, time, text_lines))
    return out

def chunk_list(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

# --------------------------------- GEMINI ---------------------------------

def _build_prompt(target_lang: str = "vi") -> str:
    lang_name = "Việt" if target_lang.lower() == "vi" else target_lang
    return f"""
Bạn là trình dịch phụ đề chuyên nghiệp.

Nhiệm vụ:
- Dịch danh sách các dòng THOẠI từ tiếng Trung sang tiếng {lang_name}.
- Mỗi dòng input tương ứng 1 dòng subtitle.
- PHẢI giữ nguyên số lượng dòng, thứ tự dòng.

Định dạng INPUT:
- Mình sẽ gửi nhiều dòng, mỗi dòng có dạng:
  NNN|TEXT
  trong đó NNN là số thứ tự dòng (000, 001, 002, ...).
- NNN chỉ dùng để đánh dấu, KHÔNG cần dịch.

Yêu cầu OUTPUT BẮT BUỘC:
- Trả về MỖI DÒNG dưới dạng:
  NNN|DỊCH
- Dùng đúng lại NNN y hệt input.
- KHÔNG thêm dòng mới, không bớt dòng.
- Không chèn thêm giải thích, không thêm chú thích.
- Không dùng code block, không thêm ``` bất kỳ dạng nào.
- Tổng số dòng output phải đúng bằng số dòng input.
""".strip()

def _translate_lines_with_gemini(lines: List[str], target_lang: str = "vi") -> List[str]:
    """
    Dịch list dòng thoại, giữ nguyên số lượng & thứ tự.
    """
    if not lines:
        return []

    if not API_KEY:
        raise RuntimeError("Thiếu API_KEY. Hãy điền API_KEY ở đầu file trans_gemini.py.")

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = _build_prompt(target_lang)

    # Build input: NNN|TEXT
    # dùng index padding 3 chữ số để chắc
    in_lines = []
    for i, text in enumerate(lines):
        tag = f"{i:03d}"
        in_lines.append(f"{tag}|{text}")

    input_text = "\n".join(in_lines)

    resp = model.generate_content([prompt, input_text])
    t = getattr(resp, "text", "") or ""
    t = t.strip()
    # gỡ code fence nếu nó cố tình wrap
    if t.startswith("```"):
        t = t.strip("`").strip()
        # nếu nó có prefix kiểu "text" / "srt" thì cắt đi
        first_newline = t.find("\n")
        if first_newline != -1:
            maybe_header = t[:first_newline].lower()
            if "srt" in maybe_header or "text" in maybe_header:
                t = t[first_newline+1:].lstrip()

    # Parse output
    out_lines = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        # mong đợi: NNN|DỊCH
        if "|" in line:
            tag, content = line.split("|", 1)
            tag = tag.strip()
            content = content.strip()
            # bỏ prefix số & chỉ giữ content, vì thứ tự đã đảm bảo
            out_lines.append(content)
        else:
            # fallback: nếu nó phá format, coi cả line như nội dung
            out_lines.append(line)

    # Nếu model cho ít hơn nhiều, ta pad cho đủ độ dài
    if len(out_lines) < len(lines):
        out_lines.extend([""] * (len(lines) - len(out_lines)))
    # Nếu nó trả thừa (hiếm), cắt bớt
    out_lines = out_lines[: len(lines)]
    return out_lines

def translate_srt_text(
    srt_text: str,
    target_lang: str = "vi",
    progress_cb=None
) -> str:
    """
    Dịch SRT nhưng CHỈ dịch text, giữ nguyên idx + time.
    - Dùng line-based + chunk 30–40 line/lần.
    """
    blocks = parse_srt_blocks(srt_text)
    if not blocks:
        # fallback: không parse được thì cứ dịch thẳng toàn bộ text
        all_lines = [ln for ln in srt_text.splitlines()]
        translated = _translate_lines_with_gemini(all_lines, target_lang)
        return "\n".join(translated)

    # meta: (idx, time)
    blocks_meta = [(idx, time) for (idx, time, _text_lines) in blocks]
    all_lines, layout = flatten_text_lines(blocks)

    if not all_lines:
        # SRT không có text, trả nguyên
        return join_srt_blocks(blocks)

    # chunk theo MAX_LINES_PER_CHUNK
    chunks = chunk_list(all_lines, MAX_LINES_PER_CHUNK)
    translated_all: List[str] = []
    total = len(chunks)

    offset = 0
    for i, ch in enumerate(chunks, 1):
        translated = _translate_lines_with_gemini(ch, target_lang)
        translated_all.extend(translated)
        if progress_cb:
            progress_cb(int(i * 100 / total))
        offset += len(ch)

    # đảm bảo đủ độ dài
    if len(translated_all) < len(all_lines):
        translated_all.extend([""] * (len(all_lines) - len(translated_all)))
    translated_all = translated_all[: len(all_lines)]

    # build lại blocks
    new_blocks = rebuild_blocks_from_lines(blocks_meta, translated_all, layout)
    return join_srt_blocks(new_blocks)

# ----------------- Worker cho PyQt -----------------

class SrtTranslateWorker(QObject):
    finished = pyqtSignal(str)        # emit out_path
    error = pyqtSignal(str)
    progress = pyqtSignal(int)        # 0..100

    def __init__(self, in_path: str, out_path: str, target_lang: str = "vi"):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.target_lang = target_lang

    @pyqtSlot()
    def run(self):
        try:
            srt_text = _read_text_guess_encoding(self.in_path)

            def _on_progress(p):
                self.progress.emit(p)

            result = translate_srt_text(
                srt_text,
                target_lang=self.target_lang,
                progress_cb=_on_progress
            )

            cleaned = result.rstrip("\n")
            _write_text_utf8(self.out_path, cleaned + "\n")
            self.progress.emit(100)
            self.finished.emit(self.out_path)
        except Exception as e:
            self.error.emit(str(e))
