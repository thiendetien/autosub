# -*- coding: utf-8 -*-
"""
Subtitle OCR — PaddleOCR v5 (Qt, Windows/Linux) - FIXED VERSION + FULL AUTO
- Fix 1: Merge segments properly across chunk boundaries
- Fix 2: Reduce CPU usage with throttling
- NEW: Full Auto pipeline:
  Video -> OCR (SRT zh) -> Gemini dịch -> TTS + Burn SRT vi -> MP4 tiếng Việt
"""

import time
import multiprocessing as mp
import os
import sys

# Fix Qt conflict with opencv
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import cv2
import math
import traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from PyQt5 import QtCore, QtGui, QtWidgets


from trans_gemini import SrtTranslateWorker as GeminiSrtWorker
# Lazy import: from trans_local import LocalSrtWorker (chỉ import khi dùng)

from srt_to_video import MergeWorker as SrtToVideoMergeWorker



# Từ đoạn này bắt đầu sửa này :> nếu hỏng thì nhớ quay về đây mà fixx nhé chaizoo

from video_inpainting import remove_subtitle_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-detect ffmpeg based on OS
if sys.platform == "win32":
    FFMPEG = os.path.join(BASE_DIR, "ffmpeg.exe")
    if not os.path.isfile(FFMPEG):
        raise FileNotFoundError("Không tìm thấy ffmpeg.exe (cần để mux audio sau inpaint)")
else:
    # Linux/Mac: try system ffmpeg first, then local binary
    import shutil
    FFMPEG = shutil.which("ffmpeg")
    if FFMPEG is None:
        # Try local ffmpeg binary (no .exe extension)
        local_ffmpeg = os.path.join(BASE_DIR, "ffmpeg")
        if os.path.isfile(local_ffmpeg):
            FFMPEG = local_ffmpeg
        else:
            raise FileNotFoundError("Không tìm thấy ffmpeg. Cài bằng: sudo apt install ffmpeg")




# ---- Optional: Hán phồn -> Giản thể (t2s) ----
try:
    from opencc import OpenCC
    OPENCC_OK = True
except Exception:
    OPENCC_OK = False

# ---- Similarity ----
try:
    from rapidfuzz.fuzz import ratio as fuzz_ratio
except Exception:
    from difflib import SequenceMatcher

    def fuzz_ratio(a, b):
        return int(SequenceMatcher(None, a, b).ratio() * 100)

# ---- HUD CPU/GPU ----
import psutil
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# ================= Global Cache for Optimization =================
# CACHE 1: Main Process OCR (for single-thread mode)
_CACHED_OCR = None
_CACHED_CC = None

# CACHE 2: Persistent Process Pool (for parallel mode)
_GLOBAL_POOL = None
_GLOBAL_POOL_SIZE = 0

# Worker-side globals (inside subprocess)
_WORKER_OCR = None
_WORKER_CC = None

def _worker_init(model_name, use_orientation, use_opencc, normalize_simplified):
    """Initializer for worker processes to load model once"""
    global _WORKER_OCR, _WORKER_CC
    from paddleocr import PaddleOCR
    
    # Init OCR
    try:
        if _WORKER_OCR is None:
            print(f"🔧 Worker processing init: loading PaddleOCR...")
            _WORKER_OCR = PaddleOCR(
                device='gpu:0',
                text_recognition_model_name=model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_orientation,
            )
        
        # Init OpenCC
        if use_opencc and normalize_simplified and _WORKER_CC is None:
            try:
                from opencc import OpenCC
                _WORKER_CC = OpenCC("t2s")
            except:
                pass
                
    except Exception as e:
        print(f"❌ Worker init failed: {e}")

def get_process_pool(num_workers, ctx):
    """Get or create a global persistent process pool"""
    global _GLOBAL_POOL, _GLOBAL_POOL_SIZE
    
    # If pool size changes or pool doesn't exist, create new one
    if _GLOBAL_POOL is None or _GLOBAL_POOL_SIZE != num_workers:
        if _GLOBAL_POOL is not None:
            print("🔄 Shutting down old pool to resize...")
            _GLOBAL_POOL.shutdown()
        
        print(f"🚀 Starting new persistent pool with {num_workers} workers...")
        _GLOBAL_POOL = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=("PP-OCRv5_server_rec", False, True, True) # Default args
        )
        _GLOBAL_POOL_SIZE = num_workers
        
    return _GLOBAL_POOL



# ================= Utils =================

def union_box(a, b):
    """(x1,y1,x2,y2) union, handle None"""
    if a is None:
        return b
    if b is None:
        return a
    return (
        float(min(a[0], b[0])),
        float(min(a[1], b[1])),
        float(max(a[2], b[2])),
        float(max(a[3], b[3])),
    )


def smart_detect_subtitle_roi(video_path):
    """
    Smart auto-detect subtitle ROI from video
    - Only OCR bottom 40% of frame
    - Filter text in subtitle zone (bottom 25%)
    - Return [x, y, w, h] or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample 3 frames from different positions
        sample_frames = [
            int(total * 0.3),
            int(total * 0.5),
            int(total * 0.7),
        ]
        
        all_boxes = []
        
        # Quick OCR (no GPU to avoid blocking UI)
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            device='cpu',  # Use CPU for quick detection
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            
            # Crop bottom 40% only
            crop_y = int(H * 0.6)
            bottom_crop = frame[crop_y:, :]
            
            # OCR
            result = ocr.ocr(bottom_crop, cls=False)
            
            # Extract boxes in subtitle zone (bottom 25% of full frame)
            if result and isinstance(result, list):
                for item in result:
                    if hasattr(item, 'dt_polys') and item.dt_polys:
                        for poly in item.dt_polys:
                            # Get Y coordinate (adjusted to full frame)
                            ys = [p[1] + crop_y for p in poly]
                            y_min = min(ys)
                            
                            # Only keep text in subtitle zone
                            if y_min > H * 0.75:
                                # Adjust poly coordinates to full frame
                                adjusted_poly = [[p[0], p[1] + crop_y] for p in poly]
                                all_boxes.append(adjusted_poly)
        
        cap.release()
        
        if not all_boxes:
            # Fallback: estimate
            return None
        
        # Merge all boxes into one ROI
        all_x = []
        all_y = []
        for poly in all_boxes:
            for p in poly:
                all_x.append(p[0])
                all_y.append(p[1])
        
        x1 = int(min(all_x))
        y1 = int(min(all_y))
        x2 = int(max(all_x))
        y2 = int(max(all_y))
        
        # Add padding
        pad = int(H * 0.02)
        x = max(0, x1 - pad)
        y = max(0, y1 - pad)
        w = min(W, x2 + pad) - x
        h = min(H, y2 + pad) - y
        
        return [x, y, w, h]
        
    except Exception as e:
        print(f"Auto-detect ROI failed: {e}")
        return None



def _poly_to_xyxy(poly, roi_x, roi_y, scale_fx, scale_fy):
    """
    poly: 4x2 points in OCR-input coords
    map về VIDEO coords:
      - chia scale (vì crop có thể resize x2)
      - cộng offset ROI (roi_x, roi_y)
    """
    p = np.array(poly, dtype=np.float32).reshape(-1, 2)
    xs = p[:, 0] / float(scale_fx)
    ys = p[:, 1] / float(scale_fy)
    x1 = float(xs.min()) + float(roi_x)
    y1 = float(ys.min()) + float(roi_y)
    x2 = float(xs.max()) + float(roi_x)
    y2 = float(ys.max()) + float(roi_y)
    return (x1, y1, x2, y2)


def extract_joined_text_conf_and_box(res, roi_x, roi_y, scale_fx=1.0, scale_fy=1.0, roi_w=0, watermark_texts=None):
    """
    Trả về:
      joined_text: "line1  line2"
      best_conf: float
      union_box: (x1,y1,x2,y2) trong VIDEO coords (None nếu không có)
    """
    texts = []
    scores = []
    polys = []


    # PaddleOCR 2.7.3: res = [[[box, (text, score)], ...], None]
    if isinstance(res, list) and res and isinstance(res[0], list):
        page_result = res[0] if res else []
        for item in page_result:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                box, (text, score) = item
                t = str(text).strip()
                if t:
                    texts.append(t)
                    scores.append(float(score))
                    polys.append(box)
    # PaddleOCR 3.x format
    elif isinstance(res, list):
        for item in res:
            # dạng object (PaddleOCR v5)
            if hasattr(item, "rec_text"):
                tlist = list(item.rec_text) if item.rec_text is not None else []
                slist = []
                if hasattr(item, "rec_score") and item.rec_score is not None:
                    slist = list(item.rec_score)
                else:
                    slist = [1.0] * len(tlist)

                box_list = []
                if hasattr(item, "boxes") and item.boxes is not None:
                    box_list = list(item.boxes)

                for i, t in enumerate(tlist):
                    t = (str(t) if t is not None else "").strip()
                    if not t:
                        continue
                    texts.append(t)
                    scores.append(float(slist[i]) if i < len(slist) else 1.0)
                    poly = box_list[i] if (box_list and i < len(box_list)) else None
                    polys.append(poly)

            # dạng dict
            elif isinstance(item, dict):
                tlist = item.get("rec_texts", []) or (item.get("res", {}) or {}).get("rec_texts", [])
                slist = item.get("rec_scores", []) or (item.get("res", {}) or {}).get("rec_scores", [])
                if not slist:
                    slist = [1.0] * len(tlist)

                box_arr = item.get("rec_boxes", None)
                if box_arr is None:
                    box_arr = item.get("rec_polys", None)

                box_list = []
                if box_arr is not None:
                    try:
                        box_list = list(np.array(box_arr))
                    except Exception:
                        box_list = []

                for i, t in enumerate(tlist):
                    t = (str(t) if t is not None else "").strip()
                    if not t:
                        continue
                    texts.append(t)
                    scores.append(float(slist[i]) if i < len(slist) else 1.0)
                    poly = box_list[i] if (box_list and i < len(box_list)) else None
                    polys.append(poly)

    if not texts:
        return "", 0.0, None

    # build boxes + order theo y
    entries = []
    for t, sc, poly in zip(texts, scores, polys):
        if poly is None:
            entries.append((t, sc, None))
        else:
            try:
                box = _poly_to_xyxy(poly, roi_x, roi_y, scale_fx, scale_fy)
                entries.append((t, sc, box))
            except Exception:
                entries.append((t, sc, None))

    # === WATERMARK REMOVAL ===
    # Use fuzzy matching to filter out watermark text and its OCR variations.
    # Handles: exact matches, fuzzy variants, and merged watermark items.
    if watermark_texts:
        filtered = []
        for e in entries:
            is_wm = False
            for wm in watermark_texts:
                # Exact match
                if e[0] == wm:
                    is_wm = True
                    break
                # Fuzzy match (handles OCR variations like "bilibili" vs "biibili")
                if fuzz_ratio(e[0].lower(), wm.lower()) > 65:
                    is_wm = True
                    break
                # Substring: watermark text embedded in entry (e.g. "江南漫馆bil")
                if wm in e[0] and len(e[0]) <= len(wm) + 5:
                    is_wm = True
                    break
            if not is_wm:
                filtered.append(e)
        entries = filtered
        if not entries:
            return "", 0.0, None

    # sort by y1 then x1 if has box
    with_box = [e for e in entries if e[2] is not None]
    if with_box:
        entries.sort(key=lambda e: (e[2][1] if e[2] else 1e9, e[2][0] if e[2] else 1e9))

    joined = "  ".join([e[0] for e in entries if e[0]])
    best = float(max([e[1] for e in entries] or [1.0]))

    u = None
    for _, _, b in entries:
        u = union_box(u, b)

    return joined.strip(), best, u



#================== Add remove sub ============
def srt_time_str(seconds: float) -> str:
    """float seconds -> 'HH:MM:SS,mmm' (round ms)"""
    ms = int(max(0.0, seconds) * 1000 + 0.5)
    hh = ms // 3600000
    mm = (ms % 3600000) // 60000
    ss = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"


def build_srt(segments, path):
    lines = []
    for i, seg in enumerate(segments, 1):
        text = (seg["text"] or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{srt_time_str(seg['start'])} --> {srt_time_str(seg['end'])}")
        lines.append(text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def safe_extract_texts_scores_boxes(res):
    """Chuẩn hoá kết quả PaddleOCR 2.x/3.x về (texts, scores, yboxes)"""
    texts, scores, yboxes = [], [], []
    
    # PaddleOCR 2.7.3: res = [[[box, (text, score)], ...], None] hoặc [[...]]
    if isinstance(res, list) and res and isinstance(res[0], list):
        # Lấy list đầu tiên (page đầu)
        page_result = res[0] if res else []
        for item in page_result:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                box, (text, score) = item
                texts.append(str(text))
                scores.append(float(score))
                # Extract y coords from box
                if box is not None:
                    try:
                        by = np.array(box)[:, 1]
                        yboxes.append((float(by.min()), float(by.max())))
                    except:
                        yboxes.append((0.0, 0.0))
                else:
                    yboxes.append((0.0, 0.0))
        return texts, scores, yboxes
    
    # PaddleOCR 3.x format (object-based)
    if isinstance(res, list):
        for item in res:
            if hasattr(item, "rec_text"):
                t = list(item.rec_text) if item.rec_text is not None else []
                s = (
                    list(item.rec_score)
                    if hasattr(item, "rec_score") and item.rec_score is not None
                    else [1.0] * len(t)
                )
                texts.extend([str(x) for x in t])
                scores.extend([float(x) for x in s])
                if hasattr(item, "boxes") and item.boxes is not None:
                    for b in item.boxes:
                        by = np.array(b)[:, 1]
                        yboxes.append((float(by.min()), float(by.max())))
                else:
                    yboxes.extend([(0.0, 0.0)] * len(t))
            elif isinstance(item, dict):
                t = item.get("rec_texts", [])
                s = item.get("rec_scores", [])
                if not t and "res" in item and isinstance(item["res"], dict):
                    t = item["res"].get("rec_texts", [])
                    s = item["res"].get("rec_scores", [])
                texts.extend([str(x) for x in t])
                scores.extend([float(x) for x in s] if s else [1.0] * len(t))

                boxes_arr = None
                if "rec_boxes" in item and item["rec_boxes"] is not None:
                    boxes_arr = np.array(item["rec_boxes"])
                elif "rec_polys" in item and item["rec_polys"] is not None:
                    boxes_arr = np.array(item["rec_polys"])
                if boxes_arr is not None and boxes_arr.size > 0:
                    try:
                        if boxes_arr.ndim == 3 and boxes_arr.shape[-1] == 2:
                            for b in boxes_arr:
                                by = b[:, 1]
                                yboxes.append((float(by.min()), float(by.max())))
                        else:
                            yboxes.extend([(0.0, 0.0)] * len(t))
                    except Exception:
                        yboxes.extend([(0.0, 0.0)] * len(t))
                else:
                    yboxes.extend([(0.0, 0.0)] * len(t))
    return texts, scores, yboxes


# def merge_and_fix_segments(segments, min_duration=0.8, gap_merge=0.25):
#     """
#     Sort + chống chồng chéo + ép tối thiểu + gộp liền kề cùng text.
#     FIX: Thêm logic gộp segments giống nhau gần nhau (từ các chunk khác nhau)
#     """
#     if not segments:
#         return []

#     segments.sort(key=lambda s: (s["start"], s["end"]))

#     # Bước 1: Fix overlapping và min duration
#     fixed = []
#     for seg in segments:
#         s = max(0.0, seg["start"])
#         e = max(s + min_duration, seg["end"])
#         if fixed and s < fixed[-1]["end"] - 1e-3:
#             s = fixed[-1]["end"]
#             e = max(e, s + min_duration)
#         fixed.append({"start": s, "end": e, "text": seg["text"].strip()})

#     # Bước 2: Merge segments liền kề có text giống nhau
#     merged = []
#     for seg in fixed:
#         if not merged:
#             merged.append(seg)
#         else:
#             prev = merged[-1]
#             # Check if same text AND close in time
#             if seg["text"] == prev["text"] and (seg["start"] - prev["end"]) <= gap_merge:
#                 prev["end"] = max(prev["end"], seg["end"])
#             # FIX: Merge similar text (90%+ similarity) across chunk boundaries
#             elif (seg["start"] - prev["end"]) <= gap_merge:
#                 sim = fuzz_ratio(seg["text"], prev["text"])
#                 if sim >= 90:  # Very similar, likely same subtitle
#                     # Keep longer text
#                     if len(seg["text"]) > len(prev["text"]):
#                         prev["text"] = seg["text"]
#                     prev["end"] = max(prev["end"], seg["end"])
#                 else:
#                     merged.append(seg)
#             else:
#                 merged.append(seg)

#     return merged

def merge_and_fix_segments(segments, min_duration=0.8, gap_merge=0.25):
    """
    Sort + chống chồng chéo + ép tối thiểu + gộp liền kề cùng text.
    GIỮ THÊM: box (x1,y1,x2,y2) và union box khi merge
    """
    if not segments:
        return []

    segments.sort(key=lambda s: (s["start"], s["end"]))

    fixed = []
    for seg in segments:
        s = max(0.0, float(seg["start"]))
        e = max(s + float(min_duration), float(seg["end"]))
        if fixed and s < fixed[-1]["end"] - 1e-3:
            s = fixed[-1]["end"]
            e = max(e, s + float(min_duration))
        fixed.append({
            "start": s,
            "end": e,
            "text": (seg.get("text") or "").strip(),
            "box": seg.get("box", None),
        })

    merged = []
    for seg in fixed:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        close_enough = (seg["start"] - prev["end"]) <= float(gap_merge)

        if seg["text"] == prev["text"] and close_enough:
            prev["end"] = max(prev["end"], seg["end"])
            prev["box"] = union_box(prev.get("box"), seg.get("box"))
        elif close_enough:
            sim = fuzz_ratio(seg["text"], prev["text"]) if (seg["text"] and prev["text"]) else 0
            if sim >= 90:
                if len(seg["text"]) > len(prev["text"]):
                    prev["text"] = seg["text"]
                prev["end"] = max(prev["end"], seg["end"])
                prev["box"] = union_box(prev.get("box"), seg.get("box"))
            else:
                merged.append(seg)
        else:
            merged.append(seg)

    return merged


# ============== Watermark Auto-Detection ==============
def detect_watermark_texts(video_path, roi, ocr_engine, n_samples=20, threshold=0.5):
    """
    Sample n_samples frames, collect per-box text items, group similar texts 
    using fuzzy matching, and mark groups that appear in >threshold of frames
    as watermark. Returns set of all watermark text variants.
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return set()
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x, y, w, h = roi
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    
    step = max(1, total // n_samples)
    sample_indices = list(range(0, total, step))[:n_samples]
    
    per_frame_texts = []
    all_texts = set()
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        
        crop = frame[y:y + h, x:x + w]
        sfx = 1.0
        if h < 60 or max(w, h) < 1200:
            crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            sfx = 2.5
        
        res = ocr_engine.ocr(crop, cls=False)
        
        frame_texts = set()
        if isinstance(res, list):
            for item in res:
                if hasattr(item, "rec_text") and item.rec_text is not None:
                    for t in item.rec_text:
                        ts = str(t).strip()
                        if ts:
                            frame_texts.add(ts)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, (list, tuple)) and len(sub) == 2:
                            _, (text, _) = sub
                            ts = str(text).strip()
                            if ts:
                                frame_texts.add(ts)
        
        per_frame_texts.append(frame_texts)
        all_texts.update(frame_texts)
    
    cap.release()
    
    valid_samples = len(per_frame_texts)
    if valid_samples == 0:
        return set()
    
    # === Fuzzy-group similar texts ===
    # OCR reads "bilibili" as "biibili", "bilibii", "Lilibili", etc.
    # Group them so combined count exceeds threshold
    groups = []
    for text in sorted(all_texts):
        placed = False
        for group in groups:
            if fuzz_ratio(text.lower(), group[0].lower()) > 60:
                group.append(text)
                placed = True
                break
        if not placed:
            groups.append([text])
    
    # For each group: count how many frames had ANY member
    min_count = max(1, int(valid_samples * threshold))
    watermarks = set()
    
    for group in groups:
        group_set = set(group)
        frame_count = sum(
            1 for ft in per_frame_texts if ft & group_set
        )
        if frame_count >= min_count:
            watermarks.update(group)
    
    if watermarks:
        print(f"🔍 Detected watermark text ({len(watermarks)} variants): {watermarks}")
    
    return watermarks




# ============== Subprocess worker (CHUNK LIÊN TIẾP) ==============
def _ocr_chunk_worker(args):
    (
        video_path,
        roi,
        start_idx,
        end_idx,
        step,
        fps,
        min_conf,
        sim_threshold,
        min_duration,
        grace,
        normalize_simplified,
        prog_counter,
        watermark_texts_list,  # list (pickleable) of watermark strings
    ) = args
    watermark_texts = set(watermark_texts_list) if watermark_texts_list else None

    from paddleocr import PaddleOCR
    import cv2
    import numpy as np
    import time

    try:
        from opencc import OpenCC
        _opencc_ok = True
    except Exception:
        _opencc_ok = False

    # Use cached worker globals if available
    global _WORKER_OCR, _WORKER_CC
    
    if _WORKER_OCR is not None:
        ocr = _WORKER_OCR
    else:
        # Fallback if caches missing (e.g. not using persistent pool)
        ocr = PaddleOCR(
            device='gpu:0',
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False, 
        )
    
    if _WORKER_CC is not None:
        cc = _WORKER_CC
    else:
        cc = OpenCC("t2s") if (_opencc_ok and normalize_simplified) else None

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x, y, w, h = roi
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    fallback_box = (float(x), float(y), float(x + w), float(y + h))

    segments = []
    cur_text = ""
    cur_start = None
    cur_box = None
    pending_close_until = None

    def norm(s): return (s or "").strip()

    for idx in range(start_idx, end_idx, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            try:
                prog_counter.value += 1
            except Exception:
                pass
            continue

        ts_sec = float(idx) / float(fps)

        crop = frame[y:y + h, x:x + w]
        scale_fx = 1.0
        scale_fy = 1.0
        
        # FIX: Aggressive upscaling for better OCR on small subtitles
        # Always upscale if height is small, or if overall area is small
        if h < 60 or max(w, h) < 1200:
            # 2.5x upscale often helps clearer text edges
            crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            scale_fx = 2.5
            scale_fy = 1.0

        res = ocr.ocr(crop, cls=False)
        t_now, conf, box_now = extract_joined_text_conf_and_box(res, x, y, scale_fx, scale_fy, roi_w=w, watermark_texts=watermark_texts)

        if cc and t_now:
            t_now = cc.convert(t_now)

        if (not t_now) or float(conf) < float(min_conf):
            t_now = ""
            box_now = None

        c_now = norm(cur_text)
        t_now = norm(t_now)

        try:
            from rapidfuzz.fuzz import ratio as _ratio
            sim = _ratio(t_now, c_now) if (t_now and c_now) else 0
        except Exception:
            from difflib import SequenceMatcher
            sim = int(SequenceMatcher(None, t_now, c_now).ratio() * 100) if (t_now and c_now) else 0

        contains = (t_now and c_now) and (t_now in c_now or c_now in t_now)

        if not cur_text:
            if t_now:
                cur_text = t_now
                cur_start = ts_sec
                cur_box = box_now or fallback_box
                pending_close_until = None
        else:
            if t_now:
                if sim >= int(sim_threshold) or contains:
                    pending_close_until = None
                    if contains and len(t_now) > len(cur_text):
                        cur_text = t_now
                    cur_box = union_box(cur_box, box_now) if box_now else cur_box
                else:
                    if pending_close_until is None:
                        pending_close_until = ts_sec + float(grace)
                    elif ts_sec >= pending_close_until:
                        end_t = max(ts_sec, cur_start + float(min_duration))
                        if end_t - cur_start >= float(min_duration) and cur_text:
                            segments.append({"start": cur_start, "end": end_t, "text": cur_text, "box": cur_box or fallback_box})
                        cur_text = t_now
                        cur_start = ts_sec
                        cur_box = box_now or fallback_box
                        pending_close_until = None
            else:
                if pending_close_until is None:
                    pending_close_until = ts_sec + float(grace)
                elif ts_sec >= pending_close_until:
                    end_t = max(ts_sec, cur_start + float(min_duration))
                    if end_t - cur_start >= float(min_duration) and cur_text:
                        segments.append({"start": cur_start, "end": end_t, "text": cur_text, "box": cur_box or fallback_box})
                    cur_text = ""
                    cur_start = None
                    cur_box = None
                    pending_close_until = None

        try:
            prog_counter.value += 1
        except Exception:
            pass



    if cur_text and cur_start is not None:
        end_t = max((end_idx - 1) / fps, cur_start + float(min_duration))
        if end_t - cur_start >= float(min_duration):
            segments.append({"start": cur_start, "end": end_t, "text": cur_text, "box": cur_box or fallback_box})

    cap.release()
    return segments


# ================= UI Widgets =================
class ImageLabel(QtWidgets.QLabel):
    roi_changed = QtCore.pyqtSignal(int, int, int, int)  # x, y, w, h
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)
        self._pix = None
        self.roi = QtCore.QRect(0, 0, 0, 0)
        
        # Mouse selection state
        self.selecting = False
        self.start_point = None
        self.current_point = None
        self.setMouseTracking(True)

    def set_frame(self, frame_bgr, roi_rect=None):
        self.roi = roi_rect or QtCore.QRect(0, 0, 0, 0)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QtGui.QPainter(self)
        if not self._pix:
            p.end()
            return
        pix = self._pix.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        p.drawPixmap(x, y, pix)
        
        # Draw ROI rectangle
        if self.roi.width() > 0 and self.roi.height() > 0:
            img_w = self._pix.width()
            img_h = self._pix.height()
            scale = min(self.width() / img_w, self.height() / img_h)
            r = self.roi
            rx = int(r.x() * scale) + x
            ry = int(r.y() * scale) + y
            rw = int(r.width() * scale)
            rh = int(r.height() * scale)
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
            p.setPen(pen)
            p.drawRect(QtCore.QRect(rx, ry, rw, rh))
        
        # Draw selection rectangle while dragging
        if self.selecting and self.start_point and self.current_point:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
            p.setPen(pen)
            rect = QtCore.QRect(self.start_point, self.current_point).normalized()
            p.drawRect(rect)
        
        p.end()
    
    def mousePressEvent(self, event):
        """Start ROI selection"""
        if event.button() == QtCore.Qt.LeftButton and self._pix:
            # Get image position
            pix = self._pix.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            x_offset = (self.width() - pix.width()) // 2
            y_offset = (self.height() - pix.height()) // 2
            
            # Check if click is within image bounds
            pos = event.pos()
            if (x_offset <= pos.x() <= x_offset + pix.width() and
                y_offset <= pos.y() <= y_offset + pix.height()):
                self.selecting = True
                self.start_point = pos
                self.current_point = pos
    
    def mouseMoveEvent(self, event):
        """Update ROI selection"""
        if self.selecting:
            self.current_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Finish ROI selection"""
        if event.button() == QtCore.Qt.LeftButton and self.selecting:
            self.selecting = False
            
            if self.start_point and self.current_point and self._pix:
                # Get image position and scale
                pix = self._pix.scaled(
                    self.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                x_offset = (self.width() - pix.width()) // 2
                y_offset = (self.height() - pix.height()) // 2
                
                img_w = self._pix.width()
                img_h = self._pix.height()
                scale = min(self.width() / img_w, self.height() / img_h)
                
                # Convert screen coordinates to image coordinates
                x1 = int((self.start_point.x() - x_offset) / scale)
                y1 = int((self.start_point.y() - y_offset) / scale)
                x2 = int((self.current_point.x() - x_offset) / scale)
                y2 = int((self.current_point.y() - y_offset) / scale)
                
                # Normalize and clamp
                x = max(0, min(x1, x2))
                y = max(0, min(y1, y2))
                w = min(img_w - x, abs(x2 - x1))
                h = min(img_h - y, abs(y2 - y1))
                
                if w > 10 and h > 10:  # Minimum size
                    self.roi = QtCore.QRect(x, y, w, h)
                    self.roi_changed.emit(x, y, w, h)
                    self.update()
            
            self.start_point = None
            self.current_point = None


class OCRWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(str, list)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        video_path,
        roi,
        frame_step,
        min_conf,
        sim_threshold,
        min_duration,
        grace_sec,
        normalize_simplified,
        num_workers=1,
        parent=None,
        fixed_out_path=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.roi = tuple(roi)
        self.frame_step = max(1, int(frame_step))
        self.min_conf = float(min_conf)
        self.sim_threshold = int(sim_threshold)
        self.min_duration = float(min_duration)
        self.grace = float(grace_sec)
        self.normalize_simplified = bool(normalize_simplified)
        self.num_workers = max(1, int(num_workers))
        self._stop = False
        self.fixed_out_path = fixed_out_path
        
        # Lazy loading - chỉ init khi cần
        self.ocr = None
        self.cc = None

    def stop(self):
        self._stop = True

    def combine_frame_text(self, res):
        texts, scores, yboxes = safe_extract_texts_scores_boxes(res)
        if not texts:
            return "", 0.0
        if (
            yboxes
            and len(yboxes) == len(texts)
            and any(y1 != 0.0 or y2 != 0.0 for (y1, y2) in yboxes)
        ):
            order = np.argsort([yb[0] for yb in yboxes])
            ordered = [
                texts[i].strip()
                for i in order
                if texts[i] and texts[i].strip()
            ]
        else:
            ordered = [t.strip() for t in texts if t and t.strip()]
        joined = "  ".join(ordered)
        best = max(scores) if scores else 1.0
        return joined, float(best)

    def run_single(self):
        # Use Global Cache for Single Thread
        global _CACHED_OCR, _CACHED_CC
        
        if _CACHED_OCR is None or self.ocr is None:
            self.progress.emit(0)
            
            if _CACHED_OCR is None:
                print("🧠 Loading Global Main OCR Model...")
                from paddleocr import PaddleOCR
                _CACHED_OCR = PaddleOCR(
                    device='gpu:0',
                    text_recognition_model_name="PP-OCRv5_server_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            
            self.ocr = _CACHED_OCR
            
            if OPENCC_OK and self.normalize_simplified:
                if _CACHED_CC is None:
                    _CACHED_CC = OpenCC("t2s")
                self.cc = _CACHED_CC
            else:
                self.cc = None
                
            self.progress.emit(1)
        
        # Ensure self.ocr is set (in case it was already set but _CACHED_OCR was None initially?)
        # Actually self.ocr = _CACHED_OCR above covers it.
        # But if we are re-using OCRWorker instance?
        if self.ocr is None:
            self.ocr = _CACHED_OCR
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Không mở được video.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y, w, h = self.roi
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        fallback_box = (float(x), float(y), float(x + w), float(y + h))

        segments = []

        # === Auto-detect watermark text ===
        self.watermark_texts = detect_watermark_texts(self.video_path, self.roi, self.ocr)

        cur_text = ""
        cur_start = None
        cur_box = None
        pending_close_until = None

        def norm(s): return (s or "").strip()

        i = 0
        last_ts = 0.0
        while i < total and not self._stop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break

            ts_sec = float(i) / float(fps)

            crop = frame[y:y + h, x:x + w]
            scale_fx = 1.0
            scale_fy = 1.0
            # FIX: Aggressive upscaling for better OCR on small subtitles
            if h < 60 or max(w, h) < 1200:
                crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                scale_fx = 2.5
                scale_fy = 2.5

            res = self.ocr.ocr(crop, cls=False)
            t_now, conf, box_now = extract_joined_text_conf_and_box(res, x, y, scale_fx, scale_fy, roi_w=w, watermark_texts=self.watermark_texts)

            if self.cc and t_now:
                t_now = self.cc.convert(t_now)

            if (not t_now) or float(conf) < float(self.min_conf):
                t_now = ""
                box_now = None

            c_now = norm(cur_text)
            t_now = norm(t_now)
            sim = fuzz_ratio(t_now, c_now) if (t_now and c_now) else 0
            contains = (t_now and c_now) and (t_now in c_now or c_now in t_now)

            if not cur_text:
                if t_now:
                    cur_text = t_now
                    cur_start = ts_sec
                    cur_box = box_now or fallback_box
                    pending_close_until = None
            else:
                if t_now:
                    if sim >= self.sim_threshold or contains:
                        pending_close_until = None
                        if contains and len(t_now) > len(cur_text):
                            cur_text = t_now
                        cur_box = union_box(cur_box, box_now) if box_now else cur_box
                    else:
                        if pending_close_until is None:
                            pending_close_until = ts_sec + self.grace
                        elif ts_sec >= pending_close_until:
                            end_t = max(ts_sec, cur_start + self.min_duration)
                            if end_t - cur_start >= self.min_duration and cur_text:
                                segments.append({"start": cur_start, "end": end_t, "text": cur_text, "box": cur_box or fallback_box})
                            cur_text = t_now
                            cur_start = ts_sec
                            cur_box = box_now or fallback_box
                            pending_close_until = None
                else:
                    if pending_close_until is None:
                        pending_close_until = ts_sec + self.grace
                    elif ts_sec >= pending_close_until:
                        end_t = max(ts_sec, cur_start + self.min_duration)
                        if end_t - cur_start >= self.min_duration and cur_text:
                            segments.append({"start": cur_start, "end": end_t, "text": cur_text, "box": cur_box or fallback_box})
                        cur_text = ""
                        cur_start = None
                        cur_box = None
                        pending_close_until = None

            last_ts = ts_sec
            if total > 0:
                self.progress.emit(int(i * 100 / total))
            i += self.frame_step



        if cur_text and cur_start is not None:
            end_t = max(last_ts, cur_start + self.min_duration)
            if end_t - cur_start >= self.min_duration and cur_text:
                segments.append({"start": cur_start, "end": end_t, "text": cur_text, "box": cur_box or fallback_box})

        cap.release()
        return segments


    def run_parallel(self):
        # Load OCR model in main process if not loaded yet
        # This is needed for watermark detection before spawning workers
        global _CACHED_OCR, _CACHED_CC
        if _CACHED_OCR is None:
            print("🧠 Loading OCR model for watermark detection...")
            from paddleocr import PaddleOCR
            _CACHED_OCR = PaddleOCR(
                device='gpu:0',
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        self.ocr = _CACHED_OCR
        self.progress.emit(1)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Không mở được video.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # === Auto-detect watermark text before spawning workers ===
        self.watermark_texts = detect_watermark_texts(self.video_path, self.roi, self.ocr)
        watermark_texts_list = list(self.watermark_texts)  # list is pickleable

        indices = list(range(0, total, self.frame_step))
        if not indices:
            return []

        N = self.num_workers
        block_size = math.ceil(len(indices) / N)
        
        # FIX: Add overlap between chunks to prevent subtitle loss at boundaries
        # Each chunk processes extra frames from the next chunk's start
        overlap_frames = max(30, int(2.0 * fps / self.frame_step))  # ~2 seconds overlap
        
        chunks = []
        for k in range(N):
            block = indices[k * block_size : (k + 1) * block_size]
            if not block:
                continue
            start_idx = block[0]
            # Extend end_idx by overlap to capture subtitles at chunk boundaries
            end_idx = block[-1] + self.frame_step
            if k < N - 1:  # Not the last chunk
                end_idx = min(end_idx + overlap_frames * self.frame_step, total)
            chunks.append((start_idx, end_idx, self.frame_step))

        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        prog_counter = manager.Value("i", 0)
        total_frames_to_process = sum(
            len(range(s, e, self.frame_step)) for (s, e, _) in chunks
        )

        args_list = []
        for (start_idx, end_idx, step) in chunks:
            args_list.append(
                (
                    self.video_path,
                    self.roi,
                    start_idx,
                    end_idx,
                    step,
                    fps,
                    self.min_conf,
                    self.sim_threshold,
                    self.min_duration,
                    self.grace,
                    self.normalize_simplified,
                    prog_counter,
                    watermark_texts_list,
                )
            )

        segments_all = []
        
        # USE PERSISTENT GLOBAL POOL
        pool = get_process_pool(N, ctx)
        
        # Note: We do NOT use 'with pool as ex:' because that would shutdown the pool!
        # Instead we just use pool.submit
        futs = [pool.submit(_ocr_chunk_worker, a) for a in args_list]

        # FIX: Poll ít hơn để giảm CPU overhead
        while True:
            done = sum(1 for f in futs if f.done())
            try:
                processed = prog_counter.value
            except Exception:
                processed = 0

            if total_frames_to_process > 0:
                pct = int(min(100, processed * 100 / total_frames_to_process))
                self.progress.emit(pct)

            if done == len(futs):
                break
            time.sleep(0.5)  # 500ms

        for i, f in enumerate(futs):
            try:
                segments_all.extend(f.result())
            except Exception as e:
                # BUG FIX: Don't silently swallow errors - log them!
                print(f"❌ Worker {i} FAILED: {e} — Segments from this chunk are LOST!")
                import traceback
                traceback.print_exc()

        self.progress.emit(100)
        return segments_all

    def run(self):
        try:
            if self.num_workers == 1:
                segments = self.run_single()
            else:
                segments = self.run_parallel()

            # FIX: Merge tốt hơn với gap_merge lớn hơn
            # Use larger gap for parallel mode to handle chunk boundary overlaps
            merge_gap = 1.0 if self.num_workers > 1 else 0.5
            merged = merge_and_fix_segments(
                segments, min_duration=self.min_duration, gap_merge=merge_gap
            )

            # === GAP VERIFICATION PASS ===
            # After parallel OCR, re-scan suspicious long gaps (>3s) to catch
            # subtitles missed due to GPU contention in parallel workers
            if self.num_workers > 1 and merged and self.ocr is not None:
                cap_verify = cv2.VideoCapture(self.video_path)
                if cap_verify.isOpened():
                    fps_v = cap_verify.get(cv2.CAP_PROP_FPS) or 30.0
                    W_v = int(cap_verify.get(cv2.CAP_PROP_FRAME_WIDTH))
                    H_v = int(cap_verify.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    xr, yr, wr, hr = self.roi
                    xr = max(0, min(xr, W_v - 1))
                    yr = max(0, min(yr, H_v - 1))
                    wr = max(1, min(wr, W_v - xr))
                    hr = max(1, min(hr, H_v - yr))
                    fallback_box_v = (float(xr), float(yr), float(xr + wr), float(yr + hr))

                    # Build list of gaps to verify
                    gaps_to_check = []
                    # Check gap before first segment
                    if merged[0]["start"] > 3.0:
                        gaps_to_check.append((0.0, merged[0]["start"]))
                    # Check gaps between segments
                    for gi in range(len(merged) - 1):
                        gap_start = merged[gi]["end"]
                        gap_end = merged[gi + 1]["start"]
                        if (gap_end - gap_start) > 3.0:
                            gaps_to_check.append((gap_start, gap_end))

                    if gaps_to_check:
                        print(f"🔍 Gap verification: checking {len(gaps_to_check)} suspicious gaps...")
                        extra_segs = []
                        wm = getattr(self, 'watermark_texts', None)
                        
                        for gap_s, gap_e in gaps_to_check:
                            cur_text_v = ""
                            cur_start_v = None
                            cur_box_v = None
                            
                            start_frame = int(gap_s * fps_v)
                            end_frame = int(gap_e * fps_v)
                            
                            for fidx in range(start_frame, end_frame, self.frame_step):
                                cap_verify.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                                ok_v, frame_v = cap_verify.read()
                                if not ok_v:
                                    continue
                                
                                ts_v = float(fidx) / fps_v
                                crop_v = frame_v[yr:yr + hr, xr:xr + wr]
                                sfx_v = 1.0
                                sfy_v = 1.0
                                if hr < 60 or max(wr, hr) < 1200:
                                    crop_v = cv2.resize(crop_v, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                                    sfx_v = 2.5
                                    sfy_v = 2.5
                                
                                res_v = self.ocr.ocr(crop_v, cls=False)
                                t_v, conf_v, box_v = extract_joined_text_conf_and_box(
                                    res_v, xr, yr, sfx_v, sfy_v, roi_w=wr, watermark_texts=wm
                                )
                                if self.cc and t_v:
                                    t_v = self.cc.convert(t_v)
                                if (not t_v) or float(conf_v) < float(self.min_conf):
                                    t_v = ""
                                    box_v = None
                                
                                t_v = (t_v or "").strip()
                                
                                if t_v and not cur_text_v:
                                    cur_text_v = t_v
                                    cur_start_v = ts_v
                                    cur_box_v = box_v or fallback_box_v
                                elif t_v and cur_text_v:
                                    sim_v = fuzz_ratio(t_v, cur_text_v)
                                    if sim_v >= self.sim_threshold or t_v in cur_text_v or cur_text_v in t_v:
                                        if len(t_v) > len(cur_text_v):
                                            cur_text_v = t_v
                                        cur_box_v = union_box(cur_box_v, box_v)
                                    else:
                                        if ts_v - cur_start_v >= self.min_duration:
                                            extra_segs.append({"start": cur_start_v, "end": ts_v, "text": cur_text_v, "box": cur_box_v or fallback_box_v})
                                        cur_text_v = t_v
                                        cur_start_v = ts_v
                                        cur_box_v = box_v or fallback_box_v
                                elif not t_v and cur_text_v:
                                    end_t_v = ts_v
                                    if end_t_v - cur_start_v >= self.min_duration:
                                        extra_segs.append({"start": cur_start_v, "end": end_t_v, "text": cur_text_v, "box": cur_box_v or fallback_box_v})
                                    cur_text_v = ""
                                    cur_start_v = None
                                    cur_box_v = None
                            
                            # Flush remaining
                            if cur_text_v and cur_start_v is not None:
                                end_t_v = float(end_frame) / fps_v
                                if end_t_v - cur_start_v >= self.min_duration:
                                    extra_segs.append({"start": cur_start_v, "end": end_t_v, "text": cur_text_v, "box": cur_box_v or fallback_box_v})
                        
                        if extra_segs:
                            print(f"✅ Gap verification found {len(extra_segs)} missed subtitles!")
                            for es in extra_segs:
                                print(f"   + [{es['start']:.1f}s-{es['end']:.1f}s] {es['text'][:50]}")
                            segments.extend(extra_segs)
                            merged = merge_and_fix_segments(
                                segments, min_duration=self.min_duration, gap_merge=merge_gap
                            )
                    
                    cap_verify.release()

            base = os.path.splitext(os.path.basename(self.video_path))[0]
            out_path = (
                self.fixed_out_path
                or os.path.join(os.path.dirname(self.video_path), base + ".srt")
            )
            build_srt(merged, out_path)
            self.finished.emit(out_path, merged)

        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")



class InpaintWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    log = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, input_video, output_video, segments, parent=None):
        super().__init__(parent)
        self.input_video = input_video
        self.output_video = output_video
        self.segments = segments

    def run(self):
        try:
            self.log.emit("🧼 RemoveSub: start inpainting...")

            # Safety check: warn if subtitle coverage is very high (likely logo/watermark)
            if self.segments:
                total_sub_time = sum(s.get("end", 0) - s.get("start", 0) for s in self.segments)
                max_end = max(s.get("end", 0) for s in self.segments)
                if max_end > 0:
                    coverage = total_sub_time / max_end
                    if coverage > 0.8:
                        self.log.emit(f"⚠️ High subtitle coverage ({coverage*100:.0f}%) - may be logo/watermark. Using chunked processing.")

            def _pcb(p): self.progress.emit(int(p))

            remove_subtitle_video(
                input_video=self.input_video,
                output_video=self.output_video,
                segments=self.segments,
                method="sttn_inpaint",
                inpaint_radius=3,
                box_pad_px=20,
                mask_dilate_px=18,
                horizontal_coverage=0.90,
                ffmpeg_path=FFMPEG,
                progress_cb=_pcb
            )
            self.log.emit("✅ RemoveSub: done")
            self.finished.emit(self.output_video)
        except MemoryError:
            import gc
            gc.collect()
            self.error.emit("❌ Out of Memory! Video quá lớn hoặc subtitle coverage quá cao (có thể là logo). "
                          "Hãy thử chuyển sang FFmpeg Delogo trong Settings.")
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")






class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle OCR — PaddleOCR v5 (Qt) - FULL AUTO")
        self.resize(1200, 760)

        self.video_path = ""
        self.cap = None
        self.frame = None
        self.W = 0
        self.H = 0

        # self.ocr = PaddleOCR(
        #     text_recognition_model_name="PP-OCRv5_server_rec",
        #     use_doc_orientation_classify=False,
        #     use_doc_unwarping=False,
        #     use_textline_orientation=True,
        # )
        self.ocr = None

        
        self.cc = OpenCC("t2s") if OPENCC_OK else None

        # state cho full auto
        self._auto_final_video = None
        self._auto_margin_v = 60
        self._auto_zh_srt = None
        self._auto_vi_srt = None
        self._auto_tts_speed = 1.2

        self.build_ui()
        self.start_util_timer()
        self._auto_clean_video = None
        self._auto_ocr_segments = None


    def set_step_state(self, state: str):
        """
        Quản lý enable/disable theo bước để tránh bấm nhầm.
        state:
        - "IDLE"              : rảnh
        - "OCR_ONLY"          : đang OCR tách SRT
        - "OCR_INPAINT"       : đang OCR + remove sub
        - "TRANSLATE"         : đang dịch
        - "MERGE"             : đang ghép TTS + burn
        - "AUTO"              : đang auto all
        """
        has_video = bool(self.video_path)

        # các nút luôn phụ thuộc "có video" (trừ Dịch manual & Merge-only có file dialog riêng)
        self.btnTest.setEnabled(has_video and state == "IDLE")
        self.btnRun.setEnabled(has_video and state == "IDLE")
        self.btnOcrInpaint.setEnabled(has_video and state == "IDLE")
        self.btnFullAuto.setEnabled(has_video and state == "IDLE")

        # dịch manual / merge-only: chỉ cho bấm khi IDLE (để khỏi chạy chồng)
        self.btnTrans.setEnabled(state == "IDLE")
        self.btnMergeOnly.setEnabled(state == "IDLE")

        # stop chỉ bật khi đang chạy
        self.btnStop.setEnabled(state in ("OCR_ONLY", "OCR_INPAINT", "AUTO", "MERGE", "TRANSLATE"))


    def set_busy(self, busy: bool, mode: str = "IDLE", indeterminate: bool = False):
        """
        busy True: khóa UI theo mode
        busy False: về IDLE
        indeterminate: progress bar quay vô hạn (0,0)
        """
        if busy:
            self.set_step_state(mode)
            if indeterminate:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 100)
                self.progress.setValue(0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.set_step_state("IDLE")


    
    def build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.btnOpen = QtWidgets.QPushButton("Select Video")
        self.lblVideo = QtWidgets.QLabel("No video selected")
        self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setEnabled(False)
        self.frameSlider.valueChanged.connect(self.on_seek)
        top.addWidget(self.btnOpen)
        top.addWidget(self.lblVideo, 1)
        top.addWidget(self.frameSlider, 2)
        layout.addLayout(top)

        body = QtWidgets.QHBoxLayout()
        self.view = ImageLabel()
        self.view.setMinimumSize(720, 400)
        self.view.roi_changed.connect(self.on_roi_drawn)  # Connect mouse ROI selection
        body.addWidget(self.view, 3)

        right = QtWidgets.QVBoxLayout()
        roi_box = QtWidgets.QGroupBox("ROI (X, Y, Width, Height)")
        grid = QtWidgets.QGridLayout()
        self.sldX = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldY = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldW = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldH = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spX = QtWidgets.QSpinBox()
        self.spY = QtWidgets.QSpinBox()
        self.spW = QtWidgets.QSpinBox()
        self.spH = QtWidgets.QSpinBox()

        for s in (self.sldX, self.sldY, self.sldW, self.sldH):
            s.setEnabled(False)
        for sp in (self.spX, self.spY, self.spW, self.spH):
            sp.setEnabled(False)

        self.sldX.valueChanged.connect(self.spX.setValue)
        self.spX.valueChanged.connect(self.sldX.setValue)
        self.sldY.valueChanged.connect(self.spY.setValue)
        self.spY.valueChanged.connect(self.sldY.setValue)
        self.sldW.valueChanged.connect(self.spW.setValue)
        self.spW.valueChanged.connect(self.sldW.setValue)
        self.sldH.valueChanged.connect(self.spH.setValue)
        self.spH.valueChanged.connect(self.sldH.setValue)

        grid.addWidget(QtWidgets.QLabel("X"), 0, 0)
        grid.addWidget(self.sldX, 0, 1)
        grid.addWidget(self.spX, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Y"), 1, 0)
        grid.addWidget(self.sldY, 1, 1)
        grid.addWidget(self.spY, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Width"), 2, 0)
        grid.addWidget(self.sldW, 2, 1)
        grid.addWidget(self.spW, 2, 2)
        grid.addWidget(QtWidgets.QLabel("Height"), 3, 0)
        grid.addWidget(self.sldH, 3, 1)
        grid.addWidget(self.spH, 3, 2)
        roi_box.setLayout(grid)

        set_box = QtWidgets.QGroupBox("Settings")
        form = QtWidgets.QFormLayout()

        self.spinEvery = QtWidgets.QSpinBox()
        self.spinEvery.setRange(1, 30)
        self.spinEvery.setValue(2)

        self.dblConf = QtWidgets.QDoubleSpinBox()
        self.dblConf.setRange(0, 1.0)
        self.dblConf.setSingleStep(0.05)
        self.dblConf.setValue(0.5)

        self.spinSim = QtWidgets.QSpinBox()
        self.spinSim.setRange(0, 100)
        self.spinSim.setValue(50)

        self.dblMinDur = QtWidgets.QDoubleSpinBox()
        self.dblMinDur.setRange(0.1, 10.0)
        self.dblMinDur.setSingleStep(0.1)
        self.dblMinDur.setValue(0.6)

        self.dblGrace = QtWidgets.QDoubleSpinBox()
        self.dblGrace.setRange(0.0, 2.0)
        self.dblGrace.setSingleStep(0.05)
        self.dblGrace.setValue(0.0)

        self.spinWorkers = QtWidgets.QSpinBox()
        self.spinWorkers.setRange(1, 4)
        self.spinWorkers.setValue(2)

        self.dblTtsSpeed = QtWidgets.QDoubleSpinBox()
        self.dblTtsSpeed.setRange(0.5, 3.0)
        self.dblTtsSpeed.setSingleStep(0.1)
        self.dblTtsSpeed.setValue(1.0)

        self.chkSimplified = QtWidgets.QCheckBox("Normalize to Simplified (OpenCC t2s)")
        self.chkSimplified.setChecked(True if OPENCC_OK else False)
        if not OPENCC_OK:
            self.chkSimplified.setEnabled(False)
            self.chkSimplified.setToolTip("Install opencc-python-reimplemented to enable")

        self.chkLocalTrans = QtWidgets.QCheckBox("Dùng model dịch OFFLINE (zh→vi)")
        self.chkLocalTrans.setChecked(True)

        form.addRow("Every N frames:", self.spinEvery)
        form.addRow("Min confidence:", self.dblConf)
        form.addRow("Similarity (merge):", self.spinSim)
        form.addRow("Min duration (s):", self.dblMinDur)
        form.addRow("Grace (s):", self.dblGrace)
        form.addRow("Workers (processes):", self.spinWorkers)
        form.addRow("TTS speed:", self.dblTtsSpeed)
        form.addRow(self.chkSimplified)
        form.addRow(self.chkLocalTrans)
        set_box.setLayout(form)

        # ===== Buttons (chia step) =====
        self.btnTest = QtWidgets.QPushButton("Test OCR on current frame")

        self.btnRun = QtWidgets.QPushButton("1) Tách SRT (OCR → Save .srt)")
        self.btnOcrInpaint = QtWidgets.QPushButton("2) Tách SRT + RemoveSub (OCR → Inpaint)")

        self.btnTrans = QtWidgets.QPushButton("3) Dịch (SRT zh → vi)")
        self.btnMergeOnly = QtWidgets.QPushButton("4) Ghép SRT + Audio → Video (chọn video + srt)")

        self.btnFullAuto = QtWidgets.QPushButton("5) Auto ALL: OCR → RemoveSub → Dịch → Render")

        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        right.addWidget(roi_box)
        right.addWidget(set_box)
        right.addWidget(self.btnTest)
        right.addWidget(self.btnRun)
        right.addWidget(self.btnOcrInpaint)
        right.addWidget(self.btnStop)
        right.addWidget(self.progress)
        right.addWidget(self.btnTrans)
        right.addWidget(self.btnMergeOnly)
        right.addWidget(self.btnFullAuto)
        right.addStretch(1)

        body.addLayout(right, 2)
        layout.addLayout(body)

        self.utilLabel = QtWidgets.QLabel("CPU --% | GPU --%")
        self.utilLabel.setStyleSheet("color:#1976d2; font-weight:bold; padding:4px;")
        layout.addWidget(self.utilLabel, alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)

        # ===== Connect =====
        self.btnOpen.clicked.connect(self.on_open)
        self.btnTest.clicked.connect(self.on_test)

        self.btnRun.clicked.connect(self.on_run)
        self.btnOcrInpaint.clicked.connect(self.on_ocr_and_inpaint_click)

        self.btnTrans.clicked.connect(self.on_translate_click)
        self.btnMergeOnly.clicked.connect(self.on_merge_only_click)

        self.btnFullAuto.clicked.connect(self.on_full_auto)
        self.btnStop.clicked.connect(self.on_stop)

        self.sldX.valueChanged.connect(self.update_preview)
        self.sldY.valueChanged.connect(self.update_preview)
        self.sldW.valueChanged.connect(self.update_preview)
        self.sldH.valueChanged.connect(self.update_preview)

        self.setStyleSheet(
            """
            QLabel{ font-size: 12px; }
            QPushButton{ padding:8px; }
            QGroupBox{ font-weight: bold; }
            """
        )

        # trạng thái ban đầu
        self.set_step_state("IDLE")

    def on_ocr_and_inpaint_click(self):
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "Info", "Chọn video trước.")
            return

        r = self.current_roi_rect()
        if r.width() <= 0 or r.height() <= 0:
            QtWidgets.QMessageBox.information(self, "Info", "ROI chưa hợp lệ.")
            return

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        default_srt = os.path.join(os.path.dirname(self.video_path), base + ".srt")
        zh_srt_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu SRT tiếng Trung (OCR)", default_srt, "SubRip (*.srt)"
        )
        if not zh_srt_path:
            return

        base_no_ext = os.path.splitext(self.video_path)[0]
        self._auto_clean_video = base_no_ext + "_nosub.mp4"

        self.set_busy(True, mode="OCR_INPAINT", indeterminate=False)

        self.ocr_auto_worker = OCRWorker(
            video_path=self.video_path,
            roi=(r.x(), r.y(), r.width(), r.height()),
            frame_step=self.spinEvery.value(),
            min_conf=self.dblConf.value(),
            sim_threshold=self.spinSim.value(),
            min_duration=self.dblMinDur.value(),
            grace_sec=self.dblGrace.value(),
            normalize_simplified=self.chkSimplified.isChecked(),
            num_workers=self.spinWorkers.value(),
            fixed_out_path=zh_srt_path,
        )
        self.worker = self.ocr_auto_worker

        self.ocr_auto_worker.progress.connect(self.progress.setValue)
        self.ocr_auto_worker.error.connect(self.on_error)

        def _after_ocr(zh_path, segments):
            self._auto_zh_srt = zh_path
            self._auto_ocr_segments = segments

            self.progress.setRange(0, 100)
            self.progress.setValue(0)

            self.inpaint_worker_auto = InpaintWorker(
                input_video=self.video_path,
                output_video=self._auto_clean_video,
                segments=self._auto_ocr_segments,
                parent=self
            )
            self.worker = self.inpaint_worker_auto

            self.inpaint_worker_auto.progress.connect(self.progress.setValue)
            self.inpaint_worker_auto.error.connect(self.on_error)

            def _after_inpaint(out_clean):
                self.progress.setValue(100)
                self.set_busy(False)
                QtWidgets.QMessageBox.information(
                    self, "Xong", f"Đã OCR + RemoveSub:\nSRT: {zh_path}\nVideo: {out_clean}"
                )

            self.inpaint_worker_auto.finished.connect(_after_inpaint)
            self.inpaint_worker_auto.start()

        self.ocr_auto_worker.finished.connect(_after_ocr)
        self.ocr_auto_worker.start()

    def on_merge_only_click(self):
        video, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn video để ghép", "", "Videos (*.mp4 *.mkv *.avi *.mov)"
        )
        if not video:
            return

        srt, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn SRT (tiếng Việt)", "", "SubRip Subtitle (*.srt)"
        )
        if not srt:
            return

        default_out = os.path.splitext(video)[0] + "-vi.mp4"
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu video output", default_out, "MP4 (*.mp4)"
        )
        if not out:
            return
        if not out.lower().endswith(".mp4"):
            out += ".mp4"

        self.set_busy(True, mode="MERGE", indeterminate=True)

        # Nếu muốn dùng ROI hiện tại để đặt vùng sub (khi đang mở 1 video trong UI)
        roi_top = 0
        roi_bottom = 0
        try:
            r = self.current_roi_rect()
            roi_top = r.y()
            roi_bottom = r.y() + r.height()
        except Exception:
            pass

        self.merge_worker_auto = SrtToVideoMergeWorker(
            video=video,
            srt=srt,
            out=out,
            speed=float(self.dblTtsSpeed.value()),
            margin_v=60,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
        )
        self.worker = self.merge_worker_auto

        self.merge_worker_auto.log.connect(self.on_merge_log_auto)
        self.merge_worker_auto.error.connect(self.on_error)

        def _done(out_path):
            self.set_busy(False)
            self.progress.setValue(100)
            QtWidgets.QMessageBox.information(self, "Xong", f"Đã render:\n{out_path}")

        self.merge_worker_auto.finished.connect(_done)
        self.merge_worker_auto.start()


    # ============ TRANS MANUAL (nút Dùng AI dịch) ============
    def on_translate_click(self):
        in_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn file SRT (tiếng Trung)", "", "SubRip Subtitle (*.srt)"
        )
        if not in_path:
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Lưu file SRT đã dịch",
            in_path.replace(".srt", "-vi.srt"),
            "SubRip Subtitle (*.srt)",
        )
        if not out_path:
            return

        self.btnTrans.setEnabled(False)
        self.progress.setRange(0, 0)  # indeterminate

        # self.thread = QtCore.QThread(self)
        # self.worker_trans = SrtTranslateWorker(in_path, out_path, target_lang="vi")
        # self.worker_trans.moveToThread(self.thread)
        self.thread = QtCore.QThread(self)

        # chọn worker theo checkbox
        if self.chkLocalTrans.isChecked():
            from trans_local import LocalSrtWorker  # Lazy import
            worker_cls = LocalSrtWorker      # OFFLINE
        else:
            worker_cls = GeminiSrtWorker     # GEMINI

        self.worker_trans = worker_cls(in_path, out_path, target_lang="vi")
        self.worker_trans.moveToThread(self.thread)


        self.thread.started.connect(self.worker_trans.run)
        self.worker_trans.finished.connect(self.on_translated)
        self.worker_trans.error.connect(self.on_error)
        self.worker_trans.progress.connect(self.progress.setValue)

        self.worker_trans.finished.connect(self.thread.quit)
        self.worker_trans.finished.connect(self.worker_trans.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_translated(self, out_path: str):
        self.btnTrans.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        QtWidgets.QMessageBox.information(
            self, "Xong rồi", f"Đã dịch xong:\n{out_path}"
        )

    # ================== SYSTEM UTIL HUD ==================
    def start_util_timer(self):
        self.utilTimer = QtCore.QTimer(self)
        self.utilTimer.setInterval(1000)
        self.utilTimer.timeout.connect(self.update_util_label)
        self.utilTimer.start()

    def update_util_label(self):
        cpu = psutil.cpu_percent(interval=None)

        gpu_txt = "--"
        gpu_err = ""

        if _NVML_OK:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                gpu_txt = str(util.gpu)
            except Exception as e:
                gpu_txt = "--"
                gpu_err = f" (NVML err: {type(e).__name__}: {e})"
        else:
            gpu_err = " (pynvml not available / nvmlInit failed)"

        self.utilLabel.setText(f"CPU {int(cpu)}% | GPU {gpu_txt}%{gpu_err}")


    # ================== COMMON UTILS ==================
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", "", "Videos (*.mp4 *.mkv *.avi *.mov)"
        )
        if not path:
            return
        self.video_path = path
        self.lblVideo.setText(path)

        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open video")
            self.video_path = ""
            self.set_step_state("IDLE")
            return

        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameSlider.setEnabled(True)
        self.frameSlider.setRange(0, max(0, total - 1))
        self.frameSlider.setValue(0)

        for w in (self.sldX, self.sldY, self.sldW, self.sldH, self.spX, self.spY, self.spW, self.spH):
            w.setEnabled(True)

        self.sldX.setRange(0, self.W - 1)
        self.spX.setRange(0, self.W - 1)
        self.sldY.setRange(0, self.H - 1)
        self.spY.setRange(0, self.H - 1)
        self.sldW.setRange(1, self.W)
        self.spW.setRange(1, self.W)
        self.sldH.setRange(1, self.H)
        self.spH.setRange(1, self.H)

        # Set default ROI first (instant, no freeze)
        x, y = 0, int(self.H * 0.7)
        w, h = self.W, self.H - y
        self.sldX.setValue(x)
        self.sldY.setValue(y)
        self.sldW.setValue(w)
        self.sldH.setValue(h)

        # Auto-detect in background (non-blocking)
        def _detect_roi_bg():
            try:
                print("🔍 Auto-detecting subtitle ROI (background)...")
                detected_roi = smart_detect_subtitle_roi(path)
                
                if detected_roi:
                    x, y, w, h = detected_roi
                    print(f"✅ Auto-detected ROI: x={x}, y={y}, w={w}, h={h}")
                    # Update UI in main thread
                    self.sldX.setValue(x)
                    self.sldY.setValue(y)
                    self.sldW.setValue(w)
                    self.sldH.setValue(h)
                else:
                    print(f"⚠️  Auto-detect failed, keeping estimate")
            except Exception as e:
                print(f"❌ Auto-detect error: {e}")
                import traceback
                traceback.print_exc()
        
        # Run in thread (DISABLED - causes crash)
        # import threading
        # threading.Thread(target=_detect_roi_bg, daemon=True).start()
        self.update_preview()
        self.set_step_state("IDLE")


    def on_roi_drawn(self, x, y, w, h):
        """Handle ROI drawn by mouse"""
        # Update sliders
        self.sldX.setValue(x)
        self.sldY.setValue(y)
        self.sldW.setValue(w)
        self.sldH.setValue(h)
        # Update preview
        self.update_preview()

    def on_seek(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
            self.update_preview()

    def current_roi_rect(self):
        return QtCore.QRect(
            self.sldX.value(),
            self.sldY.value(),
            self.sldW.value(),
            self.sldH.value(),
        )

    def update_preview(self):
        if self.frame is None and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameSlider.value())
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
        if self.frame is not None:
            self.view.set_frame(self.frame, self.current_roi_rect())

    def on_test(self):
        if self.ocr is None:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )

        if self.frame is None:
            QtWidgets.QMessageBox.information(
                self, "Info", "Load video & chọn frame trước."
            )
            return
        r = self.current_roi_rect()
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        if w <= 0 or h <= 0:
            QtWidgets.QMessageBox.information(
                self, "Info", "ROI chưa hợp lệ."
            )
            return

        crop = self.frame[y : y + h, x : x + w].copy()
        if max(w, h) < 1200:
            crop = cv2.resize(
                crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )

        res = self.ocr.ocr(crop, cls=False)
        texts, scores, yboxes = safe_extract_texts_scores_boxes(res)
        if not texts:
            QtWidgets.QMessageBox.information(
                self, "OCR Test", "Không nhận được text trong ROI này."
            )
            return
        if (
            yboxes
            and len(yboxes) == len(texts)
            and any(y1 != 0.0 or y2 != 0.0 for (y1, y2) in yboxes)
        ):
            order = np.argsort([yb[0] for yb in yboxes])
            ordered = [
                texts[i].strip()
                for i in order
                if texts[i] and texts[i].strip()
            ]
        else:
            ordered = [t.strip() for t in texts if t and t.strip()]
        text = "  ".join(ordered)
        conf = max(scores) if scores else 1.0
        if self.chkSimplified.isChecked() and OPENCC_OK:
            text = OpenCC("t2s").convert(text)
        QtWidgets.QMessageBox.information(
            self, "OCR Test", f"Text: {text}\nConf: {conf:.3f}"
        )

    # ================ MANUAL OCR → SAVE SRT =================
    def on_run(self):
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "Info", "Chọn video trước.")
            return
        r = self.current_roi_rect()
        if r.width() <= 0 or r.height() <= 0:
            QtWidgets.QMessageBox.information(self, "Info", "ROI chưa hợp lệ.")
            return

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        default = os.path.join(os.path.dirname(self.video_path), base + ".srt")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save SRT As", default, "SubRip (*.srt)"
        )
        if not out_path:
            return

        self.set_busy(True, mode="OCR_ONLY", indeterminate=False)

        self.worker = OCRWorker(
            video_path=self.video_path,
            roi=(r.x(), r.y(), r.width(), r.height()),
            frame_step=self.spinEvery.value(),
            min_conf=self.dblConf.value(),
            sim_threshold=self.spinSim.value(),
            min_duration=self.dblMinDur.value(),
            grace_sec=self.dblGrace.value(),
            normalize_simplified=self.chkSimplified.isChecked(),
            num_workers=self.spinWorkers.value(),
            fixed_out_path=out_path,
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.error.connect(self.on_error)

        def _done(srt_path, segments):
            self.progress.setValue(100)
            self.set_busy(False)
            QtWidgets.QMessageBox.information(
                self, "Finished", f"Done!\nSaved SRT:\n{srt_path}\nSegments: {len(segments)}"
            )

        self.worker.finished.connect(_done)
        self.worker.start()


    def on_stop(self):
        # OCRWorker có stop()
        if hasattr(self, "worker") and self.worker is not None:
            try:
                if hasattr(self.worker, "stop"):
                    self.worker.stop()
            except Exception:
                pass
            try:
                if hasattr(self.worker, "isRunning") and self.worker.isRunning():
                    self.worker.wait()
            except Exception:
                pass

        self.set_busy(False)


    def on_finished(self, srt_path, segments):
        self.btnRun.setEnabled(True)
        self.btnStop.setEnabled(False)
        QtWidgets.QMessageBox.information(
            self,
            "Finished",
            f"Done!\nSaved SRT:\n{srt_path}\nSegments: {len(segments)}",
        )

    # ================ FULL AUTO PIPELINE =================
    def compute_margin_v(self) -> int:
        """
        Giữ lại cho đủ code cũ (giờ gần như không dùng nữa).
        """
        r = self.current_roi_rect()
        y = int(r.y())
        return max(0, y + 10)

    def on_full_auto(self):
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "Info", "Chọn video trước.")
            return

        r = self.current_roi_rect()
        if r.width() <= 0 or r.height() <= 0:
            QtWidgets.QMessageBox.information(self, "Info", "ROI chưa hợp lệ.")
            return

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        default_out = os.path.join(os.path.dirname(self.video_path), base + "-vi.mp4")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Chọn nơi lưu video tiếng Việt", default_out, "MP4 (*.mp4)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        self._auto_final_video = out_path
        self._auto_margin_v = 60
        self._auto_tts_speed = float(self.dblTtsSpeed.value())

        base_no_ext = os.path.splitext(self.video_path)[0]
        self._auto_zh_srt = base_no_ext + ".srt"
        self._auto_vi_srt = base_no_ext + "-vi.srt"

        self.set_busy(True, mode="AUTO", indeterminate=False)

        self.ocr_auto_worker = OCRWorker(
            video_path=self.video_path,
            roi=(r.x(), r.y(), r.width(), r.height()),
            frame_step=self.spinEvery.value(),
            min_conf=self.dblConf.value(),
            sim_threshold=self.spinSim.value(),
            min_duration=self.dblMinDur.value(),
            grace_sec=self.dblGrace.value(),
            normalize_simplified=self.chkSimplified.isChecked(),
            num_workers=self.spinWorkers.value(),
            fixed_out_path=self._auto_zh_srt,
        )
        self.worker = self.ocr_auto_worker

        self.ocr_auto_worker.progress.connect(self.progress.setValue)
        self.ocr_auto_worker.finished.connect(self.on_ocr_finished_auto)
        self.ocr_auto_worker.error.connect(self.on_error)
        self.ocr_auto_worker.start()


    def on_ocr_finished_auto(self, zh_srt_path, segments):
        self._auto_zh_srt = zh_srt_path
        self._auto_ocr_segments = segments

        base_no_ext = os.path.splitext(self.video_path)[0]
        self._auto_clean_video = base_no_ext + "_nosub.mp4"

        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.inpaint_worker_auto = InpaintWorker(
            input_video=self.video_path,
            output_video=self._auto_clean_video,
            segments=self._auto_ocr_segments,
            parent=self
        )
        self.worker = self.inpaint_worker_auto

        self.inpaint_worker_auto.progress.connect(self.progress.setValue)
        self.inpaint_worker_auto.finished.connect(self.on_inpaint_finished_auto)
        self.inpaint_worker_auto.error.connect(self.on_error)
        self.inpaint_worker_auto.start()


    def on_inpaint_finished_auto(self, clean_video_path: str):
        self._auto_clean_video = clean_video_path

        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.trans_thread_auto = QtCore.QThread(self)

        if self.chkLocalTrans.isChecked():
            from trans_local import LocalSrtWorker  # Lazy import
            worker_cls = LocalSrtWorker
        else:
            worker_cls = GeminiSrtWorker
        self.trans_worker_auto = worker_cls(self._auto_zh_srt, self._auto_vi_srt, target_lang="vi")
        self.trans_worker_auto.moveToThread(self.trans_thread_auto)

        self.trans_thread_auto.started.connect(self.trans_worker_auto.run)
        self.trans_worker_auto.progress.connect(self.progress.setValue)
        self.trans_worker_auto.finished.connect(self.on_translate_finished_auto)
        self.trans_worker_auto.error.connect(self.on_error)

        self.trans_worker_auto.finished.connect(self.trans_thread_auto.quit)
        self.trans_worker_auto.finished.connect(self.trans_worker_auto.deleteLater)
        self.trans_thread_auto.finished.connect(self.trans_thread_auto.deleteLater)

        self.trans_thread_auto.start()



    def on_translate_finished_auto(self, vi_srt_path):
        self._auto_vi_srt = vi_srt_path

        # render lâu -> indeterminate
        self.progress.setRange(0, 0)

        r = self.current_roi_rect()
        roi_top = r.y()
        roi_bottom = r.y() + r.height()

        src_video_for_render = self._auto_clean_video or self.video_path

        self.merge_worker_auto = SrtToVideoMergeWorker(
            video=src_video_for_render,
            srt=self._auto_vi_srt,
            out=self._auto_final_video,
            speed=self._auto_tts_speed,
            margin_v=self._auto_margin_v,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
        )
        self.worker = self.merge_worker_auto

        self.merge_worker_auto.log.connect(self.on_merge_log_auto)
        self.merge_worker_auto.finished.connect(self.on_merge_finished_auto)
        self.merge_worker_auto.error.connect(self.on_error)
        self.merge_worker_auto.start()


    def on_merge_log_auto(self, msg: str):
        # hiện tại chỉ print ra console, nếu muốn có box log thì thêm UI.
        print(msg)

    def on_merge_finished_auto(self, out_path: str):
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.set_busy(False)
        QtWidgets.QMessageBox.information(self, "Hoàn tất", f"Đã xuất video tiếng Việt:\n{out_path}")

    # =================== ERROR CHUNG ====================
    def on_error(self, msg: str):
        self.set_busy(False)
        QtWidgets.QMessageBox.critical(self, "Lỗi", msg)


def main():
    if sys.platform.startswith("win"):
        mp.freeze_support()

    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
