#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vse_runner.py - Single-Thread PaddleOCR Subtitle Extractor

Replaces YaoFANGUK's broken multiprocessing pipeline with a clean
single-thread sequential extraction:
  1. Sample every Nth frame from the video
  2. OCR each frame with a single PaddleOCR instance (GPU)
  3. Deduplicate consecutive identical subtitles
  4. Write SRT file

This avoids ALL concurrent GPU contexts = zero CUDA segfaults.
"""
import sys
import os
import argparse
import cv2
import time
from pathlib import Path
from difflib import SequenceMatcher


def process_vse(video_path, roi_ymin, roi_ymax, roi_xmin, roi_xmax, mode="fast"):
    """
    Extract subtitles from video using single-thread PaddleOCR.
    
    Mode determines sampling rate:
      fast:     every 1 second (skip frames)
      auto:     every 0.5 seconds
      accurate: every 0.33 seconds (3 samples/sec)
    """
    # Video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        sys.exit(1)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    duration = frame_count / fps

    print(f"VSE_RUNNER_INIT: {video_path}")
    print(f"VSE_RUNNER_ROI: ymin={roi_ymin}, ymax={roi_ymax}, xmin={roi_xmin}, xmax={roi_xmax}")
    print(f"VSE_RUNNER_VIDEO: {frame_count} frames, {fps:.1f} fps, {frame_width}x{frame_height}, {duration:.1f}s")

    # Sampling interval based on mode
    if mode == "accurate":
        frame_step = 1  # 🚨 QUAN TRỌNG: Quét TỪNG KHUNG HÌNH (Every single frame) để ép bắt text nháy 1-2 frames
        sample_interval = 1.0 / fps
    elif mode == "auto":
        sample_interval = 0.33  # ~3 samples per second
        frame_step = max(1, int(fps * sample_interval))
    else:  # fast
        sample_interval = 1.0   # 1 sample per second
        frame_step = max(1, int(fps * sample_interval))
        
    total_samples = frame_count // frame_step
    print(f"MODE: {mode}, sampling every {frame_step} frames ({sample_interval}s interval), ~{total_samples} samples")

    # ======== Initialize RapidOCR (ONNX) ========
    print("PHASE1_START: Initializing RapidOCR (ONNX Runtime)...")
    
    from rapidocr_onnxruntime import RapidOCR
    
    ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    print("RapidOCR initialized")

    # ======== Frame-by-frame OCR ========
    print("PHASE2_START: Extracting subtitles frame-by-frame...")
    
    sub_area = (roi_ymin, roi_ymax, roi_xmin, roi_xmax)
    raw_entries = []  # list of {"time_s": float, "text": str, "box": (x1,y1,x2,y2) or None}
    
    current_frame = 0
    samples_done = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame += 1
        
        # Skip frames based on sampling rate
        if (current_frame - 1) % frame_step != 0:
            continue
        
        samples_done += 1
        time_s = (current_frame - 1) / fps
        
        # Progress reporting
        pct = int((current_frame / frame_count) * 100)
        if samples_done % 10 == 0:
            print(f"PROGRESS: {pct}%")
        
        # Run OCR on this frame
        try:
            ocr_result, _ = ocr(frame)
        except Exception as e:
            continue
        
        if not ocr_result:
            raw_entries.append({"time_s": time_s, "text": "", "box": None})
            continue
        
        # Filter detections by ROI and collect bounding boxes
        texts = []
        union_box = None  # (x1, y1, x2, y2) tight around all detected text lines
        
        for line in ocr_result:
            poly = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = line[1]
            confidence = line[2]
            
            if confidence < 0.5:
                continue
            
            # Check if box center is inside ROI
            cy = (poly[0][1] + poly[2][1]) / 2
            cx = (poly[0][0] + poly[2][0]) / 2
            
            if roi_ymin <= cy <= roi_ymax and roi_xmin <= cx <= roi_xmax:
                texts.append(text)
                
                # Extract tight bounding box from polygon
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
                
                # Union with existing box
                if union_box is None:
                    union_box = (bx1, by1, bx2, by2)
                else:
                    union_box = (
                        min(union_box[0], bx1),
                        min(union_box[1], by1),
                        max(union_box[2], bx2),
                        max(union_box[3], by2),
                    )
        
        combined = " ".join(texts) if texts else ""
        raw_entries.append({"time_s": time_s, "text": combined, "box": union_box})
    
    cap.release()
    print(f"PHASE2_DONE: Processed {samples_done} frames, {len(raw_entries)} entries")

    # ======== Merge consecutive identical subtitles ========
    print("PHASE3_START: Merging consecutive identical subtitles...")
    
    merged = []
    current_sub = None
    
    for entry in raw_entries:
        text = entry["text"].strip()
        time_s = entry["time_s"]
        entry_box = entry.get("box")
        
        if not text:
            # No subtitle in this frame
            if current_sub is not None:
                # End current subtitle
                current_sub["end_s"] = time_s
                merged.append(current_sub)
                current_sub = None
            continue
        
        if current_sub is None:
            # Start new subtitle
            current_sub = {
                "start_s": time_s,
                "end_s": time_s + sample_interval,
                "text": text,
                "box": entry_box,
            }
        elif _texts_similar(current_sub["text"], text):
            # Extend current subtitle (same text continues)
            current_sub["end_s"] = time_s + sample_interval
            # Union the boxes
            if entry_box and current_sub.get("box"):
                ob = current_sub["box"]
                current_sub["box"] = (
                    min(ob[0], entry_box[0]),
                    min(ob[1], entry_box[1]),
                    max(ob[2], entry_box[2]),
                    max(ob[3], entry_box[3]),
                )
            elif entry_box:
                current_sub["box"] = entry_box
        else:
            # Different subtitle - save previous, start new
            current_sub["end_s"] = time_s
            merged.append(current_sub)
            current_sub = {
                "start_s": time_s,
                "end_s": time_s + sample_interval,
                "text": text,
                "box": entry_box,
            }
    
    # Don't forget the last subtitle
    if current_sub is not None:
        merged.append(current_sub)
    
    # Bỏ hoàn toàn việc lọc Sub ngắn (Cho phép cả những chữ chớp nháy 1 khung hình 0.033 giây lọt qua màn lưới)
    merged = [s for s in merged if (s["end_s"] - s["start_s"]) > 0.0]
    
    print(f"PHASE3_DONE: {len(merged)} unique subtitles")

    # ======== Write SRT ========
    output_srt = os.path.splitext(video_path)[0] + ".srt"
    print(f"PHASE4_START: Writing SRT to {output_srt}")
    
    with open(output_srt, 'w', encoding='utf-8') as f:
        for idx, entry in enumerate(merged, 1):
            start_str = _seconds_to_srt_time(entry["start_s"])
            end_str = _seconds_to_srt_time(entry["end_s"])
            f.write(f"{idx}\n{start_str} --> {end_str}\n{entry['text']}\n\n")

    # ======== Write segments.json for InpaintWorker ========
    import json
    segments_path = output_srt.replace(".srt", ".segments.json")
    segments_json = []
    for entry in merged:
        seg = {
            "start": entry["start_s"],
            "end": entry["end_s"],
            "text": entry["text"],
            "box": list(entry["box"]) if entry.get("box") else None,
            "stable_box": list(entry["box"]) if entry.get("box") else None,
        }
        segments_json.append(seg)
    
    with open(segments_path, 'w', encoding='utf-8') as f:
        json.dump(segments_json, f, ensure_ascii=False, indent=2)
    print(f"SEGMENTS_JSON: {segments_path}")
    
    print(f"OUTPUT_SRT: {output_srt}")
    print(f"TOTAL_SUBS: {len(merged)}")
    print("PROGRESS: 100%")
    print("VSE_RUNNER_DONE")


def _texts_similar(t1, t2, threshold=0.6):
    """Check if two subtitle texts are similar enough to merge"""
    if not t1 or not t2:
        return False
    t1 = t1.strip()
    t2 = t2.strip()
    if t1 == t2:
        return True
    # Use SequenceMatcher for smarter similarity
    ratio = SequenceMatcher(None, t1, t2).ratio()
    return ratio >= threshold


def _seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSE Single-Thread Subtitle Extractor")
    parser.add_argument("--video", required=True)
    parser.add_argument("--ymin", type=int, required=True)
    parser.add_argument("--ymax", type=int, required=True)
    parser.add_argument("--xmin", type=int, required=True)
    parser.add_argument("--xmax", type=int, required=True)
    parser.add_argument("--mode", default="fast", choices=["fast", "auto", "accurate"])

    args = parser.parse_args()

    # CRITICAL: Wipe sys.argv so PaddleOCR doesn't crash on our custom flags
    sys.argv = [sys.argv[0]]

    process_vse(args.video, args.ymin, args.ymax, args.xmin, args.xmax, args.mode)
