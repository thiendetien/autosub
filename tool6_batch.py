#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tool6 Batch - Batch Video OCR Processing
Uses OCRWorker from tool6.py (working code)
"""

# Disable slow model connectivity check for faster startup
import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys

import cv2
import numpy as np
from enum import Enum
from PyQt5 import QtWidgets, QtCore, QtGui, sip
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QScrollArea, QGridLayout, QLineEdit, QCheckBox, QComboBox,
    QProgressBar, QSizePolicy, QSpacerItem, QGroupBox, QMessageBox, QStatusBar, QButtonGroup
)
from PyQt5.QtCore import (
    QThread, pyqtSignal, QObject, Qt, QTimer, QMutex, QMutexLocker, QSettings
)
from PyQt5.QtGui import QColor, QFont

# Import working OCR from tool6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool6 import OCRWorker, InpaintWorker
from srt_to_video import MergeWorker
import download8movie_gui  # Import the downloader GUI module
import subprocess
import json
import pysrt


class VSEWorker(QThread):
    """Worker that runs vse_runner.py as a subprocess for subtitle extraction"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, list)  # srt_path, segments
    error = pyqtSignal(str)

    def __init__(self, video_path, roi, vse_mode="fast"):
        super().__init__()
        self.video_path = video_path
        self.roi = roi  # [x, y, w, h] from GUI
        self.vse_mode = vse_mode

    def run(self):
        try:
            # Convert GUI ROI [x, y, w, h] to VSE format [ymin, ymax, xmin, xmax]
            x, y, w, h = self.roi
            ymin = y
            ymax = y + h
            xmin = x
            xmax = x + w

            cmd = [
                sys.executable, "-u", "vse_runner.py",
                "--video", self.video_path,
                "--ymin", str(int(ymin)),
                "--ymax", str(int(ymax)),
                "--xmin", str(int(xmin)),
                "--xmax", str(int(xmax)),
                "--mode", self.vse_mode,
            ]

            self.progress.emit(5)

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8',
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            srt_path = None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                print(f"[VSE] {line}")

                # Parse progress
                if line.startswith("PROGRESS:"):
                    try:
                        pct = int(line.split(":")[1].strip().replace("%", ""))
                        self.progress.emit(10 + int(pct * 0.8))
                    except:
                        pass
                elif line.startswith("OUTPUT_SRT:"):
                    srt_path = line.split(":", 1)[1].strip()
                elif line.startswith("PHASE1_START"):
                    self.progress.emit(10)
                elif line.startswith("PHASE3_START"):
                    self.progress.emit(30)
                elif line.startswith("PHASE4_START"):
                    self.progress.emit(90)

            process.wait()

            if process.returncode != 0:
                self.error.emit(f"VSE subprocess exited with code {process.returncode}")
                return

            self.progress.emit(95)

            # Build segments from segments.json (has proper bounding boxes)
            segments = []
            if srt_path:
                segments_json_path = srt_path.replace(".srt", ".segments.json")
                if os.path.exists(segments_json_path):
                    try:
                        with open(segments_json_path, 'r', encoding='utf-8') as f:
                            segments = json.load(f)
                        # Convert lists to tuples for boxes
                        for seg in segments:
                            if seg.get("box"):
                                seg["box"] = tuple(seg["box"])
                            if seg.get("stable_box"):
                                seg["stable_box"] = tuple(seg["stable_box"])
                        print(f"Loaded {len(segments)} segments with bounding boxes")
                    except Exception as e:
                        print(f"Warning: Failed to load segments.json: {e}")
                
                # Fallback: build from SRT if no segments.json
                if not segments and os.path.exists(srt_path):
                    try:
                        subs = pysrt.open(srt_path, encoding='utf-8')
                        for sub in subs:
                            start_s = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                            end_s = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                            segments.append({
                                "text": sub.text,
                                "start": start_s,
                                "end": end_s,
                                "box": (xmin, ymin, xmax, ymax),
                            })
                    except Exception as e:
                        print(f"Warning: Failed to parse SRT for segments: {e}")

            self.progress.emit(100)
            self.finished.emit(srt_path or "", segments)

        except Exception as e:
            self.error.emit(str(e))

class TranslationWorker(QThread):
    """Worker for Auto-Translating SRT via Gemini API"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str) # Return translated SRT path
    error = pyqtSignal(str)

    def __init__(self, srt_path, api_key="", model_name="gemini-3-pro-preview"):
        super().__init__()
        self.srt_path = srt_path
        self.api_key = api_key
        self.model_name = model_name
        
    def run(self):
        try:
            import os
            from trans_api import translate_srt
            
            self.progress.emit(10)
            base_name = os.path.splitext(self.srt_path)[0]
            # Output pattern expected by TTS step
            out_srt = base_name + "_vi.srt"
            
            self.progress.emit(50)
            success, err_msg = translate_srt(self.srt_path, out_srt, self.api_key, self.model_name)
            
            if success and os.path.exists(out_srt):
                self.progress.emit(100)
                self.finished.emit(out_srt)
            else:
                self.error.emit(f"API Failed: {err_msg}")
        except Exception as e:
            self.error.emit(str(e))


class STTWorker(QThread):
    """Worker for Speech-to-Text SRT extraction using WhisperX (auto_subtitle pipeline)"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # Return generated SRT path
    error = pyqtSignal(str)

    def __init__(self, video_path, language="en", model="large-v3"):
        super().__init__()
        self.video_path = video_path
        self.language = language
        self.model = model

    def run(self):
        try:
            import tempfile
            self.progress.emit(5)

            # Import auto_subtitle modules
            auto_sub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auto_subtitle')
            if auto_sub_dir not in sys.path:
                sys.path.insert(0, auto_sub_dir)

            from subtitle_tool.audio import extract_audio
            from subtitle_tool.transcriber import Transcriber
            from subtitle_tool.segmenter import SubtitleSegmenter
            from subtitle_tool.exporters import export_srt

            video_dir = os.path.dirname(self.video_path)
            basename = os.path.splitext(os.path.basename(self.video_path))[0]
            out_srt = os.path.join(video_dir, f"{basename}_en.srt")

            self.progress.emit(10)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_wav = os.path.join(temp_dir, "audio.wav")

                # Extract audio
                print(f"[STT] Extracting audio from {os.path.basename(self.video_path)}...")
                extract_audio(self.video_path, temp_wav)
                self.progress.emit(20)

                # Load model and transcribe
                print(f"[STT] Loading WhisperX model '{self.model}'...")
                transcriber = Transcriber(model_size=self.model, device="cuda", compute_type="float16")
                transcriber.load_model()
                self.progress.emit(40)

                print(f"[STT] Transcribing + Forced Alignment...")
                words, metadata = transcriber.transcribe(temp_wav, self.language)
                self.progress.emit(70)

                if not words:
                    self.error.emit("No speech detected in video.")
                    return

                # Segment
                segmenter = SubtitleSegmenter(
                    max_gap=1.0,
                    max_duration=6.0,
                    start_padding_ms=40,
                    end_padding_ms=100
                )
                detected_lang = metadata.get("language", "en")
                cues = segmenter.segment(words, detected_lang)
                self.progress.emit(85)

                # Export SRT only
                print(f"[STT] Exporting {len(cues)} cues to SRT...")
                export_srt(cues, out_srt)
                self.progress.emit(100)

                print(f"[STT] Done! {len(words)} words, {len(cues)} subtitles → {os.path.basename(out_srt)}")
                self.finished.emit(out_srt)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class DelogoWorker(QThread):
    """Worker for Fast FFmpeg Delogo"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str) # Return output path
    error = pyqtSignal(str)

    def __init__(self, input_video, output_video, segments, roi=None):
        super().__init__()
        self.input_video = input_video
        self.output_video = output_video
        self.segments = segments
        self.roi = roi
        self.stop_requested = False
        
    def run(self):
        try:
            if not self.roi:
                raise ValueError("ROI not set for delogo")
            
            # Get Video Dimensions
            import cv2
            cap = cv2.VideoCapture(self.input_video)
            if not cap.isOpened():
                raise RuntimeError("Cannot open video to check dimensions")
            
            V_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            V_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0
            cap.release()
                
            x, y, w, h = self.roi
            
            # 1. Enforce strict bounds (Delogo needs pixels AROUND the box)
            # Must have at least 1 pixel margin on all sides if possible
            x = max(1, x)
            y = max(1, y)
            
            # If box is too big, shrink it
            if x + w >= V_W:
                w = V_W - x - 1
            if y + h >= V_H:
                h = V_H - y - 1
                
            # 2. Enforce Even coordinates (for YUV420)
            if x % 2 != 0: x += 1
            if y % 2 != 0: y += 1
            if w % 2 != 0: w -= 1
            if h % 2 != 0: h -= 1
            
            # Final sanity check
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid ROI after adjustment: {x},{y},{w},{h} (Video: {V_W}x{V_H})")

            output_path = self.output_video

            self.progress.emit(10)
            
            self.progress.emit(10)
            
            # Generate filter complex script
            # enable='between(t,start,end)+between(t,start,end)...'
            
            # 1. Sort segments and merge closer ones to avoid huge filter string
            # (Optional optimized merging logic could go here)
            
            # 2. Build time string
            time_exprs = []
            if self.segments:
                for seg in self.segments:
                    start = max(0, seg['start'] - 0.2) # Pad a bit
                    end = seg['end'] + 0.2
                    time_exprs.append(f"between(t,{start:.3f},{end:.3f})")
            
            if not time_exprs:
                # No segments, just copy Video
                import shutil
                shutil.copy2(self.input_video, output_path)
                self.progress.emit(100)
                self.finished.emit(output_path)
                return

            
            # 3. Per-Box Blur Strategy
            # Blur each detected text box individually with light blur
            # This preserves background better than ROI-wide blur
            
            box_time_map = {}  # (x,y,w,h) -> list of time ranges
            
            for seg in self.segments:
                if 'box' not in seg or not seg['box']:
                    continue
                    
                box = seg['box']
                bx1, by1, bx2, by2 = box
                
                bx = int(bx1)
                by = int(by1)
                bw = int(bx2 - bx1)
                bh = int(by2 - by1)
                
                # Ensure even for YUV420
                if bx % 2 != 0: bx += 1
                if by % 2 != 0: by += 1
                if bw % 2 != 0: bw -= 1
                if bh % 2 != 0: bh -= 1
                
                if bw <= 4 or bh <= 4:
                    continue
                
                box_key = (bx, by, bw, bh)
                start = max(0, seg['start'] - 0.05)
                end = seg['end'] + 0.05
                
                if box_key not in box_time_map:
                    box_time_map[box_key] = []
                box_time_map[box_key].append((start, end))
            
            if not box_time_map:
                import shutil
                shutil.copy2(self.input_video, output_path)
                self.progress.emit(100)
                self.finished.emit(output_path)
                return
            
            print(f"🔹 Hybrid Clone+Blur: {len(box_time_map)} unique text boxes")
            
            # Build filter: split -> for each box: crop->blur->overlay
            num_boxes = len(box_time_map)
            split_labels = ['main'] + [f'b{i}' for i in range(num_boxes)]
            
            split_str = "][".join(split_labels)
            filter_parts = [f"split={num_boxes+1}[{split_str}]"]
            
            
            current = 'main'
            for idx, (box_key, times) in enumerate(box_time_map.items()):
                bx, by, bw, bh = box_key
                
                # Expand box MUCH wider and taller for better coverage
                # Horizontal: +30px each side (60px total wider)
                # Vertical: +20px (taller)
                h_margin = 30
                v_margin = 20
                
                bx = max(0, bx - h_margin)
                by = max(0, by - v_margin)
                bw = min(V_W - bx, bw + 2*h_margin)
                bh = min(V_H - by, bh + 2*v_margin)
                
                # Clone from ABOVE the subtitle (offset by height + small gap)
                clone_y = max(0, by - bh - 10)  # 10px gap above
                
                time_exprs = [f"between(t,{s:.3f},{e:.3f})" for s, e in times]
                enable = "+".join(time_exprs)
                
                clone_label = f'clone{idx}'
                next_label = f'v{idx}' if idx < num_boxes - 1 else 'out'
                
                # Calculate feather border thickness
                feather_t = min(12, bw//2 - 2, bh//2 - 2)  # 10-12px soft edge
                feather_t = max(6, feather_t)  # Minimum 6px
                
                # Define stream names
                clone_raw = f"clone_raw_{idx}"
                clone_rgba = f"clone_rgba_{idx}"
                mask_base = f"mask_base_{idx}"
                mask_white = f"mask_white_{idx}"
                mask_border = f"mask_border_{idx}"
                mask_final = f"mask_final_{idx}"
                clone_feathered = f"clone_feathered_{idx}"
                
                # FILTER GRAPH: Sharp Clone + Feathered Alpha Mask
                # 1. Crop clean region from above
                filter_parts.append(f"[b{idx}]crop={bw}:{bh}:{bx}:{clone_y}[{clone_raw}]")
                
                # 2. Convert to RGBA (keep sharp!)
                filter_parts.append(f"[{clone_raw}]format=rgba[{clone_rgba}]")
                
                # 3. Create mask base (same crop for alignment)
                filter_parts.append(f"[b{idx}]crop={bw}:{bh}:{bx}:{clone_y}[{mask_base}]")
                
                # 4. Build alpha mask: white center, black border, blur
                filter_parts.append(f"[{mask_base}]format=rgba[{mask_white}]")
                filter_parts.append(f"[{mask_white}]drawbox=c=white:t=fill:replace=1[{mask_border}]")
                filter_parts.append(f"[{mask_border}]drawbox=x=0:y=0:w=iw:h=ih:c=black:t={feather_t}:replace=1,gblur=sigma=6[{mask_final}]")
                
                # 5. Alpha merge: Sharp content + Feathered mask
                filter_parts.append(f"[{clone_rgba}][{mask_final}]alphamerge[{clone_feathered}]")
                
                # 6. Overlay on main video at subtitle position
                filter_parts.append(f"[{current}][{clone_feathered}]overlay={bx}:{by}:enable='{enable}'[{next_label}]")
                
                current = next_label
            
            
            filter_script_content = ";\n".join(filter_parts)
            
            print(f"📝 Hybrid Clone+Blur: {len(filter_parts)} filter parts, {len(box_time_map)} boxes")
            
            # Write filter to temp script file
            filter_script_path = self.input_video + ".filter_script.txt"
            with open(filter_script_path, "w", encoding="utf-8") as f:
                f.write(filter_script_content)
                
            self.progress.emit(20)
            
            # 4. Run FFmpeg
            # Using libx264 for CPU compatibility and speed
            # Use -filter_complex_script to apply complex filter from file
            
            # Detect FFmpeg path - prioritize /usr/bin/ffmpeg to avoid Conda's non-GPL limits
            ffmpeg_bin = "ffmpeg"
            if os.path.exists("/usr/bin/ffmpeg"):
                ffmpeg_bin = "/usr/bin/ffmpeg"
            
            # Use h264_nvenc for Fast GPU acceleration
            v_codec = "h264_nvenc"
            
            cmd = [
                ffmpeg_bin, "-y",
                "-i", self.input_video,
                "-filter_complex_script", filter_script_path,
                "-map", "[out]",
                "-map", "0:a?",
                "-c:v", v_codec,
                "-preset", "p4",    # Balanced NVENC preset
                "-cq", "22",        # Constant Quality for NVENC
                "-c:a", "copy",
                output_path
            ]
            
            print(f"⚡ Running Blur Delogo (NVENC): {' '.join(cmd)}")
            
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    universal_newlines=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to launch FFmpeg: {e}")
            
            # Read log to update simple progress and capture error
            full_log = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    full_log.append(line.strip())
                    # Parse "time=00:01:23.45" to calculate progress
                    if "time=" in line:
                        try:
                            # Example: frame=  123 fps=0.0 q=0.0 size=    123kB time=00:00:01.23 bitrate=123.4kbits/s speed=2.34x
                            parts = line.split("time=")
                            if len(parts) > 1:
                                t_str = parts[1].split()[0]
                                h, m, s = t_str.split(':')
                                seconds = float(h) * 3600 + float(m) * 60 + float(s)
                                if video_duration > 0:
                                    progress_pct = min(99, int((seconds / video_duration) * 100))
                                    self.progress.emit(progress_pct) 
                        except:
                            pass
            
            if process.returncode != 0:
                # Extract last few lines for error message
                err_msg = "\n".join(full_log[-10:])
                is_nvenc_error = "Unknown encoder 'h264_nvenc'" in "\n".join(full_log) or "nvenc" in err_msg.lower() or "error selecting an encoder" in "\n".join(full_log).lower()
                if is_nvenc_error:
                    print(f"⚠️ NVENC failed, falling back to libx264...")
                    cmd[11] = "libx264"
                    cmd[13] = "superfast"
                    cmd[15] = "22"
                    cmd[14] = "-crf"  # Replace -cq with -crf
                    
                    print(f"⚡ Fallback CMD: {' '.join(cmd)}")
                    process = subprocess.Popen(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        universal_newlines=True
                    )
                    full_log = []
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            full_log.append(line.strip())
                            if "time=" in line:
                                try:
                                    parts = line.split("time=")
                                    if len(parts) > 1:
                                        t_str = parts[1].split()[0]
                                        h, m, s = t_str.split(':')
                                        seconds = float(h) * 3600 + float(m) * 60 + float(s)
                                        if video_duration > 0:
                                            progress_pct = min(99, int((seconds / video_duration) * 100))
                                            self.progress.emit(progress_pct) 
                                except:
                                    pass
                    if process.returncode != 0:
                        err_msg = "\n".join(full_log[-10:])
                        raise RuntimeError(f"FFmpeg fallback failed (code {process.returncode}):\n{err_msg}")
                else:
                    raise RuntimeError(f"FFmpeg failed (code {process.returncode}):\n{err_msg}")
                
            # Cleanup
            if os.path.exists(filter_script_path):
                os.remove(filter_script_path)
                
            self.progress.emit(100)
            self.finished.emit(output_path)
            
        except Exception as e:
            self.error.emit(str(e))


class LogoRemoveWorker(QThread):
    """Worker for removing logo using FFmpeg delogo filter (no OCR needed)"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # output path
    error = pyqtSignal(str)

    def __init__(self, cmd, output_video, video_duration, ffmpeg_bin, input_video, x, y, w, h):
        super().__init__()
        self.cmd = cmd
        self.output_video = output_video
        self.video_duration = video_duration
        self.ffmpeg_bin = ffmpeg_bin
        self.input_video = input_video
        self.x, self.y, self.w, self.h = x, y, w, h

    def run(self):
        try:
            self.progress.emit(5)

            try:
                process = subprocess.Popen(
                    self.cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to launch FFmpeg: {e}")

            full_log = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    full_log.append(line.strip())
                    if "time=" in line:
                        try:
                            parts = line.split("time=")
                            if len(parts) > 1:
                                t_str = parts[1].split()[0]
                                h, m, s = t_str.split(':')
                                seconds = float(h) * 3600 + float(m) * 60 + float(s)
                                if self.video_duration > 0:
                                    pct = min(99, int((seconds / self.video_duration) * 100))
                                    self.progress.emit(pct)
                        except:
                            pass

            if process.returncode != 0:
                err_msg = "\n".join(full_log[-10:])
                # Check for NVENC error and fallback to libx264
                is_nvenc_error = any(k in "\n".join(full_log) for k in [
                    "Unknown encoder 'h264_nvenc'", "nvenc", "error selecting an encoder"
                ])
                if is_nvenc_error:
                    print("⚠️ NVENC failed, falling back to libx264...")
                    fallback_cmd = [
                        self.ffmpeg_bin, "-y",
                        "-i", self.input_video,
                        "-vf", f"delogo=x={self.x}:y={self.y}:w={self.w}:h={self.h}",
                        "-c:v", "libx264",
                        "-preset", "superfast",
                        "-crf", "22",
                        "-c:a", "copy",
                        self.output_video
                    ]
                    process = subprocess.Popen(
                        fallback_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True
                    )
                    full_log = []
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            full_log.append(line.strip())
                            if "time=" in line:
                                try:
                                    parts = line.split("time=")
                                    if len(parts) > 1:
                                        t_str = parts[1].split()[0]
                                        h, m, s = t_str.split(':')
                                        seconds = float(h) * 3600 + float(m) * 60 + float(s)
                                        if self.video_duration > 0:
                                            pct = min(99, int((seconds / self.video_duration) * 100))
                                            self.progress.emit(pct)
                                except:
                                    pass
                    if process.returncode != 0:
                        err_msg = "\n".join(full_log[-10:])
                        raise RuntimeError(f"FFmpeg fallback failed (code {process.returncode}):\n{err_msg}")
                else:
                    raise RuntimeError(f"FFmpeg failed (code {process.returncode}):\n{err_msg}")

            self.progress.emit(100)
            self.finished.emit(self.output_video)

        except Exception as e:
            self.error.emit(str(e))


# ============= ENUMS =============

class VideoStatus(Enum):
    PENDING = "pending"
    ROI_SET = "roi_set"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class ProcessMode(Enum):
    SRT_ONLY = "srt_only"
    SRT_REMOVESUB = "srt_removesub"
    TTS_MERGE = "tts_merge"  # TTS + Burn SRT into video
    TTS_MERGE_MULTI = "tts_merge_multi" # Multi-language auto detect
    REMOVE_LOGO = "remove_logo"  # Remove logo/watermark only (no OCR)
    BURN_ONLY = "burn_only"
    STT_EXTRACT = "stt_extract"  # Speech-to-Text SRT extraction (WhisperX)


# ============= DATA CLASSES =============

class VideoQueueItem:
    """Single video in queue"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.roi = None  # [x, y, w, h]
        self.status = VideoStatus.PENDING
        self.output_srt = None
        self.segments = None
        self.error_msg = None
        self.srt_path = None  # For TTS+Merge mode - path to translated SRT
        self.sub_overrides = None  # Per-video subtitle settings override dict
        self.logo_start_sec = 0.0  # Logo time range start (seconds)
        self.logo_end_sec = 0.0    # Logo time range end (0 = entire video)
        self.speaker_mapping = {}  # Map of SpeakerName -> VoiceName
        self.line_mapping = {}     # Map of SRT line index (int) -> VoiceName
        self.target_lang = None    # Override target language (e.g., 'Tiếng Hindi')
        self.minimax_voice = None  # Override MiniMax voice
        self.kokoro_voice = None   # Override Kokoro voice
    
    def auto_detect_srt(self):
        """Auto-detect SRT file for this video"""
        import os
        video_dir = os.path.dirname(self.video_path)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        video_base = os.path.splitext(self.video_path)[0]
        
        # Also try base name without _nosub suffix
        base_without_nosub = video_base.replace("_nosub", "")
        name_without_nosub = video_name.replace("_nosub", "")
        
        srt_candidates = [
            base_without_nosub + "_en.srt",       # English
            video_base + "_en.srt",
            base_without_nosub + ".en.srt",
            video_base + ".en.srt",
            base_without_nosub + "_vi.srt",       # video_vi.srt
            video_base + "_vi.srt",               # video_nosub_vi.srt
            base_without_nosub + ".vi.srt",       # video.vi.srt
            video_base + ".vi.srt",               # video_nosub.vi.srt
            base_without_nosub + "_translated.srt", 
            video_base + "_translated.srt",
            base_without_nosub + "_dich.srt",
            video_base + "_dich.srt",
            video_base + ".srt",
            base_without_nosub + ".srt",
        ]
        
        for candidate in srt_candidates:
            if os.path.exists(candidate):
                self.srt_path = candidate
                return candidate
        return None

    def auto_detect_all_srts(self):
        """Auto-detect all multi-language SRTs for this video. Returns a list of dicts: [{'srt_path': path, 'lang': lang_name, 'suffix': suffix}]"""
        import os
        import glob
        video_base = os.path.splitext(self.video_path)[0]
        base_without_nosub = video_base.replace("_nosub", "")
        vid_name = os.path.splitext(os.path.basename(self.video_path))[0].replace("_nosub", "")
        video_dir = os.path.dirname(self.video_path)
        parent_dir = os.path.dirname(video_dir)
        
        # Mapping from suffix to language name
        suffix_to_lang = {
            "_ar": "Tiếng Ả Rập",
            "_hi": "Tiếng Hindi",
            "_id": "Tiếng Indonesia",
            "_it": "Tiếng Ý",
            "_ta": "Tiếng Tamil",
            "_en": "English",
            "_vi": "Tiếng Việt",
            "_zh": "Tiếng Trung",
            "_ko": "Tiếng Hàn",
            "_ja": "Tiếng Nhật",
            "_es": "Tiếng Tây Ban Nha",
            "_fr": "Tiếng Pháp"
        }
        
        results = []
        
        # Search in same directory as video
        pattern_same = base_without_nosub + "*.srt"
        found_files = glob.glob(pattern_same)
        
        # Also search in parent directory (SRTs from chatgpt-auto/grok-auto may land there)
        pattern_parent = os.path.join(parent_dir, vid_name + "*.srt")
        found_files += glob.glob(pattern_parent)
        
        # Also search in video_dir with just the basename pattern (in case base path differs)
        pattern_viddir = os.path.join(video_dir, vid_name + "*.srt")
        found_files += glob.glob(pattern_viddir)
        
        # Deduplicate file paths
        seen_paths = set()
        unique_files = []
        for f in found_files:
            rp = os.path.realpath(f)
            if rp not in seen_paths:
                seen_paths.add(rp)
                unique_files.append(f)
        
        for file in unique_files:
            name = os.path.splitext(os.path.basename(file))[0]
            # Get suffix by comparing to vid_name
            if name.startswith(vid_name):
                suffix = name[len(vid_name):]
            else:
                continue
                
            if suffix in suffix_to_lang:
                results.append({"srt_path": file, "lang": suffix_to_lang[suffix], "suffix": suffix})
            elif suffix in ("_translated", "_dich"):
                results.append({"srt_path": file, "lang": "Tiếng Việt", "suffix": suffix})
                
        # Deduplicate by language
        unique_results = {}
        for r in results:
            if r["lang"] not in unique_results:
                unique_results[r["lang"]] = r
                
        return list(unique_results.values())
    
    def auto_load_roi_from_segments(self):
        """Load ROI from segments.json stable_box"""
        import os
        import json
        video_base = os.path.splitext(self.video_path)[0]
        base_without_nosub = video_base.replace("_nosub", "")
        
        segments_candidates = [
            video_base + ".segments.json",
            base_without_nosub + ".segments.json",
        ]
        
        for segments_file in segments_candidates:
            if os.path.exists(segments_file):
                try:
                    with open(segments_file, 'r', encoding='utf-8') as f:
                        segments = json.load(f)
                    if segments and len(segments) > 0:
                        # Get stable_box from first segment
                        box = segments[0].get("stable_box")
                        if box and len(box) == 4:
                            self.roi = [int(x) for x in box]  # [x, y, w, h]
                            return self.roi
                except Exception as e:
                    print(f"⚠️ Error loading segments: {e}")
        return None


# ============= UI WIDGETS =============

class RangeSlider(QtWidgets.QWidget):
    """Custom dual-handle range slider for selecting time ranges"""
    rangeChanged = QtCore.pyqtSignal(int, int)  # (start_val, end_val)
    
    def __init__(self, minimum=0, maximum=100, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._low = minimum
        self._high = maximum
        self._pressed_handle = None  # 'low', 'high', or None
        self._handle_width = 14
        self._bar_height = 8
        self.setMinimumHeight(36)
        self.setMaximumHeight(36)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._fmt_func = None  # Optional format function for labels
    
    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = max(minimum, maximum)
        self._low = max(self._low, minimum)
        self._high = min(self._high, self._max)
        self.update()
    
    def setLow(self, val):
        self._low = max(self._min, min(val, self._high))
        self.update()
        self.rangeChanged.emit(self._low, self._high)
    
    def setHigh(self, val):
        self._high = max(self._low, min(val, self._max))
        self.update()
        self.rangeChanged.emit(self._low, self._high)
    
    def low(self):
        return self._low
    
    def high(self):
        return self._high
    
    def setFormatFunc(self, func):
        """Set function to format value to label (e.g., seconds to MM:SS)"""
        self._fmt_func = func
    
    def _val_to_x(self, val):
        """Convert value to x pixel position"""
        w = self.width() - self._handle_width
        if self._max == self._min:
            return self._handle_width // 2
        ratio = (val - self._min) / (self._max - self._min)
        return int(ratio * w) + self._handle_width // 2
    
    def _x_to_val(self, x):
        """Convert x pixel position to value"""
        w = self.width() - self._handle_width
        if w <= 0:
            return self._min
        ratio = (x - self._handle_width // 2) / w
        ratio = max(0.0, min(1.0, ratio))
        return int(self._min + ratio * (self._max - self._min))
    
    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor, QFont
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        hw = self._handle_width
        bh = self._bar_height
        cy = self.height() - bh - 4  # bar y center
        
        # Draw track background
        track_rect = QtCore.QRectF(hw // 2, cy, self.width() - hw, bh)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QColor(200, 200, 200))
        p.drawRoundedRect(track_rect, bh // 2, bh // 2)
        
        # Draw selected range (highlighted)
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)
        range_rect = QtCore.QRectF(x_low, cy, x_high - x_low, bh)
        p.setBrush(QColor(0, 123, 255))
        p.drawRoundedRect(range_rect, bh // 2, bh // 2)
        
        # Draw handles
        for val, color in [(self._low, QColor(40, 167, 69)), (self._high, QColor(220, 53, 69))]:
            hx = self._val_to_x(val)
            handle_rect = QtCore.QRectF(hx - hw // 2, cy - 3, hw, bh + 6)
            p.setBrush(color)
            p.setPen(QColor(255, 255, 255))
            p.drawRoundedRect(handle_rect, 3, 3)
        
        # Draw time labels above handles
        p.setPen(QColor(0, 0, 0))
        font = QFont()
        font.setPixelSize(10)
        font.setBold(True)
        p.setFont(font)
        
        fmt = self._fmt_func or str
        
        # Low label
        low_text = fmt(self._low)
        low_x = self._val_to_x(self._low)
        p.drawText(QtCore.QRectF(low_x - 30, 0, 60, 16), QtCore.Qt.AlignCenter, low_text)
        
        # High label
        if self._high >= self._max:
            high_text = "END"
        else:
            high_text = fmt(self._high)
        high_x = self._val_to_x(self._high)
        p.drawText(QtCore.QRectF(high_x - 30, 0, 60, 16), QtCore.Qt.AlignCenter, high_text)
        
        p.end()
    
    def mousePressEvent(self, event):
        x = event.pos().x()
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)
        
        # Check which handle is closer
        dist_low = abs(x - x_low)
        dist_high = abs(x - x_high)
        
        if dist_low <= dist_high and dist_low < 20:
            self._pressed_handle = 'low'
        elif dist_high < 20:
            self._pressed_handle = 'high'
        elif x < x_low:
            self._pressed_handle = 'low'
        else:
            self._pressed_handle = 'high'
        
        self._move_handle(x)
    
    def mouseMoveEvent(self, event):
        if self._pressed_handle:
            self._move_handle(event.pos().x())
    
    def mouseReleaseEvent(self, event):
        self._pressed_handle = None
    
    def _move_handle(self, x):
        val = self._x_to_val(x)
        if self._pressed_handle == 'low':
            self._low = max(self._min, min(val, self._high))
        elif self._pressed_handle == 'high':
            self._high = max(self._low, min(val, self._max))
        self.update()
        self.rangeChanged.emit(self._low, self._high)


class VideoCard(QtWidgets.QFrame):
    """Card widget for each video in grid"""
    roi_changed = QtCore.pyqtSignal(object, int, int, int, int)
    remove_requested = QtCore.pyqtSignal(object)
    apply_all_requested = QtCore.pyqtSignal(object)
    
    # Class-level frame cache: {video_path: (frame_bgr, total_frames, fps, duration)}
    _frame_cache = {}
    
    def __init__(self, queue_item, is_portrait=False, parent=None):
        super().__init__(parent)
        self.queue_item = queue_item
        self.is_portrait = is_portrait
        self.cap = None
        
        self.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Raised)
        self.setLineWidth(2)
        
        self.init_ui()
        self.load_video()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        filename = os.path.basename(self.queue_item.video_path)
        if self.queue_item.target_lang:
            display_name = f"[{self.queue_item.target_lang}] {filename}"
        else:
            display_name = filename
        self.name_label = QtWidgets.QLabel(display_name)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(30)
        layout.addWidget(self.name_label)
        
        if self.is_portrait:
            self.preview = VideoPreview(self.queue_item.video_path)
            self.preview.setMinimumSize(200, 350)
            self.preview.setMaximumSize(250, 450)
        else:
            self.preview = VideoPreview(self.queue_item.video_path)
            self.preview.setMinimumSize(300, 200)
            self.preview.setMaximumSize(400, 300)
        
        self.preview.roi_changed.connect(self.on_roi_changed)
        layout.addWidget(self.preview)
        
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("⏱️"))
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QtWidgets.QLabel("00:00 / 00:00")
        self.frame_label.setMinimumWidth(100)
        slider_layout.addWidget(self.frame_label)
        
        layout.addLayout(slider_layout)
        
        self.status_label = QtWidgets.QLabel("⏳ Pending - Draw ROI")
        self.status_label.setStyleSheet("padding: 5px; background: #fff3cd; border-radius: 3px; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        # SRT info row (for TTS+Merge mode)
        srt_row = QtWidgets.QHBoxLayout()
        self.srt_label = QtWidgets.QLabel("SRT: (not set)")
        self.srt_label.setStyleSheet("font-size: 9px; color: #666;")
        self.srt_label.setWordWrap(True)
        self.srt_label.setMaximumHeight(20)
        srt_row.addWidget(self.srt_label)
        
        self.btn_browse_srt = QtWidgets.QPushButton("📂")
        self.btn_browse_srt.setMaximumWidth(30)
        self.btn_browse_srt.setMaximumHeight(20)
        self.btn_browse_srt.setToolTip("Browse for SRT file")
        self.btn_browse_srt.clicked.connect(self.on_browse_srt)
        srt_row.addWidget(self.btn_browse_srt)
        
        self.btn_stt = QtWidgets.QPushButton("🎤 STT")
        self.btn_stt.setMaximumWidth(60)
        self.btn_stt.setMaximumHeight(20)
        self.btn_stt.setToolTip("Speech-to-Text: Tách SRT từ video bằng WhisperX")
        self.btn_stt.setStyleSheet("font-size: 9px; background: #6f42c1; color: white; border-radius: 3px;")
        self.btn_stt.clicked.connect(self.on_stt)
        srt_row.addWidget(self.btn_stt)
        layout.addLayout(srt_row)
        
        # Update SRT label if already detected
        self.update_srt_display()
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(15)
        layout.addWidget(self.progress_bar)

        
        # Action buttons row
        btn_row = QtWidgets.QHBoxLayout()
        
        self.btn_sub_per_video = QtWidgets.QPushButton("🎨")
        self.btn_sub_per_video.setMaximumWidth(35)
        self.btn_sub_per_video.setMaximumHeight(25)
        self.btn_sub_per_video.setToolTip("Per-video Subtitle Settings")
        self.btn_sub_per_video.clicked.connect(self.on_per_video_sub_settings)
        btn_row.addWidget(self.btn_sub_per_video)
        self.update_sub_btn_style()
        
        self.btn_voice_mapping = QtWidgets.QPushButton("🗣️")
        self.btn_voice_mapping.setMaximumWidth(35)
        self.btn_voice_mapping.setMaximumHeight(25)
        self.btn_voice_mapping.setToolTip("Voice Mapping (Multi-Speaker TTS)")
        self.btn_voice_mapping.clicked.connect(self.on_voice_mapping)
        btn_row.addWidget(self.btn_voice_mapping)
        
        self.btn_open_folder = QtWidgets.QPushButton("📁")
        self.btn_open_folder.setMaximumWidth(35)
        self.btn_open_folder.setMaximumHeight(25)
        self.btn_open_folder.setToolTip("Open output folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self.on_open_folder)
        btn_row.addWidget(self.btn_open_folder)
        
        self.btn_preview_srt = QtWidgets.QPushButton("👁️")
        self.btn_preview_srt.setMaximumWidth(35)
        self.btn_preview_srt.setMaximumHeight(25)
        self.btn_preview_srt.setToolTip("Preview SRT")
        self.btn_preview_srt.setEnabled(False)
        self.btn_preview_srt.clicked.connect(self.on_preview_srt)
        btn_row.addWidget(self.btn_preview_srt)
        
        self.btn_apply_all = QtWidgets.QPushButton("🌍 Apply All")
        self.btn_apply_all.setMaximumHeight(25)
        self.btn_apply_all.setToolTip("Apply this ROI to all videos in queue")
        self.btn_apply_all.setEnabled(False)
        self.btn_apply_all.clicked.connect(lambda: self.apply_all_requested.emit(self.queue_item))
        btn_row.addWidget(self.btn_apply_all)
        
        btn_row.addStretch()
        
        self.btn_remove = QtWidgets.QPushButton("🗑️")
        self.btn_remove.setMaximumWidth(35)
        self.btn_remove.setMaximumHeight(25)
        self.btn_remove.setToolTip("Remove video")
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self.queue_item))
        btn_row.addWidget(self.btn_remove)
        
        layout.addLayout(btn_row)
        
        # Logo time range panel (visible only in REMOVE_LOGO mode)
        self.logo_panel = QtWidgets.QFrame()
        self.logo_panel.setStyleSheet("background: #e8f4f8; border: 1px solid #b8daff; border-radius: 4px; padding: 4px;")
        logo_panel_layout = QtWidgets.QVBoxLayout()
        logo_panel_layout.setContentsMargins(6, 4, 6, 4)
        logo_panel_layout.setSpacing(2)
        
        logo_title = QtWidgets.QLabel("✂️ Kéo chọn khoảng thời gian logo xuất hiện")
        logo_title.setStyleSheet("font-size: 10px; font-weight: bold; border: none; background: transparent;")
        logo_panel_layout.addWidget(logo_title)
        
        # Range slider (single bar with 2 handles)
        self.logo_range_slider = RangeSlider(0, 100)
        self.logo_range_slider.setFormatFunc(self._sec_to_mmss)
        self.logo_range_slider.rangeChanged.connect(self._on_logo_range_changed)
        logo_panel_layout.addWidget(self.logo_range_slider)
        
        self.logo_range_info = QtWidgets.QLabel("🎯 Kéo 🟢 start và 🔴 end để chọn vùng logo")
        self.logo_range_info.setStyleSheet("font-size: 9px; color: #555; border: none; background: transparent;")
        logo_panel_layout.addWidget(self.logo_range_info)
        
        self.logo_panel.setLayout(logo_panel_layout)
        self.logo_panel.setVisible(False)
        layout.addWidget(self.logo_panel)
        
        self._video_duration_sec = 0.0
        
        self.setLayout(layout)
    
    def load_video(self):
        vpath = self.queue_item.video_path
        
        # Check cache first (for multi-lang: same video, different SRTs)
        if vpath in VideoCard._frame_cache:
            cached = VideoCard._frame_cache[vpath]
            frame, total, fps, dur = cached
            self._fps = fps
            self._video_duration_sec = dur
            self.frame_slider.setRange(0, max(0, total - 1))
            self.frame_slider.setValue(total // 2)
            self.frame_slider.setEnabled(True)
            self.preview.set_frame(frame)
            cur_time = self._sec_to_mmss(int((total // 2) / fps))
            total_time = self._sec_to_mmss(int(total / fps))
            self.frame_label.setText(f"{cur_time} / {total_time}")
            dur_int = int(dur)
            self.logo_range_slider.setRange(0, dur_int)
            self.logo_range_slider.setHigh(dur_int)
            return
        
        self.cap = cv2.VideoCapture(vpath)
        if self.cap.isOpened():
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._video_duration_sec = total / self._fps
            
            self.frame_slider.setRange(0, max(0, total - 1))
            mid = total // 2
            self.frame_slider.setValue(mid)
            self.frame_slider.setEnabled(True)
            
            # Read middle frame and cache it
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = self.cap.read()
            if ok:
                self.preview.set_frame(frame)
                VideoCard._frame_cache[vpath] = (frame.copy(), total, self._fps, self._video_duration_sec)
                cur_time = self._sec_to_mmss(int(mid / self._fps))
                total_time = self._sec_to_mmss(int(total / self._fps))
                self.frame_label.setText(f"{cur_time} / {total_time}")
            
            # Release capture immediately to save resources
            self.cap.release()
            self.cap = None
            
            dur = int(self._video_duration_sec)
            self.logo_range_slider.setRange(0, dur)
            self.logo_range_slider.setHigh(dur)
    
    def on_frame_changed(self, frame_idx):
        vpath = self.queue_item.video_path
        
        # Re-open video capture only when user scrubs the slider
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        
        if ok:
            self.preview.set_frame(frame)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = getattr(self, '_fps', 30.0)
            cur_time = self._sec_to_mmss(int(frame_idx / fps))
            total_time = self._sec_to_mmss(int(total / fps))
            self.frame_label.setText(f"{cur_time} / {total_time}")
        
        cap.release()
    
    def _on_logo_range_changed(self, low, high):
        """Handle range slider change"""
        dur = int(self._video_duration_sec)
        self.queue_item.logo_start_sec = float(low)
        self.queue_item.logo_end_sec = 0.0 if high >= dur else float(high)
        
        # Update info label
        if high >= dur:
            range_text = f"{self._sec_to_mmss(low)} → END ({dur - low}s)"
        else:
            range_text = f"{self._sec_to_mmss(low)} → {self._sec_to_mmss(high)} ({high - low}s)"
        self.logo_range_info.setText(f"🎯 Remove: {range_text} / Video: {self._sec_to_mmss(dur)}")
    
    def show_logo_panel(self):
        """Show the logo time range panel"""
        self.logo_panel.setVisible(True)
        # Trigger initial info update
        self._on_logo_range_changed(self.logo_range_slider.low(), self.logo_range_slider.high())
    
    def hide_logo_panel(self):
        """Hide the logo time range panel"""
        self.logo_panel.setVisible(False)
    
    @staticmethod
    def _sec_to_mmss(sec):
        m = int(sec) // 60
        s = int(sec) % 60
        return f"{m:02d}:{s:02d}"
    
    def on_roi_changed(self, x, y, w, h):
        self.queue_item.roi = [x, y, w, h]
        self.queue_item.status = VideoStatus.ROI_SET
        self.update_status()
        self.roi_changed.emit(self.queue_item, x, y, w, h)
    
    def update_status(self):
        status = self.queue_item.status
        
        if status == VideoStatus.PENDING:
            self.status_label.setText("⏳ Pending - Draw ROI")
            self.status_label.setStyleSheet("padding: 5px; background: #fff3cd; border-radius: 3px; font-size: 10px;")
            self.progress_bar.setVisible(False)
            self.btn_apply_all.setEnabled(False)
            self.status_label.setText("⏳ Pending - Draw ROI")
            self.status_label.setStyleSheet("padding: 5px; background: #fff3cd; border-radius: 3px; font-size: 10px;")
            self.progress_bar.setVisible(False)
        elif status == VideoStatus.ROI_SET:
            roi = self.queue_item.roi
            self.status_label.setText(f"✅ ROI: {roi[2]}x{roi[3]}")
            self.status_label.setStyleSheet("padding: 5px; background: #d4edda; border-radius: 3px; font-size: 10px;")
            self.progress_bar.setVisible(False)
            self.btn_apply_all.setEnabled(True)
        elif status == VideoStatus.PROCESSING:
            self.status_label.setText("🔄 Processing...")
            self.status_label.setStyleSheet("padding: 5px; background: #d1ecf1; border-radius: 3px; font-size: 10px;")
            self.progress_bar.setVisible(True)
        elif status == VideoStatus.DONE:
            self.status_label.setText("✅ Done!")
            self.status_label.setStyleSheet("padding: 5px; background: #d4edda; border-radius: 3px; font-size: 10px;")
            self.progress_bar.setVisible(False)
            # Enable output buttons
            self.btn_open_folder.setEnabled(True)
            if self.queue_item.output_srt and os.path.exists(self.queue_item.output_srt):
                self.btn_preview_srt.setEnabled(True)
            self.status_label.setText("✅ Done!")
            self.status_label.setStyleSheet("padding: 5px; background: #d4edda; border-radius: 3px; font-size: 10px;")
            self.progress_bar.setVisible(False)
        elif status == VideoStatus.ERROR:
            self.status_label.setText("❌ Error")
            self.status_label.setStyleSheet("padding: 5px; background: #f8d7da; border-radius: 3px; font-size: 10px;")
            self.progress_bar.setVisible(False)
    
    def set_progress(self, value, phase="ocr"):
        """Set progress with phase support: ocr=0-50%, inpaint=50-100%"""
        if phase == "ocr":
            # OCR phase: 0-50%
            scaled = int(value * 0.5)
            self.progress_bar.setValue(scaled)
            self.status_label.setText(f"🔄 OCR: {value}%")
        else:
            # Inpaint phase: 50-100%
            scaled = 50 + int(value * 0.5)
            self.progress_bar.setValue(scaled)
            self.status_label.setText(f"🧼 RemoveSub: {value}%")
        self.status_label.setStyleSheet("padding: 5px; background: #d1ecf1; border-radius: 3px; font-size: 10px;")
    
    
    def on_open_folder(self):
        """Open output folder"""
        import subprocess
        folder = os.path.dirname(self.queue_item.video_path)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])
    
    def on_preview_srt(self):
        """Preview SRT content"""
        if not self.queue_item.output_srt or not os.path.exists(self.queue_item.output_srt):
            QtWidgets.QMessageBox.warning(self, "No SRT", "SRT file not found!")
            return
        
        with open(self.queue_item.output_srt, 'r', encoding='utf-8') as f:
            content = f.read()
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"SRT Preview - {os.path.basename(self.queue_item.output_srt)}")
        dialog.resize(600, 400)
        
        layout = QtWidgets.QVBoxLayout()
        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlainText(content)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)
        
        dialog.setLayout(layout)
        dialog.exec_()
    

    
    def on_browse_srt(self):
        """Browse for SRT file manually"""
        srt_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Select SRT for {os.path.basename(self.queue_item.video_path)}",
            os.path.dirname(self.queue_item.video_path),
            "Subtitle Files (*.srt)"
        )
        if srt_path:
            self.queue_item.srt_path = srt_path
            self.update_srt_display()
            print(f"📝 Manual SRT set: {os.path.basename(srt_path)}")
    
    def update_srt_display(self):
        """Update SRT label with current srt_path"""
        if self.queue_item.srt_path:
            srt_name = os.path.basename(self.queue_item.srt_path)
            self.srt_label.setText(f"SRT: {srt_name}")
            self.srt_label.setStyleSheet("font-size: 9px; color: #28a745;")  # Green
        else:
            self.srt_label.setText("SRT: (not set)")
            self.srt_label.setStyleSheet("font-size: 9px; color: #dc3545;")  # Red

    def on_stt(self):
        """Run Speech-to-Text to generate SRT from video audio using WhisperX"""
        self.btn_stt.setEnabled(False)
        self.btn_stt.setText("⏳...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("🎤 STT: Đang tách phụ đề...")
        self.status_label.setStyleSheet("padding: 5px; background: #e8d5f5; border-radius: 3px; font-size: 10px;")

        self._stt_worker = STTWorker(
            video_path=self.queue_item.video_path,
            language="en",
            model="large-v3"
        )
        self._stt_worker.progress.connect(lambda v: self.progress_bar.setValue(v))
        self._stt_worker.finished.connect(self._on_stt_finished)
        self._stt_worker.error.connect(self._on_stt_error)
        self._stt_worker.start()

    def _on_stt_finished(self, srt_path):
        """Handle STT completion"""
        self.queue_item.srt_path = srt_path
        self.update_srt_display()
        self.btn_stt.setEnabled(True)
        self.btn_stt.setText("🎤 STT")
        self.progress_bar.setVisible(False)
        self.status_label.setText("✅ STT Done!")
        self.status_label.setStyleSheet("padding: 5px; background: #d4edda; border-radius: 3px; font-size: 10px;")
        print(f"✅ STT finished: {os.path.basename(srt_path)}")

    def _on_stt_error(self, error_msg):
        """Handle STT error"""
        self.btn_stt.setEnabled(True)
        self.btn_stt.setText("🎤 STT")
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ STT Error")
        self.status_label.setStyleSheet("padding: 5px; background: #f8d7da; border-radius: 3px; font-size: 10px;")
        print(f"❌ STT Error: {error_msg}")
        QtWidgets.QMessageBox.warning(self, "STT Error", f"Speech-to-Text failed:\n{error_msg}")
    
    def on_per_video_sub_settings(self):
        """Open quick per-video subtitle settings dialog"""
        from srt_to_video import load_subtitle_settings
        
        # Load current settings (global or per-video override)
        current = load_subtitle_settings()
        if self.queue_item.sub_overrides:
            current.update(self.queue_item.sub_overrides)
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"🎨 Subtitle - {os.path.basename(self.queue_item.video_path)}")
        dialog.setMinimumWidth(350)
        form = QtWidgets.QFormLayout(dialog)
        
        # Font Size
        spin_fontsize = QtWidgets.QSpinBox()
        spin_fontsize.setRange(10, 300)
        spin_fontsize.setValue(current.get('font_size', 24))
        spin_fontsize.setSuffix(" px")
        form.addRow("Font Size:", spin_fontsize)
        
        # Font Name 
        edit_fontname = QtWidgets.QLineEdit(current.get('font_name', 'UTM-IMPACT'))
        form.addRow("Font Name:", edit_fontname)
        
        # Font Color
        edit_fontcolor = QtWidgets.QLineEdit(current.get('font_color', '#FFFFFF'))
        form.addRow("Font Color:", edit_fontcolor)
        
        # Border Width
        spin_border = QtWidgets.QSpinBox()
        spin_border.setRange(0, 20)
        spin_border.setValue(current.get('border_width', 2))
        form.addRow("Border Width:", spin_border)
        
        # MarginV (bottom margin)
        spin_margin = QtWidgets.QSpinBox()
        spin_margin.setRange(0, 500)
        spin_margin.setValue(current.get('position_v', 60))
        spin_margin.setSuffix(" px")
        form.addRow("Margin Bottom:", spin_margin)
        
        # Bold
        chk_bold = QtWidgets.QCheckBox()
        chk_bold.setChecked(current.get('bold', True))
        form.addRow("Bold:", chk_bold)
        
        # Background
        chk_bg = QtWidgets.QCheckBox()
        chk_bg.setChecked(current.get('bg_enabled', True))
        form.addRow("Background:", chk_bg)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        btn_reset = QtWidgets.QPushButton("🔄 Reset (Dùng Global)")
        btn_reset.setStyleSheet("background: #dc3545; color: white;")
        def on_reset():
            self.queue_item.sub_overrides = None
            self.update_sub_btn_style()
            dialog.accept()
        btn_reset.clicked.connect(on_reset)
        btn_layout.addWidget(btn_reset)
        
        btn_ok = QtWidgets.QPushButton("✅ Áp dụng")
        btn_ok.setStyleSheet("background: #28a745; color: white; font-weight: bold;")
        def on_apply():
            self.queue_item.sub_overrides = {
                'font_size': spin_fontsize.value(),
                'font_name': edit_fontname.text(),
                'font_color': edit_fontcolor.text(),
                'border_width': spin_border.value(),
                'position_v': spin_margin.value(),
                'bold': chk_bold.isChecked(),
                'bg_enabled': chk_bg.isChecked(),
            }
            self.update_sub_btn_style()
            dialog.accept()
        btn_ok.clicked.connect(on_apply)
        btn_layout.addWidget(btn_ok)
        
        form.addRow(btn_layout)
        dialog.exec_()
    
    def update_sub_btn_style(self):
        """Update 🎨 button style to indicate per-video settings"""
        if self.queue_item.sub_overrides:
            self.btn_sub_per_video.setStyleSheet("background: #ff9800; border: 2px solid #e65100; font-size: 14px;")
            self.btn_sub_per_video.setToolTip("Per-video Subtitle (CUSTOM ✅)")
        else:
            self.btn_sub_per_video.setStyleSheet("")
            self.btn_sub_per_video.setToolTip("Per-video Subtitle Settings")

    def on_voice_mapping(self):
        video_dir = os.path.dirname(self.queue_item.video_path)
        video_base = os.path.splitext(os.path.basename(self.queue_item.video_path))[0]
        base_without_nosub = video_base.replace("_nosub", "")
        
        json_candidates = [
            f"{base_without_nosub}_speaker.json",
            f"{video_base}_speaker.json",
            f"{base_without_nosub}_raw_gpt_speaker.json"
        ]
        
        json_path = None
        for cand in json_candidates:
            path = os.path.join(video_dir, cand)
            if os.path.exists(path):
                json_path = path
                break
                
        if not json_path:
            json_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Chọn file JSON chứa Speaker Mapping", video_dir, "JSON Files (*.json)"
            )
            
        if json_path:
            self.queue_item.auto_detect_srt() # ensure srt is detected
            dialog = VoiceMappingDialog(self.queue_item, json_path, self)
            dialog.exec_()
    
    def cleanup(self):
        if self.cap:
            self.cap.release()



class VideoPreview(QtWidgets.QLabel):
    """Video preview with click & drag ROI selection"""
    roi_changed = QtCore.pyqtSignal(int, int, int, int)
    
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._pix = None
        self.roi = None
        
        self.selecting = False
        self.start_point = None
        self.current_point = None
        
        self.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMouseTracking(True)
    
    def set_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._pix:
            return
        
        p = QtGui.QPainter(self)
        
        pix = self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        x_offset = (self.width() - pix.width()) // 2
        y_offset = (self.height() - pix.height()) // 2
        p.drawPixmap(x_offset, y_offset, pix)
        
        if self.roi:
            scale = pix.width() / self._pix.width()
            rx = int(self.roi[0] * scale) + x_offset
            ry = int(self.roi[1] * scale) + y_offset
            rw = int(self.roi[2] * scale)
            rh = int(self.roi[3] * scale)
            
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
            p.setPen(pen)
            p.drawRect(rx, ry, rw, rh)
        
        if self.selecting and self.start_point and self.current_point:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)
            p.setPen(pen)
            rect = QtCore.QRect(self.start_point, self.current_point).normalized()
            p.drawRect(rect)
        
        p.end()
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._pix:
            self.selecting = True
            self.start_point = event.pos()
            self.current_point = event.pos()
    
    def mouseMoveEvent(self, event):
        if self.selecting:
            self.current_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.selecting:
            self.selecting = False
            
            if self.start_point and self.current_point and self._pix:
                pix = self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                x_offset = (self.width() - pix.width()) // 2
                y_offset = (self.height() - pix.height()) // 2
                
                scale = self._pix.width() / pix.width()
                
                x1 = int((self.start_point.x() - x_offset) * scale)
                y1 = int((self.start_point.y() - y_offset) * scale)
                x2 = int((self.current_point.x() - x_offset) * scale)
                y2 = int((self.current_point.y() - y_offset) * scale)
                
                x = max(0, min(x1, x2))
                y = max(0, min(y1, y2))
                w = min(self._pix.width() - x, abs(x2 - x1))
                h = min(self._pix.height() - y, abs(y2 - y1))
                
                if w > 10 and h > 10:
                    self.roi = [x, y, w, h]
                    self.roi_changed.emit(x, y, w, h)
                    self.update()
            
            self.start_point = None
            self.current_point = None

class NoScrollFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel:
            if not obj.hasFocus():
                parent = obj.parentWidget()
                if parent:
                    QtWidgets.QApplication.sendEvent(parent, event)
                return True
        return super().eventFilter(obj, event)


class SettingsDialog(QtWidgets.QDialog):
    """Settings dialog for batch processing - with all OCR params"""
    
    def __init__(self, settings=None, current_mode=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Settings")
        self.resize(1000, 600)
        self.settings = settings or {}
        self.current_mode = current_mode or ProcessMode.SRT_REMOVESUB
        self.init_ui()
        
        # Prevent accidental scrolling on inputs
        self.scroll_filter = NoScrollFilter(self)
        for widget in self.findChildren((QtWidgets.QComboBox, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QSlider)):
            widget.setFocusPolicy(QtCore.Qt.StrongFocus)
            widget.installEventFilter(self.scroll_filter)
    
    def init_ui(self):
        outer_layout = QtWidgets.QVBoxLayout()
        
        # Scroll area to prevent overflow
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        
        # Target Language Settings
        lang_group = QtWidgets.QGroupBox("Target Language")
        lang_form = QtWidgets.QFormLayout()
        
        self.cmbTargetLang = QtWidgets.QComboBox()
        self.cmbTargetLang.addItems(["Tiếng Việt", "English"])
        self.cmbTargetLang.setCurrentText(self.settings.get("target_lang", "Tiếng Việt"))
        lang_form.addRow("Language:", self.cmbTargetLang)
        lang_group.setLayout(lang_form)
        layout.addWidget(lang_group)
        
        # Mode selection
        mode_group = QtWidgets.QGroupBox("Processing Mode")
        mode_layout = QtWidgets.QVBoxLayout()
        
        self.radio_srt_only = QtWidgets.QRadioButton("📝 SRT Only (Chỉ OCR, xuất SRT/TXT)")
        mode_layout.addWidget(self.radio_srt_only)
        
        self.radio_removesub = QtWidgets.QRadioButton("🧼 SRT + RemoveSub (OCR → SRT + Clean video)")
        mode_layout.addWidget(self.radio_removesub)
        
        self.radio_tts_merge = QtWidgets.QRadioButton("🎬 TTS + Merge (SRT → Audio + Burn subtitles)")
        mode_layout.addWidget(self.radio_tts_merge)
        
        self.radio_tts_merge_multi = QtWidgets.QRadioButton("🌍 Multi-Lang TTS + Merge (Tự nhận diện _hi, _ar...)")
        mode_layout.addWidget(self.radio_tts_merge_multi)
        
        self.radio_burn_only = QtWidgets.QRadioButton("🔥 Burn Subtitles Only (Chỉ dán Sub, KHÔNG lồng tiếng)")
        mode_layout.addWidget(self.radio_burn_only)
        
        self.radio_remove_logo = QtWidgets.QRadioButton("🚫 Remove Logo (Xóa logo/watermark - Không cần OCR)")
        mode_layout.addWidget(self.radio_remove_logo)
        
        self.radio_stt = QtWidgets.QRadioButton("🎤 Speech-to-Text (Tách SRT từ giọng nói - WhisperX)")
        mode_layout.addWidget(self.radio_stt)
        
        # Set the correct radio button based on current mode
        if self.current_mode == ProcessMode.TTS_MERGE:
            self.radio_tts_merge.setChecked(True)
        elif self.current_mode == ProcessMode.TTS_MERGE_MULTI:
            self.radio_tts_merge_multi.setChecked(True)
        elif self.current_mode == ProcessMode.BURN_ONLY:
            self.radio_burn_only.setChecked(True)
        elif self.current_mode == ProcessMode.SRT_ONLY:
            self.radio_srt_only.setChecked(True)
        elif self.current_mode == ProcessMode.REMOVE_LOGO:
            self.radio_remove_logo.setChecked(True)
        elif self.current_mode == ProcessMode.STT_EXTRACT:
            self.radio_stt.setChecked(True)
        else:
            self.radio_removesub.setChecked(True)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # OCR Settings
        ocr_group = QtWidgets.QGroupBox("OCR Settings")
        form = QtWidgets.QFormLayout()
        
        self.cmbVSEMode = QtWidgets.QComboBox()
        self.cmbVSEMode.addItems(["fast", "auto", "accurate", "ultra"])
        self.cmbVSEMode.setCurrentText(self.settings.get("vse_mode", "ultra"))
        form.addRow("VSE Extractor Mode:", self.cmbVSEMode)
        
        self.lblVseInfo = QtWidgets.QLabel("fast=nhanh | auto=cân bằng | accurate=chính xác | ultra=quét mọi frame (chậm nhất, không sót)")
        self.lblVseInfo.setStyleSheet("color: #666; font-style: italic;")
        self.lblVseInfo.setWordWrap(True)
        form.addRow("", self.lblVseInfo)
        
        ocr_group.setLayout(form)
        layout.addWidget(ocr_group)
        
        # TTS Settings (for TTS+Merge mode)
        tts_group = QtWidgets.QGroupBox("TTS Settings (for TTS+Merge mode)")
        tts_form = QtWidgets.QFormLayout()
        
        self.txtApiUrl = QtWidgets.QLineEdit()
        self.txtApiUrl.setText(self.settings.get("tts_api_url", "http://192.168.1.75:8002/api2/convert-tts"))
        self.txtApiUrl.setPlaceholderText("Gõ 'edge', 'kokoro', 'minimax', 'piper', 'native', hoặc nhập HTTP URL")
        self.txtApiUrl.setToolTip("Gõ 'edge' (Microsoft Edge TTS - miễn phí), 'kokoro' (Kokoro EN), 'minimax' (MiniMax Cloud), 'piper'/'native' (Tiếng Việt), hoặc HTTP URL.")
        tts_form.addRow("TTS API URL:", self.txtApiUrl)
        
        self.cmbKokoroVoice = QtWidgets.QComboBox()
        self.cmbKokoroVoice.addItems([
            "af_heart (Nữ Mỹ - Heart)",
            "af_bella (Nữ Mỹ - Bella)",
            "af_nicole (Nữ Mỹ - Nicole)",
            "am_michael (Nam Mỹ - Michael)",
            "am_adam (Nam Mỹ - Adam)",
            "bm_george (Nam Anh - George)"
        ])
        saved_voice = self.settings.get("kokoro_voice", "af_bella")
        voice_keys = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "bm_george"]
        if saved_voice in voice_keys:
            self.cmbKokoroVoice.setCurrentIndex(voice_keys.index(saved_voice))
            
        self.kokoroVoiceWidget = QtWidgets.QWidget()
        kv_layout = QtWidgets.QHBoxLayout(self.kokoroVoiceWidget)
        kv_layout.setContentsMargins(0, 0, 0, 0)
        kv_layout.addWidget(self.cmbKokoroVoice)
        
        btn_play_kokoro = QtWidgets.QPushButton("▶️")
        btn_play_kokoro.setMaximumWidth(30)
        btn_play_kokoro.setToolTip("Nghe thử giọng mẫu")
        btn_play_kokoro.clicked.connect(lambda: self.play_voice_preview(self.cmbKokoroVoice.currentText().split(' ')[0], "kokoro"))
        kv_layout.addWidget(btn_play_kokoro)
        
        tts_form.addRow("🇬🇧 Kokoro Voice (English):", self.kokoroVoiceWidget)
        
        # --- MiniMax Voice Settings ---
        try:
            from minimax_bridge import get_minimax_voices
            mm_voices = get_minimax_voices()
        except Exception:
            mm_voices = {}
        
        self.cmbMinimaxLang = QtWidgets.QComboBox()
        self.cmbMinimaxVoice = QtWidgets.QComboBox()
        
        for lang in mm_voices.keys():
            self.cmbMinimaxLang.addItem(lang)
            
        def on_mm_lang_changed(lang):
            self.cmbMinimaxVoice.clear()
            self._mm_voice_ids = []
            if lang in mm_voices:
                for v in mm_voices[lang]:
                    if isinstance(v, dict) and "id" in v:
                        display = f"[{lang}] {v['name']} - {v.get('note', '')} ({v['id']})"
                        self.cmbMinimaxVoice.addItem(display, v['id'])
                        self._mm_voice_ids.append(v['id'])
                        
        self.cmbMinimaxLang.currentTextChanged.connect(on_mm_lang_changed)
        
        saved_mm_voice = self.settings.get("minimax_voice", "Vietnamese_Serene_Man")
        target_lang = self.settings.get("minimax_lang", "Tiếng Việt")
        # Auto-detect lang from saved voice
        for lang, voice_list in mm_voices.items():
            for v in voice_list:
                if isinstance(v, dict) and v.get("id") == saved_mm_voice:
                    target_lang = lang
                    break
                    
        lang_idx = self.cmbMinimaxLang.findText(target_lang)
        if lang_idx >= 0:
            self.cmbMinimaxLang.setCurrentIndex(lang_idx)
            
        on_mm_lang_changed(target_lang)
        
        mm_idx = self.cmbMinimaxVoice.findData(saved_mm_voice)
        if mm_idx >= 0:
            self.cmbMinimaxVoice.setCurrentIndex(mm_idx)
            
        self.mmVoiceWidget = QtWidgets.QWidget()
        mm_layout = QtWidgets.QHBoxLayout(self.mmVoiceWidget)
        mm_layout.setContentsMargins(0, 0, 0, 0)
        mm_layout.addWidget(self.cmbMinimaxVoice)
        
        btn_play_mm = QtWidgets.QPushButton("▶️")
        btn_play_mm.setMaximumWidth(30)
        btn_play_mm.setToolTip("Nghe thử giọng mẫu")
        btn_play_mm.clicked.connect(lambda: self.play_voice_preview(self.cmbMinimaxVoice.currentData(), "minimax"))
        mm_layout.addWidget(btn_play_mm)
            
        tts_form.addRow("🌍 MiniMax Language:", self.cmbMinimaxLang)
        tts_form.addRow("🌐 MiniMax Voice:", self.mmVoiceWidget)
        
        self.cmbMinimaxModel = QtWidgets.QComboBox()
        self.cmbMinimaxModel.addItems([
            "speech-2.8-hd",
            "speech-2.8-turbo",
            "speech-02-hd",
            "speech-02-turbo",
            "speech-2.6-hd"
        ])
        saved_mm_model = self.settings.get("minimax_model", "speech-2.8-hd")
        mm_model_idx = self.cmbMinimaxModel.findText(saved_mm_model)
        if mm_model_idx >= 0:
            self.cmbMinimaxModel.setCurrentIndex(mm_model_idx)
        tts_form.addRow("🤖 MiniMax Model:", self.cmbMinimaxModel)
        
        # --- Edge TTS Voice Settings ---
        try:
            from edge_tts_bridge import get_edge_tts_voices
            edge_voices = get_edge_tts_voices()
        except Exception:
            edge_voices = {}
        
        self.cmbEdgeLang = QtWidgets.QComboBox()
        self.cmbEdgeVoice = QtWidgets.QComboBox()
        
        for lang in edge_voices.keys():
            self.cmbEdgeLang.addItem(lang)
        
        def on_edge_lang_changed(lang):
            self.cmbEdgeVoice.clear()
            if lang in edge_voices:
                for v in edge_voices[lang]:
                    if isinstance(v, dict) and "id" in v:
                        display = f"{v['name']} ({v['id']})"
                        self.cmbEdgeVoice.addItem(display, v['id'])
        
        self.cmbEdgeLang.currentTextChanged.connect(on_edge_lang_changed)
        
        saved_edge_voice = self.settings.get("edge_voice", "vi-VN-HoaiMyNeural")
        edge_target_lang = "Tiếng Việt"
        for lang, voice_list in edge_voices.items():
            for v in voice_list:
                if isinstance(v, dict) and v.get("id") == saved_edge_voice:
                    edge_target_lang = lang
                    break
        
        edge_lang_idx = self.cmbEdgeLang.findText(edge_target_lang)
        if edge_lang_idx >= 0:
            self.cmbEdgeLang.setCurrentIndex(edge_lang_idx)
        on_edge_lang_changed(edge_target_lang)
        
        edge_voice_idx = self.cmbEdgeVoice.findData(saved_edge_voice)
        if edge_voice_idx >= 0:
            self.cmbEdgeVoice.setCurrentIndex(edge_voice_idx)
        
        self.edgeVoiceWidget = QtWidgets.QWidget()
        edge_layout = QtWidgets.QHBoxLayout(self.edgeVoiceWidget)
        edge_layout.setContentsMargins(0, 0, 0, 0)
        edge_layout.addWidget(self.cmbEdgeVoice)
        
        btn_play_edge = QtWidgets.QPushButton("▶️")
        btn_play_edge.setMaximumWidth(30)
        btn_play_edge.setToolTip("Nghe thử giọng mẫu")
        btn_play_edge.clicked.connect(lambda: self.play_voice_preview(self.cmbEdgeVoice.currentData(), "edge"))
        edge_layout.addWidget(btn_play_edge)
        
        tts_form.addRow("🆓 Edge TTS Language:", self.cmbEdgeLang)
        tts_form.addRow("🆓 Edge TTS Voice:", self.edgeVoiceWidget)
        
        self.dblTtsSpeed = QtWidgets.QDoubleSpinBox()
        self.dblTtsSpeed.setRange(0.5, 3.0)
        self.dblTtsSpeed.setSingleStep(0.1)
        self.dblTtsSpeed.setValue(self.settings.get("tts_speed", 1.0))
        tts_form.addRow("TTS Speed:", self.dblTtsSpeed)
        
        self.dblOrigVideoSpeed = QtWidgets.QDoubleSpinBox()
        self.dblOrigVideoSpeed.setRange(0.5, 2.0)
        self.dblOrigVideoSpeed.setSingleStep(0.1)
        self.dblOrigVideoSpeed.setValue(self.settings.get("orig_video_speed", 1.0))
        self.dblOrigVideoSpeed.setToolTip("Giảm xuống < 1.0 để làm chậm video gốc, tạo khoảng trống cho sub dài.")
        tts_form.addRow("Original Video Speed (0.8 = Slow):", self.dblOrigVideoSpeed)
        
        self.chkRemoveVocal = QtWidgets.QCheckBox("🎙 Bật Khử Giọng Gốc (UVR5) - Phục hồi Âm nhạc/BG")
        self.chkRemoveVocal.setChecked(self.settings.get("uvr_vocal_remove", False))
        tts_form.addRow("", self.chkRemoveVocal)
        
        self.sldUvrStrength = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldUvrStrength.setRange(0, 100)
        self.sldUvrStrength.setValue(int(self.settings.get("uvr_strength", 100)))
        self.sldUvrStrength.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sldUvrStrength.setTickInterval(10)
        self.lblUvrStrength = QtWidgets.QLabel(f"Mức khử giọng: {self.sldUvrStrength.value()}%")
        self.lblUvrStrength.setStyleSheet("color: #666; font-style: italic;")
        self.sldUvrStrength.valueChanged.connect(lambda v: self.lblUvrStrength.setText(f"Mức khử giọng: {v}%  (100%=câm bặt, 0%=giữ nguyên)"))
        tts_form.addRow(self.lblUvrStrength, self.sldUvrStrength)
        
        # Instrument Music Volume (from UVR vocal removal)
        self.dblVolBgm = QtWidgets.QDoubleSpinBox()
        self.dblVolBgm.setRange(0.0, 5.0)
        self.dblVolBgm.setSingleStep(0.1)
        self.dblVolBgm.setValue(self.settings.get("vol_bgm", 2.0))
        self.dblVolBgm.setToolTip("Âm lượng nhạc nền Instrumental (từ UVR tách giọng). 1.0 = nguyên, 2.0 = gấp đôi.")
        tts_form.addRow("🎹 Instrument Music Vol:", self.dblVolBgm)

        # --- Background Music (Import) ---
        self.lstBgmFiles = QtWidgets.QListWidget()
        self.lstBgmFiles.setMaximumHeight(80)
        self.lstBgmFiles.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        saved_bgm = self.settings.get("bgm_files", [])
        if isinstance(saved_bgm, str) and saved_bgm:
            saved_bgm = [saved_bgm]
        for fp in (saved_bgm or []):
            if os.path.exists(fp):
                self.lstBgmFiles.addItem(fp)
        
        bgm_btn_layout = QtWidgets.QHBoxLayout()
        btn_add_bgm = QtWidgets.QPushButton("➕ Thêm nhạc")
        btn_add_bgm.clicked.connect(self._add_bgm_files)
        btn_remove_bgm = QtWidgets.QPushButton("🗑️ Xóa chọn")
        btn_remove_bgm.clicked.connect(self._remove_bgm_files)
        bgm_btn_layout.addWidget(btn_add_bgm)
        bgm_btn_layout.addWidget(btn_remove_bgm)
        
        self.chkRandomBgm = QtWidgets.QCheckBox("🔀 Random nhạc nền")
        self.chkRandomBgm.setChecked(bool(self.settings.get("bgm_random", True)))
        bgm_btn_layout.addWidget(self.chkRandomBgm)
        bgm_btn_layout.addStretch()
        
        bgm_container = QtWidgets.QVBoxLayout()
        bgm_container.addWidget(self.lstBgmFiles)
        bgm_container.addLayout(bgm_btn_layout)
        tts_form.addRow("🎵 Background Music (Import):", bgm_container)
        
        # Background Music Import Volume %
        self.sldVolBgmImport = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldVolBgmImport.setRange(0, 200)
        self.sldVolBgmImport.setValue(int(self.settings.get("vol_bgm_import", 1.0) * 100))
        self.sldVolBgmImport.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sldVolBgmImport.setTickInterval(25)
        self.lblVolBgmImport = QtWidgets.QLabel(f"BG Music Vol: {self.sldVolBgmImport.value()}%")
        self.lblVolBgmImport.setStyleSheet("color: #666; font-style: italic;")
        self.sldVolBgmImport.valueChanged.connect(lambda v: self.lblVolBgmImport.setText(f"BG Music Vol: {v}%"))
        tts_form.addRow(self.lblVolBgmImport, self.sldVolBgmImport)
        
        self.cmbSyncMode = QtWidgets.QComboBox()
        self.cmbSyncMode.addItems([
            "Kéo giãn Video (Slo-mo video chờ Audio - Khuyên dùng)",
            "Xếp hàng Audio nối đuôi nhau (Giữ tốc độ Video nguyên bản)",
            "Chèn đè Audio chồng lên nhau (Giữ tốc độ Video nguyên bản)"
        ])
        
        # Backward compatibility for dynamic_retime = boolean
        old_val = self.settings.get("sync_mode", self.settings.get("dynamic_retime", True))
        if old_val is True or old_val == "stretch_video":
            self.cmbSyncMode.setCurrentIndex(0)
        elif old_val == "queue_audio":
            self.cmbSyncMode.setCurrentIndex(1)
        else: # overlap or False
            self.cmbSyncMode.setCurrentIndex(2)
            
        tts_form.addRow("✂️ Cách xử lý Audio dài:", self.cmbSyncMode)
        
        self.dblVolTts = QtWidgets.QDoubleSpinBox()
        self.dblVolTts.setRange(0.0, 5.0)
        self.dblVolTts.setSingleStep(0.1)
        self.dblVolTts.setValue(self.settings.get("vol_tts", 2.0))
        self.dblVolTts.setToolTip("Âm lượng giọng đọc TTS. 1.0 = nguyên, 2.0 = gấp đôi.")
        tts_form.addRow("🎙 Volume Giọng Đọc (TTS):", self.dblVolTts)
        
        def update_tts_ui():
            url = self.txtApiUrl.text().strip().lower()
            is_minimax = url.startswith("minimax")
            is_kokoro = url.startswith("kokoro")
            
            def set_field_visible(field, visible):
                field.setVisible(visible)
                label = tts_form.labelForField(field)
                if label:
                    label.setVisible(visible)
                    
            if is_minimax:
                set_field_visible(self.kokoroVoiceWidget, False)
                set_field_visible(self.cmbMinimaxLang, True)
                set_field_visible(self.mmVoiceWidget, True)
                set_field_visible(self.cmbMinimaxModel, True)
            elif is_kokoro:
                set_field_visible(self.kokoroVoiceWidget, True)
                set_field_visible(self.cmbMinimaxLang, False)
                set_field_visible(self.mmVoiceWidget, False)
                set_field_visible(self.cmbMinimaxModel, False)
            else:
                set_field_visible(self.kokoroVoiceWidget, True)
                set_field_visible(self.cmbMinimaxLang, True)
                set_field_visible(self.mmVoiceWidget, True)
                set_field_visible(self.cmbMinimaxModel, True)

        self.txtApiUrl.textChanged.connect(lambda text: update_tts_ui())
        update_tts_ui()
        
        tts_group.setLayout(tts_form)
        layout.addWidget(tts_group)
        

        # Prompt Settings
        prompt_group = QtWidgets.QGroupBox("Prompt Template (for TXT output)")
        prompt_layout = QtWidgets.QVBoxLayout()
        
        self.txtPrompt = QtWidgets.QPlainTextEdit()
        self.txtPrompt.setPlaceholderText("Enter prompt template here...")
        
        def update_default_prompt():
            if self.cmbTargetLang.currentText() == "English":
                return "You are a professional translator. Please translate the following SRT into English: "
            return "Bạn là một chuyên gia dịch thuật hãy dịch cho tôi đoạn srt sau qua tiếng việt : "

        default_prompt = update_default_prompt()
        self.txtPrompt.setPlainText(self.settings.get("ocr_prompt", default_prompt))
        self.txtPrompt.setFixedHeight(80)
        
        def on_lang_changed(text):
            # If user hasn't modified the default prompt of the old language, update it
            current_text = self.txtPrompt.toPlainText().strip()
            old_vi = "Bạn là một chuyên gia dịch thuật hãy dịch cho tôi đoạn srt sau qua tiếng việt :"
            old_en = "You are a professional translator. Please translate the following SRT into English:"
            if current_text == old_vi or current_text == old_en:
                self.txtPrompt.setPlainText(update_default_prompt())
                
            # Auto switch TTS URL to kokoro for English if using native/piper
            if text == "English":
                current_tts = self.txtApiUrl.text().strip().lower()
                if current_tts in ["native", "piper", "chumtts", ""]:
                    self.txtApiUrl.setText("kokoro")
            elif text == "Tiếng Việt":
                current_tts = self.txtApiUrl.text().strip().lower()
                if current_tts == "kokoro":
                    self.txtApiUrl.setText("native")
                
        self.cmbTargetLang.currentTextChanged.connect(on_lang_changed)
        
        prompt_layout.addWidget(self.txtPrompt)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # RemoveSub Method Settings (AI vs FFmpeg Delogo)
        removesub_group = QtWidgets.QGroupBox("RemoveSub Method")
        removesub_layout = QtWidgets.QHBoxLayout()
        
        self.radio_ai_inpaint = QtWidgets.QRadioButton("🤖 AI Inpainting (Slow, Best Quality)")
        self.radio_delogo = QtWidgets.QRadioButton("⚡ FFmpeg Delogo (Instant, Blurred)")
        
        method = self.settings.get("removesub_method", "ai")
        if method == "delogo":
            self.radio_delogo.setChecked(True)
        else:
            self.radio_ai_inpaint.setChecked(True)
            
        removesub_layout.addWidget(self.radio_ai_inpaint)
        removesub_layout.addWidget(self.radio_delogo)
        removesub_group.setLayout(removesub_layout)
        layout.addWidget(removesub_group)

        layout.addStretch()
        
        # End of scrollable content
        outer_layout.addWidget(scroll)

        # Profile Management — simple: dropdown auto-loads, + to save, 🗑️ to delete
        profile_frame = QtWidgets.QFrame()
        profile_frame.setStyleSheet("background: #f5f5f5; border: 1px solid #ccc; border-radius: 6px; padding: 6px;")
        profile_layout = QtWidgets.QHBoxLayout(profile_frame)
        profile_layout.setContentsMargins(10, 6, 10, 6)
        
        lbl = QtWidgets.QLabel("📋 Profile:")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px; border: none;")
        profile_layout.addWidget(lbl)
        
        self.cmb_profile = QtWidgets.QComboBox()
        self.cmb_profile.setMinimumWidth(220)
        self.cmb_profile.setStyleSheet("font-size: 13px; padding: 4px 8px;")
        self._refresh_profiles()
        self.cmb_profile.currentTextChanged.connect(self._on_profile_selected)
        profile_layout.addWidget(self.cmb_profile, 1)
        
        btn_add = QtWidgets.QPushButton("➕")
        btn_add.setFixedSize(36, 36)
        btn_add.setStyleSheet("font-size: 16px; background: #28a745; color: white; border-radius: 4px; font-weight: bold; border: none;")
        btn_add.setToolTip("Lưu settings hiện tại thành profile mới")
        btn_add.clicked.connect(self._on_add_profile)
        profile_layout.addWidget(btn_add)
        
        btn_del = QtWidgets.QPushButton("🗑️")
        btn_del.setFixedSize(36, 36)
        btn_del.setStyleSheet("font-size: 16px; background: #dc3545; color: white; border-radius: 4px; border: none;")
        btn_del.setToolTip("Xóa profile đã chọn")
        btn_del.clicked.connect(self._on_delete_profile)
        profile_layout.addWidget(btn_del)
        
        outer_layout.addWidget(profile_frame)
        
        # OK / Cancel Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("✅ OK")
        btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(btn_ok)
        
        btn_cancel = QtWidgets.QPushButton("❌ Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        
        outer_layout.addLayout(btn_layout)
        self.setLayout(outer_layout)
    
    def _add_bgm_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Chọn file nhạc nền", "",
            "Audio (*.mp3 *.wav *.m4a *.aac *.ogg *.flac)"
        )
        for fp in files:
            # Avoid duplicates
            existing = [self.lstBgmFiles.item(i).text() for i in range(self.lstBgmFiles.count())]
            if fp not in existing:
                self.lstBgmFiles.addItem(fp)
    
    def _remove_bgm_files(self):
        for item in self.lstBgmFiles.selectedItems():
            self.lstBgmFiles.takeItem(self.lstBgmFiles.row(item))
        
    def _get_profiles_dir(self):
        """Get the profiles directory, creating it if needed"""
        profiles_dir = os.path.expanduser("~/.autosub_profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        return profiles_dir
    
    def _refresh_profiles(self):
        """Refresh the profile dropdown with saved profiles"""
        self.cmb_profile.blockSignals(True)
        current_text = self.cmb_profile.currentText() if self.cmb_profile.count() > 0 else ""
        self.cmb_profile.clear()
        
        profiles_dir = self._get_profiles_dir()
        profiles = sorted([
            os.path.splitext(f)[0] for f in os.listdir(profiles_dir)
            if f.endswith(".json")
        ])
        
        if profiles:
            self.cmb_profile.addItems(profiles)
            if current_text and current_text in profiles:
                self.cmb_profile.setCurrentText(current_text)
        
        self.cmb_profile.blockSignals(False)
    
    def _on_profile_selected(self, name):
        """Auto-load profile when selected from dropdown"""
        if not name:
            return
        
        import json
        profiles_dir = self._get_profiles_dir()
        file_path = os.path.join(profiles_dir, f"{name}.json")
        
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Apply data to UI
            if "vse_mode" in data: self.cmbVSEMode.setCurrentText(data["vse_mode"])
            if "target_lang" in data: self.cmbTargetLang.setCurrentText(data["target_lang"])
            if "tts_api_url" in data: self.txtApiUrl.setText(data["tts_api_url"])
            if "kokoro_voice" in data:
                idx = self.cmbKokoroVoice.findText(data["kokoro_voice"], QtCore.Qt.MatchContains)
                if idx >= 0: self.cmbKokoroVoice.setCurrentIndex(idx)
            if "tts_speed" in data: self.dblTtsSpeed.setValue(float(data["tts_speed"]))
            if "orig_video_speed" in data: self.dblOrigVideoSpeed.setValue(float(data["orig_video_speed"]))
            if "uvr_vocal_remove" in data: self.chkRemoveVocal.setChecked(bool(data["uvr_vocal_remove"]))
            if "uvr_strength" in data: self.sldUvrStrength.setValue(int(data["uvr_strength"]))
            if "bgm_files" in data:
                self.lstBgmFiles.clear()
                for fp in data["bgm_files"]:
                    if os.path.exists(fp):
                        self.lstBgmFiles.addItem(fp)
            if "bgm_random" in data: self.chkRandomBgm.setChecked(bool(data["bgm_random"]))
            if "vol_bgm" in data: self.dblVolBgm.setValue(float(data["vol_bgm"]))
            if "vol_bgm_import" in data: self.sldVolBgmImport.setValue(int(float(data["vol_bgm_import"]) * 100))
            if "vol_tts" in data: self.dblVolTts.setValue(float(data["vol_tts"]))
            if "minimax_voice" in data:
                mm_idx = self.cmbMinimaxVoice.findData(data["minimax_voice"])
                if mm_idx >= 0: self.cmbMinimaxVoice.setCurrentIndex(mm_idx)
            if "minimax_model" in data:
                mm_model_idx = self.cmbMinimaxModel.findText(data["minimax_model"])
                if mm_model_idx >= 0: self.cmbMinimaxModel.setCurrentIndex(mm_model_idx)

            if "ocr_prompt" in data: self.txtPrompt.setPlainText(data["ocr_prompt"])
            if "removesub_method" in data:
                if data["removesub_method"] == "delogo": self.radio_delogo.setChecked(True)
                else: self.radio_ai_inpaint.setChecked(True)
            
            # Restore processing mode
            if "_mode" in data:
                mode_map = {m.value: m for m in ProcessMode}
                mode = mode_map.get(data["_mode"])
                if mode:
                    if mode == ProcessMode.TTS_MERGE: self.radio_tts_merge.setChecked(True)
                    elif mode == ProcessMode.TTS_MERGE_MULTI: self.radio_tts_merge_multi.setChecked(True)
                    elif mode == ProcessMode.BURN_ONLY: self.radio_burn_only.setChecked(True)
                    elif mode == ProcessMode.SRT_ONLY: self.radio_srt_only.setChecked(True)
                    elif mode == ProcessMode.REMOVE_LOGO: self.radio_remove_logo.setChecked(True)
                    elif mode == ProcessMode.STT_EXTRACT: self.radio_stt.setChecked(True)
                    else: self.radio_removesub.setChecked(True)
            
            print(f"✅ Profile '{name}' loaded")
        except Exception as e:
            print(f"⚠️ Failed to load profile '{name}': {e}")
    
    def _on_add_profile(self):
        """Save current settings as a new named profile"""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "➕ Tạo Profile Mới",
            "Nhập tên profile:",
            QtWidgets.QLineEdit.Normal, ""
        )
        if not ok or not name.strip():
            return
        
        name = name.strip()
        import json
        profiles_dir = self._get_profiles_dir()
        file_path = os.path.join(profiles_dir, f"{name}.json")
        
        # Check overwrite
        if os.path.exists(file_path):
            reply = QtWidgets.QMessageBox.question(
                self, "Profile đã tồn tại",
                f"Profile '{name}' đã có. Ghi đè?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        
        settings_dict = self.get_settings()
        settings_dict["_mode"] = self.get_mode().value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4, ensure_ascii=False)
        
        self._refresh_profiles()
        self.cmb_profile.setCurrentText(name)
        QtWidgets.QMessageBox.information(self, "✅", f"Đã lưu profile '{name}'!")
    
    def _on_delete_profile(self):
        """Delete selected profile"""
        name = self.cmb_profile.currentText().strip()
        if not name:
            return
        
        profiles_dir = self._get_profiles_dir()
        file_path = os.path.join(profiles_dir, f"{name}.json")
        
        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(self, "⚠️", f"Profile '{name}' không tồn tại!")
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, "Xóa Profile",
            f"Xóa profile '{name}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            os.remove(file_path)
            self._refresh_profiles()
    
    def get_mode(self):
        if self.radio_srt_only.isChecked():
            result = ProcessMode.SRT_ONLY
        elif self.radio_removesub.isChecked():
            result = ProcessMode.SRT_REMOVESUB
        elif self.radio_remove_logo.isChecked():
            result = ProcessMode.REMOVE_LOGO
        elif self.radio_burn_only.isChecked():
            result = ProcessMode.BURN_ONLY
        elif self.radio_stt.isChecked():
            result = ProcessMode.STT_EXTRACT
        elif self.radio_tts_merge_multi.isChecked():
            result = ProcessMode.TTS_MERGE_MULTI
        elif self.radio_tts_merge.isChecked():
            result = ProcessMode.TTS_MERGE
        else:
            result = ProcessMode.TTS_MERGE
        print(f"\n📌 [get_mode] radio_tts_merge_multi.isChecked={self.radio_tts_merge_multi.isChecked()}, returning: {result}")
        return result
    
    def get_settings(self):
        bgm_files = [self.lstBgmFiles.item(i).text() for i in range(self.lstBgmFiles.count())]
        return {
            "vse_mode": self.cmbVSEMode.currentText(),
            "target_lang": self.cmbTargetLang.currentText(),
            # TTS settings
            "tts_api_url": self.txtApiUrl.text().strip(),
            "kokoro_voice": ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "bm_george"][self.cmbKokoroVoice.currentIndex()],
            "minimax_voice": self.cmbMinimaxVoice.currentData() or "Vietnamese_Serene_Man",
            "minimax_lang": self.cmbMinimaxLang.currentText(),
            "minimax_model": self.cmbMinimaxModel.currentText().strip(),
            "edge_voice": self.cmbEdgeVoice.currentData() or "vi-VN-HoaiMyNeural",
            "tts_speed": self.dblTtsSpeed.value(),
            "orig_video_speed": self.dblOrigVideoSpeed.value(),
            "uvr_vocal_remove": self.chkRemoveVocal.isChecked(),
            "uvr_strength": self.sldUvrStrength.value(),
            "bgm_files": bgm_files,
            "bgm_random": self.chkRandomBgm.isChecked(),
            "vol_bgm": self.dblVolBgm.value(),
            "vol_bgm_import": self.sldVolBgmImport.value() / 100.0,
            "sync_mode": ["stretch_video", "queue_audio", "overlap"][self.cmbSyncMode.currentIndex()],
            "vol_tts": self.dblVolTts.value(),
            # Prompt settings
            "ocr_prompt": self.txtPrompt.toPlainText().strip(),
            # RemoveSub settings
            "removesub_method": "delogo" if self.radio_delogo.isChecked() else "ai",
        }

    def play_voice_preview(self, voice_id, engine):
        from PyQt5.QtMultimedia import QSound
        import os
        
        if not voice_id:
            return
            
        if engine == "minimax":
            assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax", "assets")
            os.makedirs(assets_dir, exist_ok=True)
            preview_file = os.path.join(assets_dir, f"preview_{voice_id}.wav")
            
            if not os.path.exists(preview_file):
                lang_text_map = {
                    "Tiếng Việt": "Xin chào, đây là giọng đọc thử nghiệm của tôi.",
                    "Tiếng Anh": "Hello, this is a test voice for you to hear.",
                    "Tiếng Trung": "你好，这是我的测试语音。",
                    "Tiếng Hàn": "안녕하세요, 이것은 제 테스트 음성입니다.",
                    "Tiếng Nhật": "こんにちは、これは私のテスト音声です。",
                    "Tiếng Pháp": "Bonjour, ceci est ma voix de test.",
                    "Tiếng Tây Ban Nha": "Hola, esta es mi voz de prueba.",
                    "Tiếng Tamil": "வணக்கம், இது எனது சோதனை குரல்.",
                    "Tiếng Hindi": "नमस्ते, यह मेरी परीक्षण आवाज़ है।",
                    "Tiếng Ả Rập": "مرحباً، هذا هو صوتي التجريبي.",
                    "Tiếng Indonesia": "Halo, ini adalah suara tes saya.",
                    "Tiếng Ý": "Ciao, questa è la mia voce di prova."
                }
                current_lang = self.cmbMinimaxLang.currentText()
                text = lang_text_map.get(current_lang, "Xin chào, đây là giọng đọc thử nghiệm của tôi.")
                
                mm_model = self.cmbMinimaxModel.currentText().strip()
                
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    from minimax_bridge import minimax_generate_wav
                    minimax_generate_wav(text, 1.0, preview_file, voice_id=voice_id, emotion="neutral", model=mm_model)
                except Exception as e:
                    print(f"Error generating preview: {e}")
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
            if os.path.exists(preview_file):
                QSound.play(preview_file)
        elif engine == "edge":
            assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_previews")
            os.makedirs(assets_dir, exist_ok=True)
            preview_file = os.path.join(assets_dir, f"preview_{voice_id}.wav")
            
            if not os.path.exists(preview_file):
                lang_text_map = {
                    "Tiếng Việt": "Xin chào, đây là giọng đọc thử nghiệm của tôi.",
                    "English": "Hello, this is a test voice for you to hear.",
                    "Tiếng Trung": "你好，这是我的测试语音。",
                    "Tiếng Hàn": "안녕하세요, 이것은 제 테스트 음성입니다.",
                    "Tiếng Nhật": "こんにちは、これは私のテスト音声です。",
                    "Tiếng Pháp": "Bonjour, ceci est ma voix de test.",
                    "Tiếng Tây Ban Nha": "Hola, esta es mi voz de prueba.",
                    "Tiếng Tamil": "வணக்கம், இது எனது சோதனை குரல்.",
                    "Tiếng Hindi": "नमस्ते, यह मेरी परीक्षण आवाज़ है।",
                    "Tiếng Ả Rập": "مرحباً، هذا هو صوتي التجريبي.",
                    "Tiếng Indonesia": "Halo, ini adalah suara tes saya.",
                    "Tiếng Ý": "Ciao, questa è la mia voce di prova."
                }
                current_lang = self.cmbEdgeLang.currentText()
                text = lang_text_map.get(current_lang, "Hello, this is a test voice for you to hear.")
                
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    from edge_tts_bridge import edge_tts_generate_wav
                    edge_tts_generate_wav(text, 1.0, preview_file, voice=voice_id)
                except Exception as e:
                    print(f"Error generating edge preview: {e}")
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
            if os.path.exists(preview_file):
                QSound.play(preview_file)
        else: # kokoro
            preview_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro_previews", f"kokoro-{voice_id}.wav")
            if os.path.exists(preview_file):
                QSound.play(preview_file)


class LogoCanvas(QtWidgets.QWidget):
    """Canvas widget for drag & drop logo positioning on a video frame preview"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        
        self.bg_pixmap = None   # Video frame background
        self.logo_pixmap = None  # Original logo QPixmap
        self.logo_x = 15  # Logo top-left X in canvas coords
        self.logo_y = 15  # Logo top-left Y in canvas coords
        self.logo_w = 80  # Logo display width
        self.logo_h = 80  # Logo display height
        self.opacity = 0.8
        self._dragging = False
        self._drag_offset_x = 0
        self._drag_offset_y = 0
        
        # Video dimensions for aspect ratio
        self._video_w = 1920
        self._video_h = 1080
        self._draw_rect = QtCore.QRect(0, 0, 480, 270)  # Actual draw area (letterboxed)
    
    def set_video_frame(self, video_path):
        """Extract a frame from video and use as background"""
        if not video_path or not os.path.exists(video_path):
            return
        
        try:
            import tempfile
            tmp_img = os.path.join(tempfile.gettempdir(), "_logo_preview_frame.jpg")
            ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
            
            # Extract frame at 2 seconds
            subprocess.run([
                ffmpeg_bin, "-y", "-ss", "2", "-i", video_path,
                "-frames:v", "1", "-q:v", "3", tmp_img
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(tmp_img):
                self.bg_pixmap = QtGui.QPixmap(tmp_img)
                if not self.bg_pixmap.isNull():
                    self._video_w = self.bg_pixmap.width()
                    self._video_h = self.bg_pixmap.height()
                os.remove(tmp_img)
            
            self._update_draw_rect()
            self.update()
        except Exception as e:
            print(f"⚠️ Failed to extract video frame: {e}")
    
    def _update_draw_rect(self):
        """Calculate the letterboxed draw area maintaining video aspect ratio"""
        cw, ch = self.width(), self.height()
        video_aspect = self._video_w / max(self._video_h, 1)
        canvas_aspect = cw / max(ch, 1)
        
        if video_aspect > canvas_aspect:
            # Video is wider — fit to width, letterbox top/bottom
            draw_w = cw
            draw_h = int(cw / video_aspect)
            draw_x = 0
            draw_y = (ch - draw_h) // 2
        else:
            # Video is taller — fit to height, pillarbox left/right
            draw_h = ch
            draw_w = int(ch * video_aspect)
            draw_x = (cw - draw_w) // 2
            draw_y = 0
        
        self._draw_rect = QtCore.QRect(draw_x, draw_y, draw_w, draw_h)
    
    def set_logo(self, path):
        """Load a logo image from path"""
        if path and os.path.exists(path):
            self.logo_pixmap = QtGui.QPixmap(path)
            if not self.logo_pixmap.isNull():
                aspect = self.logo_pixmap.width() / max(self.logo_pixmap.height(), 1)
                self.logo_h = int(self.logo_w / aspect)
        else:
            self.logo_pixmap = None
        self.update()
    
    def set_size_pct(self, pct):
        """Set logo size as percentage of the draw area width"""
        draw_w = self._draw_rect.width()
        self.logo_w = max(10, int(draw_w * pct / 100))
        if self.logo_pixmap and not self.logo_pixmap.isNull():
            aspect = self.logo_pixmap.width() / max(self.logo_pixmap.height(), 1)
            self.logo_h = max(10, int(self.logo_w / aspect))
        else:
            self.logo_h = self.logo_w
        # Clamp position within draw area
        dr = self._draw_rect
        self.logo_x = max(dr.x(), min(self.logo_x, dr.right() - self.logo_w))
        self.logo_y = max(dr.y(), min(self.logo_y, dr.bottom() - self.logo_h))
        self.update()
    
    def snap_to(self, position):
        """Snap logo to a corner or center within the draw area"""
        m = 10  # margin in canvas pixels
        dr = self._draw_rect
        if position == "top-left":
            self.logo_x, self.logo_y = dr.x() + m, dr.y() + m
        elif position == "top-right":
            self.logo_x, self.logo_y = dr.right() - self.logo_w - m, dr.y() + m
        elif position == "bottom-left":
            self.logo_x, self.logo_y = dr.x() + m, dr.bottom() - self.logo_h - m
        elif position == "bottom-right":
            self.logo_x, self.logo_y = dr.right() - self.logo_w - m, dr.bottom() - self.logo_h - m
        elif position == "center":
            self.logo_x = dr.x() + (dr.width() - self.logo_w) // 2
            self.logo_y = dr.y() + (dr.height() - self.logo_h) // 2
        self.update()
    
    def get_position_ratios(self):
        """Return position as ratios (0.0-1.0) relative to the draw area (= video)"""
        dr = self._draw_rect
        dw, dh = max(dr.width(), 1), max(dr.height(), 1)
        return (
            (self.logo_x - dr.x()) / dw,
            (self.logo_y - dr.y()) / dh,
            self.logo_w / dw,
        )
    
    def resizeEvent(self, event):
        self._update_draw_rect()
        super().resizeEvent(event)
    
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Black letterbox background
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        
        dr = self._draw_rect
        
        # Draw video frame or dark placeholder
        if self.bg_pixmap and not self.bg_pixmap.isNull():
            scaled_bg = self.bg_pixmap.scaled(
                dr.width(), dr.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            p.drawPixmap(dr.x(), dr.y(), scaled_bg)
        else:
            p.fillRect(dr, QtGui.QColor(30, 30, 30))
            # Draw grid lines
            p.setPen(QtGui.QPen(QtGui.QColor(50, 50, 50), 1, QtCore.Qt.DashLine))
            p.drawLine(dr.x() + dr.width() // 3, dr.y(), dr.x() + dr.width() // 3, dr.bottom())
            p.drawLine(dr.x() + 2 * dr.width() // 3, dr.y(), dr.x() + 2 * dr.width() // 3, dr.bottom())
            p.drawLine(dr.x(), dr.y() + dr.height() // 3, dr.right(), dr.y() + dr.height() // 3)
            p.drawLine(dr.x(), dr.y() + 2 * dr.height() // 3, dr.right(), dr.y() + 2 * dr.height() // 3)
        
        # Draw logo
        if self.logo_pixmap and not self.logo_pixmap.isNull():
            p.setOpacity(self.opacity)
            scaled = self.logo_pixmap.scaled(
                self.logo_w, self.logo_h,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            p.drawPixmap(int(self.logo_x), int(self.logo_y), scaled)
            p.setOpacity(1.0)
            
            # Selection border
            p.setPen(QtGui.QPen(QtGui.QColor(255, 165, 0), 2, QtCore.Qt.DashLine))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(int(self.logo_x) - 1, int(self.logo_y) - 1, self.logo_w + 2, self.logo_h + 2)
        else:
            p.setPen(QtGui.QColor(100, 100, 100))
            p.drawText(dr, QtCore.Qt.AlignCenter, "Chưa chọn logo")
        
        p.end()
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.logo_pixmap:
            ex, ey = event.x(), event.y()
            if (self.logo_x <= ex <= self.logo_x + self.logo_w and
                self.logo_y <= ey <= self.logo_y + self.logo_h):
                self._dragging = True
                self._drag_offset_x = ex - self.logo_x
                self._drag_offset_y = ey - self.logo_y
                self.setCursor(QtCore.Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        if self._dragging:
            dr = self._draw_rect
            self.logo_x = max(dr.x(), min(event.x() - self._drag_offset_x, dr.right() - self.logo_w))
            self.logo_y = max(dr.y(), min(event.y() - self._drag_offset_y, dr.bottom() - self.logo_h))
            self.update()
    
    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.setCursor(QtCore.Qt.OpenHandCursor)


class LogoPositionDialog(QtWidgets.QDialog):
    """Visual dialog for positioning a logo on the video"""
    
    def __init__(self, logo_path="", video_path="", size_pct=15, opacity=80, x_ratio=None, y_ratio=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎯 Chỉnh vị trí Logo")
        self.resize(600, 520)
        self.result_data = None
        self._saved_x_ratio = x_ratio
        self._saved_y_ratio = y_ratio
        self._init_ui(logo_path, video_path, size_pct, opacity)
    
    def _init_ui(self, logo_path, video_path, size_pct, opacity):
        layout = QtWidgets.QVBoxLayout()
        
        # Canvas
        self.canvas = LogoCanvas()
        self.canvas.opacity = opacity / 100.0
        layout.addWidget(self.canvas, 1)
        
        # Info
        info = QtWidgets.QLabel("Kéo thả logo bằng chuột để đặt vị trí")
        info.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        info.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(info)
        
        # Size slider
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Kích thước:"))
        self.sld_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_size.setRange(3, 100)
        self.sld_size.setValue(size_pct)
        self.sld_size.valueChanged.connect(self._on_size_changed)
        size_row.addWidget(self.sld_size, 1)
        self.lbl_size = QtWidgets.QLabel(f"{size_pct}%")
        self.lbl_size.setMinimumWidth(40)
        size_row.addWidget(self.lbl_size)
        layout.addLayout(size_row)
        
        # Opacity slider
        opa_row = QtWidgets.QHBoxLayout()
        opa_row.addWidget(QtWidgets.QLabel("Độ trong suốt:"))
        self.sld_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_opacity.setRange(10, 100)
        self.sld_opacity.setValue(opacity)
        self.sld_opacity.valueChanged.connect(self._on_opacity_changed)
        opa_row.addWidget(self.sld_opacity, 1)
        self.lbl_opacity = QtWidgets.QLabel(f"{opacity}%")
        self.lbl_opacity.setMinimumWidth(40)
        opa_row.addWidget(self.lbl_opacity)
        layout.addLayout(opa_row)
        
        # Quick position buttons
        pos_row = QtWidgets.QHBoxLayout()
        pos_row.addWidget(QtWidgets.QLabel("Vị trí nhanh:"))
        for label, pos in [("⬆ Trên trái", "top-left"), ("⬆ Trên phải", "top-right"),
                           ("⬇ Dưới trái", "bottom-left"), ("⬇ Dưới phải", "bottom-right"),
                           ("◎ Giữa", "center")]:
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            btn.clicked.connect(lambda _, p=pos: self.canvas.snap_to(p))
            pos_row.addWidget(btn)
        layout.addLayout(pos_row)
        
        # Confirm / Cancel
        btn_row = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("✅ Xác nhận")
        btn_ok.setStyleSheet("font-size: 14px; padding: 8px 24px; background: #28a745; color: white; font-weight: bold; border-radius: 4px;")
        btn_ok.clicked.connect(self._on_confirm)
        btn_row.addWidget(btn_ok)
        btn_cancel = QtWidgets.QPushButton("Hủy")
        btn_cancel.setStyleSheet("font-size: 13px; padding: 8px 16px;")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)
        
        self.setLayout(layout)
        
        # Load video frame + logo after layout is set
        self._video_path = video_path
        self._logo_path = logo_path
        self._size_pct = size_pct
        if logo_path or video_path:
            QtCore.QTimer.singleShot(100, self._delayed_load)
    
    def _delayed_load(self):
        """Load video frame and logo after the canvas has been laid out and has real dimensions"""
        if self._video_path:
            self.canvas.set_video_frame(self._video_path)
        if self._logo_path:
            self.canvas.set_logo(self._logo_path)
            self.canvas.set_size_pct(self._size_pct)
            # Restore saved position or default to top-right
            if self._saved_x_ratio is not None and self._saved_y_ratio is not None:
                dr = self.canvas._draw_rect
                self.canvas.logo_x = int(dr.x() + self._saved_x_ratio * dr.width())
                self.canvas.logo_y = int(dr.y() + self._saved_y_ratio * dr.height())
                self.canvas.update()
            else:
                self.canvas.snap_to("top-right")
    
    def _on_size_changed(self, val):
        self.lbl_size.setText(f"{val}%")
        self.canvas.set_size_pct(val)
    
    def _on_opacity_changed(self, val):
        self.lbl_opacity.setText(f"{val}%")
        self.canvas.opacity = val / 100.0
        self.canvas.update()
    
    def _on_confirm(self):
        rx, ry, rw = self.canvas.get_position_ratios()
        self.result_data = {
            "x_ratio": rx,
            "y_ratio": ry,
            "size_pct": self.sld_size.value(),
            "opacity": self.sld_opacity.value(),
        }
        self.accept()

import json
import srt
from PyQt5.QtMultimedia import QSound, QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QTimer

class VoiceMappingDialog(QtWidgets.QDialog):
    def __init__(self, item, json_path, parent=None):
        super().__init__(parent)
        self.item = item
        self.json_path = json_path
        self.kokoro_voices = ["af_bella", "af_heart", "af_nicole", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
        
        # Detect TTS engine & language from parent settings
        self._tts_engine = "kokoro"  # default
        self._target_lang = getattr(self.item, "target_lang", None)
        try:
            # Walk up to BatchVideoApp to get ocr_settings
            p = parent
            while p is not None:
                if hasattr(p, 'ocr_settings'):
                    self._tts_engine = p.ocr_settings.get("tts_api_url", "kokoro").strip().lower()
                    if not self._target_lang:
                        self._target_lang = p.ocr_settings.get("target_lang", "English")
                    break
                p = p.parent() if hasattr(p, 'parent') and callable(p.parent) else None
        except Exception:
            pass
            
        if not self._target_lang:
            self._target_lang = "English"
        
        # Load MiniMax voices filtered by language
        self.minimax_voices = []
        try:
            from minimax_bridge import get_minimax_voices
            mm_voices = get_minimax_voices()
            # Map target_lang to voices.json key
            lang_map = {"english": "Tiếng Anh", "tiếng việt": "Tiếng Việt", "tiếng anh": "Tiếng Anh"}
            target_key = lang_map.get(self._target_lang.lower(), self._target_lang)
            # If minimax is selected, only show voices for the target language
            if self._tts_engine.startswith("minimax") and target_key in mm_voices:
                for v in mm_voices[target_key]:
                    if isinstance(v, dict) and "id" in v:
                        self.minimax_voices.append(f"mm:{v['id']}")
            else:
                # Show all minimax voices
                for lang, voice_list in mm_voices.items():
                    for v in voice_list:
                        if isinstance(v, dict) and "id" in v:
                            self.minimax_voices.append(f"mm:{v['id']}")
        except Exception:
            pass
        
        # Build voice list based on selected TTS engine
        if self._tts_engine.startswith("minimax"):
            self.all_voices = self.minimax_voices
        elif self._tts_engine.startswith("kokoro"):
            self.all_voices = self.kokoro_voices
        else:
            self.all_voices = self.kokoro_voices + self.minimax_voices
        self.line_combo_boxes = {} # index -> QComboBox
        self.line_emotion_boxes = {} # index -> QComboBox (MiniMax emotion)
        self.subtitles = []
        self.speaker_data = {}
        
        self.cap = cv2.VideoCapture(self.item.video_path)
        if self.cap.isOpened():
            self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self._fps = 30.0
            self._total_frames = 0
            
        # Extract audio for preview (so we can hear video audio)
        import tempfile, subprocess, hashlib
        safe_name = hashlib.md5(self.item.video_path.encode('utf-8')).hexdigest()
        self.temp_audio = os.path.join(tempfile.gettempdir(), f"preview_audio_{safe_name}.wav")
        if not os.path.exists(self.temp_audio):
            ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
            subprocess.run([
                ffmpeg_bin, "-y", "-i", self.item.video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                self.temp_audio
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        self.audio_player = QMediaPlayer(self)
        self.audio_player.setVolume(100)
        if os.path.exists(self.temp_audio):
            self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.temp_audio)))
            
        def on_media_error(error):
            print(f"QMediaPlayer Error: {self.audio_player.errorString()}")
        self.audio_player.error.connect(on_media_error)
        
        self.is_playing = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000.0 / self._fps) if self._fps > 0 else 33)
        self.timer.timeout.connect(self.on_timer_tick)
            
        self.setWindowTitle("Voice Mapping (Video Player)")
        self.resize(1000, 800)
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        
        # --- Top Half: Video Preview ---
        video_container = QtWidgets.QWidget()
        video_layout = QtWidgets.QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_widget = VideoPreview(self.item.video_path, self)
        self.video_widget.setMinimumHeight(250)
        
        # Subtitle overlay label
        self.lbl_subtitle = QtWidgets.QLabel("")
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        self.lbl_subtitle.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; font-size: 18px; font-weight: bold; padding: 5px;")
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        if self._total_frames > 0:
            self.slider.setRange(0, self._total_frames - 1)
        self.slider.valueChanged.connect(self.set_position)
        
        self.lbl_time = QtWidgets.QLabel("00:00")
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.lbl_time)
        
        video_layout.addWidget(self.video_widget)
        video_layout.addWidget(self.lbl_subtitle)
        video_layout.addLayout(controls_layout)
        
        # --- Bottom Half: SRT Table ---
        table_container = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_info = QtWidgets.QLabel(f"File JSON: {os.path.basename(self.json_path)}")
        table_layout.addWidget(lbl_info)
        
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Dòng", "Thời gian", "Nội dung", "SPK_ID", "Giọng (Tùy chỉnh)", "Cảm xúc"])
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 250)
        self.table.setColumnWidth(5, 110)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemClicked.connect(self.on_table_click)
        
        table_layout.addWidget(self.table)
        
        splitter.addWidget(video_container)
        splitter.addWidget(table_container)
        splitter.setSizes([350, 450])
        
        main_layout.addWidget(splitter)
        
        # Save / Cancel
        btn_layout = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("💾 Lưu Ánh Xạ")
        btn_save.clicked.connect(self.save_mapping)
        btn_cancel = QtWidgets.QPushButton("❌ Hủy")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        main_layout.addLayout(btn_layout)
        
    def load_data(self):
        # Load JSON
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                self.speaker_data = {row.get("index", idx+1): row for idx, row in enumerate(json_data)}
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Lỗi", f"Không đọc được file JSON:\n{str(e)}")
            return
            
        # Load SRT
        if self.item.srt_path and os.path.exists(self.item.srt_path):
            srt_path = self.item.srt_path
        else:
            srt_path = self.item.video_path.replace(".mp4", ".srt").replace("_nosub", "_en")
            
        if os.path.exists(srt_path):
            with open(srt_path, 'r', encoding='utf-8') as f:
                self.subtitles = list(srt.parse(f.read()))
        else:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Không tìm thấy file SRT!")
            return
            
        # Populate table
        self.table.setRowCount(len(self.subtitles))
        for i, sub in enumerate(self.subtitles):
            idx = sub.index
            spk_info = self.speaker_data.get(idx, {})
            spk_id = spk_info.get("speaker_id", f"SPK_?")
            
            time_str = f"{sub.start.total_seconds():.1f}s -> {sub.end.total_seconds():.1f}s"
            content = sub.content.replace('\n', ' ')
            
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(idx)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(time_str))
            
            item_content = QtWidgets.QTableWidgetItem(content)
            item_content.setToolTip(content)
            self.table.setItem(i, 2, item_content)
            
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(spk_id))
            
            # Voice Combobox
            widget = QtWidgets.QWidget()
            hlayout = QtWidgets.QHBoxLayout(widget)
            hlayout.setContentsMargins(0, 0, 0, 0)
            
            cmb = QtWidgets.QComboBox()
            cmb.addItems(["(Mặc định)"] + self.all_voices)
            
            # Check existing mapping
            if idx in self.item.line_mapping:
                cb_idx = cmb.findText(self.item.line_mapping[idx])
                if cb_idx >= 0: cmb.setCurrentIndex(cb_idx)
            else:
                if spk_id in self.item.speaker_mapping:
                    cb_idx = cmb.findText(self.item.speaker_mapping[spk_id])
                    if cb_idx >= 0: cmb.setCurrentIndex(cb_idx)
                    
            hlayout.addWidget(cmb)
            
            btn_play = QtWidgets.QPushButton("▶️")
            btn_play.setMaximumWidth(30)
            btn_play.clicked.connect(lambda checked, c=cmb: self.play_preview(c.currentText()))
            hlayout.addWidget(btn_play)
            
            btn_apply_all = QtWidgets.QPushButton("All")
            btn_apply_all.setMaximumWidth(35)
            btn_apply_all.setToolTip(f"Áp dụng giọng này cho tất cả {spk_id}")
            btn_apply_all.clicked.connect(lambda checked, c=cmb, sid=spk_id: self.apply_voice_to_spk(c.currentText(), sid))
            hlayout.addWidget(btn_apply_all)
            
            self.table.setCellWidget(i, 4, widget)
            self.line_combo_boxes[idx] = cmb
            
            # Emotion combobox (column 5)
            cmb_emotion = QtWidgets.QComboBox()
            cmb_emotion.addItems(["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "fluent"])
            # Restore saved emotion if exists
            if hasattr(self.item, 'line_emotion') and idx in self.item.line_emotion:
                em_idx = cmb_emotion.findText(self.item.line_emotion[idx])
                if em_idx >= 0: cmb_emotion.setCurrentIndex(em_idx)
            self.table.setCellWidget(i, 5, cmb_emotion)
            self.line_emotion_boxes[idx] = cmb_emotion
            
        # load first frame
        self.set_position(0)
            
    def apply_voice_to_spk(self, voice_name, spk_id):
        if voice_name == "(Mặc định)": return
        for i, sub in enumerate(self.subtitles):
            idx = sub.index
            spk_info = self.speaker_data.get(idx, {})
            sid = spk_info.get("speaker_id", f"SPK_?")
            if sid == spk_id:
                cmb = self.line_combo_boxes.get(idx)
                if cmb:
                    cb_idx = cmb.findText(voice_name)
                    if cb_idx >= 0: cmb.setCurrentIndex(cb_idx)

    def play_preview(self, voice_name):
        if voice_name == "(Mặc định)": return
        
        if voice_name.startswith("mm:"):
            voice_id = voice_name[3:]
            assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax", "assets")
            os.makedirs(assets_dir, exist_ok=True)
            preview_file = os.path.join(assets_dir, f"preview_{voice_id}.wav")
            
            if not os.path.exists(preview_file):
                lang_text_map = {
                    "Tiếng Việt": "Xin chào, đây là giọng đọc thử nghiệm của tôi.",
                    "Tiếng Anh": "Hello, this is a test voice for you to hear.",
                    "English": "Hello, this is a test voice for you to hear.",
                    "Tiếng Trung": "你好，这是我的测试语音。",
                    "Tiếng Hàn": "안녕하세요, 이것은 제 테스트 음성입니다.",
                    "Tiếng Nhật": "こんにちは、これは私のテスト音声です。",
                    "Tiếng Pháp": "Bonjour, ceci est ma voix de test.",
                    "Tiếng Tây Ban Nha": "Hola, esta es mi voz de prueba.",
                    "Tiếng Tamil": "வணக்கம், இது எனது சோதனை குரல்.",
                    "Tiếng Hindi": "नमस्ते, यह मेरी परीक्षण आवाज़ है।",
                    "Tiếng Ả Rập": "مرحباً، هذا هو صوتي التجريبي.",
                    "Tiếng Indonesia": "Halo, ini adalah suara tes saya.",
                    "Tiếng Ý": "Ciao, questa è la mia voce di prova."
                }
                target_lang = getattr(self, "_target_lang", "Tiếng Việt")
                text = lang_text_map.get(target_lang, "Xin chào, đây là giọng đọc thử nghiệm của tôi.")
                
                # Get the model from parent settings if possible, else default
                mm_model = "speech-2.8-hd"
                try:
                    p = self.parent()
                    while p is not None:
                        if hasattr(p, 'ocr_settings'):
                            mm_model = p.ocr_settings.get("minimax_model", "speech-2.8-hd")
                            break
                        p = p.parent() if hasattr(p, 'parent') and callable(p.parent) else None
                except:
                    pass
                
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    from minimax_bridge import minimax_generate_wav
                    minimax_generate_wav(text, 1.0, preview_file, voice_id=voice_id, emotion="neutral", model=mm_model)
                except Exception as e:
                    print(f"Error generating preview: {e}")
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
                    
            if os.path.exists(preview_file):
                QSound.play(preview_file)
        else:
            preview_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro_previews", f"kokoro-{voice_name}.wav")
            if os.path.exists(preview_file):
                QSound.play(preview_file)
            
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            if hasattr(self, 'audio_player'): self.audio_player.pause()
            self.btn_play.setText("▶ Play")
        else:
            self.is_playing = True
            self.timer.start()
            if hasattr(self, 'audio_player'): self.audio_player.play()
            self.btn_play.setText("⏸ Pause")
            
    def on_timer_tick(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if ret:
            self.video_widget.set_frame(frame)
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            self.slider.blockSignals(True)
            self.slider.setValue(current_frame)
            self.slider.blockSignals(False)
            
            self.update_time_label(current_frame)
        else:
            self.toggle_play()

    def on_table_click(self, item):
        row = item.row()
        if row < len(self.subtitles):
            sub = self.subtitles[row]
            frame_idx = int(sub.start.total_seconds() * self._fps)
            if self._total_frames > 0:
                self.slider.setValue(min(frame_idx, self._total_frames - 1))
            
    def set_position(self, position):
        if not self.cap or not self.cap.isOpened():
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        if ret:
            self.video_widget.set_frame(frame)
            
        self.update_time_label(position)
        
        if hasattr(self, 'audio_player'):
            sec = position / self._fps if self._fps > 0 else 0
            self.audio_player.setPosition(int(sec * 1000))
    def update_time_label(self, current_frame):
        sec = current_frame / self._fps if self._fps > 0 else 0
        minutes = int(sec // 60)
        seconds = int(sec % 60)
        self.lbl_time.setText(f"{minutes:02d}:{seconds:02d}")
        
        current_text = ""
        for sub in self.subtitles:
            if sub.start.total_seconds() <= sec <= sub.end.total_seconds():
                current_text = sub.content.replace('\n', ' ')
                break
        self.lbl_subtitle.setText(current_text)
        
    def save_mapping(self):
        line_map = {}
        speaker_map = {}
        line_emotion = {}
        
        for idx, cmb in self.line_combo_boxes.items():
            voice = cmb.currentText()
            if voice != "(Mặc định)":
                line_map[idx] = voice
                spk_id = self.speaker_data.get(idx, {}).get("speaker_id")
                if spk_id:
                    speaker_map[spk_id] = voice
            # Save emotion per-line
            if idx in self.line_emotion_boxes:
                line_emotion[idx] = self.line_emotion_boxes[idx].currentText()
                    
        self.item.line_mapping = line_map
        self.item.speaker_mapping = speaker_map
        self.item.line_emotion = line_emotion
        self.item.speaker_json_path = self.json_path
        
        if self.cap: self.cap.release()
        self.timer.stop()
        if hasattr(self, 'audio_player'): self.audio_player.stop()
        self.accept()
        
    def reject(self):
        if self.cap: self.cap.release()
        self.timer.stop()
        if hasattr(self, 'audio_player'): self.audio_player.stop()
        super().reject()

class MergeDialog(QtWidgets.QDialog):
    """Dialog for merging short videos into longer compilations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔗 Gộp Video Ngắn")
        self.resize(800, 600)
        self.merge_worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Header
        header = QtWidgets.QLabel("🔗 Gộp Video Ngắn thành Video Dài")
        header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #e67e22;")
        layout.addWidget(header)
        
        desc = QtWidgets.QLabel("Chọn các video đã xử lý xong và ghép lại theo nhóm. Ví dụ: 12 tập ngắn → 3 video dài (mỗi video 4 tập).")
        desc.setStyleSheet("color: #666; font-size: 12px; padding: 0 10px 10px 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Settings row
        settings_row = QtWidgets.QHBoxLayout()
        
        self.btn_add = QtWidgets.QPushButton("➕ Thêm Video")
        self.btn_add.setStyleSheet("font-size: 13px; padding: 8px 16px; background: #28a745; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_add.clicked.connect(self.on_add_videos)
        settings_row.addWidget(self.btn_add)
        
        self.btn_add_folder = QtWidgets.QPushButton("📁 Thêm Folder")
        self.btn_add_folder.setStyleSheet("font-size: 13px; padding: 8px 16px; background: #17a2b8; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_add_folder.clicked.connect(self.on_add_folder)
        settings_row.addWidget(self.btn_add_folder)
        
        settings_row.addWidget(QtWidgets.QLabel("Số tập mỗi nhóm:"))
        self.spin_chunk = QtWidgets.QSpinBox()
        self.spin_chunk.setRange(2, 50)
        self.spin_chunk.setValue(4)
        self.spin_chunk.setStyleSheet("font-size: 13px; padding: 4px;")
        self.spin_chunk.valueChanged.connect(self.update_preview)
        settings_row.addWidget(self.spin_chunk)
        
        settings_row.addStretch()
        
        self.btn_clear = QtWidgets.QPushButton("🗑️ Xóa hết")
        self.btn_clear.setStyleSheet("font-size: 13px; padding: 8px 16px;")
        self.btn_clear.clicked.connect(self.on_clear)
        settings_row.addWidget(self.btn_clear)
        
        layout.addLayout(settings_row)
        
        # Video list
        self.video_list = QtWidgets.QListWidget()
        self.video_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.video_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.video_list.setStyleSheet("font-size: 12px; padding: 4px;")
        self.video_list.setMinimumHeight(200)
        self.video_list.model().rowsMoved.connect(self.update_preview)
        layout.addWidget(self.video_list)
        
        # Remove selected button
        rm_row = QtWidgets.QHBoxLayout()
        self.btn_remove = QtWidgets.QPushButton("❌ Xóa đã chọn")
        self.btn_remove.setStyleSheet("font-size: 11px; padding: 4px 12px;")
        self.btn_remove.clicked.connect(self.on_remove_selected)
        rm_row.addWidget(self.btn_remove)
        
        self.btn_sort = QtWidgets.QPushButton("🔤 Sắp xếp tên")
        self.btn_sort.setStyleSheet("font-size: 11px; padding: 4px 12px;")
        self.btn_sort.clicked.connect(self.on_sort)
        rm_row.addWidget(self.btn_sort)
        
        rm_row.addStretch()
        self.lbl_count = QtWidgets.QLabel("0 video")
        self.lbl_count.setStyleSheet("font-size: 12px; color: #666;")
        rm_row.addWidget(self.lbl_count)
        layout.addLayout(rm_row)
        
        # Preview grouping
        self.preview_label = QtWidgets.QLabel("")
        self.preview_label.setStyleSheet("font-size: 11px; color: #555; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 8px;")
        self.preview_label.setWordWrap(True)
        self.preview_label.setMinimumHeight(60)
        layout.addWidget(self.preview_label)
        
        # ── Logo & Music Settings ──
        extras_group = QtWidgets.QGroupBox("🎬 Logo & Nhạc nền")
        extras_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        extras_layout = QtWidgets.QVBoxLayout()
        
        # Logo checkbox + file picker
        logo_row = QtWidgets.QHBoxLayout()
        self.chk_logo = QtWidgets.QCheckBox("Chèn logo lên video")
        self.chk_logo.setStyleSheet("font-weight: bold;")
        logo_row.addWidget(self.chk_logo)
        logo_row.addStretch()
        extras_layout.addLayout(logo_row)
        
        logo_file_row = QtWidgets.QHBoxLayout()
        logo_file_row.addWidget(QtWidgets.QLabel("   File logo:"))
        self.txt_logo = QtWidgets.QLineEdit()
        self.txt_logo.setPlaceholderText("Chưa chọn file logo...")
        self.txt_logo.setReadOnly(True)
        logo_file_row.addWidget(self.txt_logo, 1)
        btn_logo = QtWidgets.QPushButton("Chọn Logo")
        btn_logo.clicked.connect(self._pick_logo)
        logo_file_row.addWidget(btn_logo)
        extras_layout.addLayout(logo_file_row)
        
        # Position button
        self.btn_logo_pos = QtWidgets.QPushButton("🎯 Chỉnh vị trí Logo (Kéo thả)")
        self.btn_logo_pos.clicked.connect(self._open_logo_position)
        extras_layout.addWidget(self.btn_logo_pos)
        
        # Logo position data (stored from dialog)
        self._logo_pos_data = {"x_ratio": 0.95, "y_ratio": 0.02, "size_pct": 15, "opacity": 80}
        
        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        extras_layout.addWidget(sep)
        
        # ── Intro Music ──
        intro_header = QtWidgets.QHBoxLayout()
        intro_header.addWidget(QtWidgets.QLabel("🎵 Nhạc đầu video (MIX chung audio):"))
        intro_header.addStretch()
        self.lbl_intro_dur = QtWidgets.QLabel("")
        self.lbl_intro_dur.setStyleSheet("color: #888; font-size: 11px;")
        intro_header.addWidget(self.lbl_intro_dur)
        extras_layout.addLayout(intro_header)
        
        intro_row = QtWidgets.QHBoxLayout()
        self.txt_intro = QtWidgets.QLineEdit()
        self.txt_intro.setPlaceholderText("Chưa chọn (để trống = không thêm)")
        self.txt_intro.setReadOnly(True)
        intro_row.addWidget(self.txt_intro, 1)
        btn_intro = QtWidgets.QPushButton("📂")
        btn_intro.setFixedWidth(35)
        btn_intro.setToolTip("Chọn file nhạc")
        btn_intro.clicked.connect(self._pick_intro)
        intro_row.addWidget(btn_intro)
        self.btn_intro_play = QtWidgets.QPushButton("▶")
        self.btn_intro_play.setFixedWidth(35)
        self.btn_intro_play.setToolTip("Nghe thử")
        self.btn_intro_play.clicked.connect(lambda: self._preview_audio(self.txt_intro.text()))
        intro_row.addWidget(self.btn_intro_play)
        btn_intro_clear = QtWidgets.QPushButton("✖")
        btn_intro_clear.setFixedWidth(30)
        btn_intro_clear.clicked.connect(lambda: (self.txt_intro.clear(), self.lbl_intro_dur.clear(), self._update_timeline()))
        intro_row.addWidget(btn_intro_clear)
        extras_layout.addLayout(intro_row)
        
        intro_opts = QtWidgets.QHBoxLayout()
        intro_opts.addWidget(QtWidgets.QLabel("   Sử dụng:"))
        self.spin_intro_sec = QtWidgets.QDoubleSpinBox()
        self.spin_intro_sec.setRange(0, 999)
        self.spin_intro_sec.setValue(0)
        self.spin_intro_sec.setSingleStep(1)
        self.spin_intro_sec.setSuffix(" giây")
        self.spin_intro_sec.setSpecialValueText("Toàn bộ")
        self.spin_intro_sec.setToolTip("0 = dùng toàn bộ file nhạc")
        self.spin_intro_sec.valueChanged.connect(self._update_timeline)
        intro_opts.addWidget(self.spin_intro_sec)
        intro_opts.addWidget(QtWidgets.QLabel("  Fade in:"))
        self.spin_intro_fade = QtWidgets.QDoubleSpinBox()
        self.spin_intro_fade.setRange(0, 30)
        self.spin_intro_fade.setValue(2.0)
        self.spin_intro_fade.setSingleStep(0.5)
        self.spin_intro_fade.setSuffix("s")
        intro_opts.addWidget(self.spin_intro_fade)
        intro_opts.addWidget(QtWidgets.QLabel("  Fade out:"))
        self.spin_intro_fadeout = QtWidgets.QDoubleSpinBox()
        self.spin_intro_fadeout.setRange(0, 30)
        self.spin_intro_fadeout.setValue(2.0)
        self.spin_intro_fadeout.setSingleStep(0.5)
        self.spin_intro_fadeout.setSuffix("s")
        intro_opts.addWidget(self.spin_intro_fadeout)
        intro_opts.addWidget(QtWidgets.QLabel("  Vol:"))
        self.spin_intro_vol = QtWidgets.QDoubleSpinBox()
        self.spin_intro_vol.setRange(0.05, 2.0)
        self.spin_intro_vol.setValue(0.5)
        self.spin_intro_vol.setSingleStep(0.05)
        intro_opts.addWidget(self.spin_intro_vol)
        intro_opts.addStretch()
        extras_layout.addLayout(intro_opts)
        
        # Separator 2
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setStyleSheet("color: #ddd;")
        extras_layout.addWidget(sep2)
        
        # ── Outro Music ──
        outro_header = QtWidgets.QHBoxLayout()
        outro_header.addWidget(QtWidgets.QLabel("🎵 Nhạc cuối video (MIX chung audio):"))
        outro_header.addStretch()
        self.lbl_outro_dur = QtWidgets.QLabel("")
        self.lbl_outro_dur.setStyleSheet("color: #888; font-size: 11px;")
        outro_header.addWidget(self.lbl_outro_dur)
        extras_layout.addLayout(outro_header)
        
        outro_row = QtWidgets.QHBoxLayout()
        self.txt_outro = QtWidgets.QLineEdit()
        self.txt_outro.setPlaceholderText("Chưa chọn (để trống = không thêm)")
        self.txt_outro.setReadOnly(True)
        outro_row.addWidget(self.txt_outro, 1)
        btn_outro = QtWidgets.QPushButton("📂")
        btn_outro.setFixedWidth(35)
        btn_outro.setToolTip("Chọn file nhạc")
        btn_outro.clicked.connect(self._pick_outro)
        outro_row.addWidget(btn_outro)
        self.btn_outro_play = QtWidgets.QPushButton("▶")
        self.btn_outro_play.setFixedWidth(35)
        self.btn_outro_play.setToolTip("Nghe thử")
        self.btn_outro_play.clicked.connect(lambda: self._preview_audio(self.txt_outro.text()))
        outro_row.addWidget(self.btn_outro_play)
        btn_outro_clear = QtWidgets.QPushButton("✖")
        btn_outro_clear.setFixedWidth(30)
        btn_outro_clear.clicked.connect(lambda: (self.txt_outro.clear(), self.lbl_outro_dur.clear(), self._update_timeline()))
        outro_row.addWidget(btn_outro_clear)
        extras_layout.addLayout(outro_row)
        
        outro_opts = QtWidgets.QHBoxLayout()
        outro_opts.addWidget(QtWidgets.QLabel("   Sử dụng:"))
        self.spin_outro_sec = QtWidgets.QDoubleSpinBox()
        self.spin_outro_sec.setRange(0, 999)
        self.spin_outro_sec.setValue(0)
        self.spin_outro_sec.setSingleStep(1)
        self.spin_outro_sec.setSuffix(" giây")
        self.spin_outro_sec.setSpecialValueText("Toàn bộ")
        self.spin_outro_sec.setToolTip("0 = dùng toàn bộ file nhạc")
        self.spin_outro_sec.valueChanged.connect(self._update_timeline)
        outro_opts.addWidget(self.spin_outro_sec)
        outro_opts.addWidget(QtWidgets.QLabel("  Fade in:"))
        self.spin_outro_fadein = QtWidgets.QDoubleSpinBox()
        self.spin_outro_fadein.setRange(0, 30)
        self.spin_outro_fadein.setValue(2.0)
        self.spin_outro_fadein.setSingleStep(0.5)
        self.spin_outro_fadein.setSuffix("s")
        outro_opts.addWidget(self.spin_outro_fadein)
        outro_opts.addWidget(QtWidgets.QLabel("  Fade out:"))
        self.spin_outro_fadeout = QtWidgets.QDoubleSpinBox()
        self.spin_outro_fadeout.setRange(0, 30)
        self.spin_outro_fadeout.setValue(2.0)
        self.spin_outro_fadeout.setSingleStep(0.5)
        self.spin_outro_fadeout.setSuffix("s")
        outro_opts.addWidget(self.spin_outro_fadeout)
        outro_opts.addWidget(QtWidgets.QLabel("  Vol:"))
        self.spin_outro_vol = QtWidgets.QDoubleSpinBox()
        self.spin_outro_vol.setRange(0.05, 2.0)
        self.spin_outro_vol.setValue(0.5)
        self.spin_outro_vol.setSingleStep(0.05)
        outro_opts.addWidget(self.spin_outro_vol)
        outro_opts.addStretch()
        extras_layout.addLayout(outro_opts)
        
        # ── Visual Timeline ──
        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setStyleSheet("color: #ddd;")
        extras_layout.addWidget(sep3)
        
        self.lbl_timeline = QtWidgets.QLabel("")
        self.lbl_timeline.setStyleSheet(
            "font-size: 11px; padding: 6px; background: #1a1a2e; color: #eee; "
            "border-radius: 4px; font-family: monospace;"
        )
        self.lbl_timeline.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_timeline.setMinimumHeight(30)
        extras_layout.addWidget(self.lbl_timeline)
        self._update_timeline()
        
        # Audio preview player
        self._preview_process = None
        
        extras_group.setLayout(extras_layout)
        layout.addWidget(extras_group)
        
        # Progress
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)
        
        # Action buttons
        btn_row = QtWidgets.QHBoxLayout()
        
        self.btn_merge = QtWidgets.QPushButton("▶️ Bắt đầu Gộp")
        self.btn_merge.setStyleSheet("font-size: 15px; padding: 12px 30px; background: #e67e22; color: white; font-weight: bold; border-radius: 6px;")
        self.btn_merge.clicked.connect(self.on_merge)
        self.btn_merge.setEnabled(False)
        btn_row.addWidget(self.btn_merge)
        
        btn_close = QtWidgets.QPushButton("✖ Đóng")
        btn_close.setStyleSheet("font-size: 13px; padding: 10px 20px;")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)
        
        layout.addLayout(btn_row)
        self.setLayout(layout)
    
    def on_add_videos(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Chọn Video", "", "Videos (*.mp4 *.mkv *.avi *.mov)"
        )
        for f in files:
            self.video_list.addItem(f)
        self.update_preview()
    
    def on_add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Chọn Folder chứa Video")
        if folder:
            import glob
            videos = sorted(glob.glob(os.path.join(folder, "*.mp4")))
            videos += sorted(glob.glob(os.path.join(folder, "*.mkv")))
            for v in videos:
                self.video_list.addItem(v)
            self.update_preview()
    
    def on_clear(self):
        self.video_list.clear()
        self.update_preview()
    
    def on_remove_selected(self):
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
        self.update_preview()
    
    def on_sort(self):
        items = []
        for i in range(self.video_list.count()):
            items.append(self.video_list.item(i).text())
        items.sort()
        self.video_list.clear()
        for item in items:
            self.video_list.addItem(item)
        self.update_preview()
    
    def _pick_logo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn Logo", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.txt_logo.setText(path)
    
    def _pick_intro(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn nhạc đầu video", "", "Audio (*.mp3 *.wav *.m4a *.aac *.ogg *.flac)"
        )
        if path:
            self.txt_intro.setText(path)
            dur = self._get_audio_duration(path)
            if dur > 0:
                mins, secs = divmod(dur, 60)
                self.lbl_intro_dur.setText(f"⏱ {int(mins)}:{secs:05.2f}")
            self._update_timeline()
    
    def _pick_outro(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn nhạc cuối video", "", "Audio (*.mp3 *.wav *.m4a *.aac *.ogg *.flac)"
        )
        if path:
            self.txt_outro.setText(path)
            dur = self._get_audio_duration(path)
            if dur > 0:
                mins, secs = divmod(dur, 60)
                self.lbl_outro_dur.setText(f"⏱ {int(mins)}:{secs:05.2f}")
            self._update_timeline()
    
    def _get_audio_duration(self, path):
        """Get audio duration in seconds using ffprobe"""
        try:
            ffprobe = "/usr/bin/ffprobe" if os.path.exists("/usr/bin/ffprobe") else "ffprobe"
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return float(result.stdout.strip())
        except:
            return 0
    
    def _preview_audio(self, path):
        """Play/stop audio preview using ffplay"""
        if self._preview_process and self._preview_process.poll() is None:
            self._preview_process.kill()
            self._preview_process = None
            self.btn_intro_play.setText("▶")
            self.btn_outro_play.setText("▶")
            return
        
        if not path or not os.path.exists(path):
            return
        
        try:
            ffplay = "/usr/bin/ffplay" if os.path.exists("/usr/bin/ffplay") else "ffplay"
            self._preview_process = subprocess.Popen(
                [ffplay, "-nodisp", "-autoexit", "-t", "10", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Update button to show stop state
            if path == self.txt_intro.text():
                self.btn_intro_play.setText("⏹")
            elif path == self.txt_outro.text():
                self.btn_outro_play.setText("⏹")
        except Exception as e:
            print(f"⚠️ Preview failed: {e}")
    
    def _update_timeline(self, _=None):
        """Update the visual timeline bar"""
        has_intro = bool(self.txt_intro.text().strip())
        has_outro = bool(self.txt_outro.text().strip())
        
        intro_sec = self.spin_intro_sec.value()
        outro_sec = self.spin_outro_sec.value()
        
        intro_txt = ""
        outro_txt = ""
        
        if has_intro:
            if intro_sec > 0:
                intro_txt = f"🎵 Intro ({intro_sec:.0f}s)"
            else:
                intro_txt = "🎵 Intro (full)"
        
        if has_outro:
            if outro_sec > 0:
                outro_txt = f"🎵 Outro ({outro_sec:.0f}s)"
            else:
                outro_txt = "🎵 Outro (full)"
        
        if has_intro and has_outro:
            timeline = f"  {intro_txt}  ◀━━━━━━  🎬 VIDEO  ━━━━━━▶  {outro_txt}  "
        elif has_intro:
            timeline = f"  {intro_txt}  ◀━━━━━━━━━━  🎬 VIDEO  ━━━━━━━━━━▶  "
        elif has_outro:
            timeline = f"  ◀━━━━━━━━━━  🎬 VIDEO  ━━━━━━━━━━▶  {outro_txt}  "
        else:
            timeline = "  ◀━━━━━━━━━━━━━━  🎬 VIDEO (không nhạc)  ━━━━━━━━━━━━━━▶  "
        
        self.lbl_timeline.setText(timeline)
    
    def _open_logo_position(self):
        """Open visual logo positioning dialog"""
        logo_path = self.txt_logo.text().strip()
        if not logo_path or not os.path.exists(logo_path):
            QtWidgets.QMessageBox.warning(self, "⚠️", "Vui lòng chọn file logo trước!")
            return
        
        # Get first video as preview background
        video_path = ""
        if self.video_list.count() > 0:
            video_path = self.video_list.item(0).text()
        
        dlg = LogoPositionDialog(
            logo_path=logo_path,
            video_path=video_path,
            size_pct=self._logo_pos_data.get("size_pct", 15),
            opacity=self._logo_pos_data.get("opacity", 80),
            x_ratio=self._logo_pos_data.get("x_ratio"),
            y_ratio=self._logo_pos_data.get("y_ratio"),
            parent=self
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.result_data:
            self._logo_pos_data = dlg.result_data
            print(f"✅ Logo position set: {self._logo_pos_data}")
    
    def update_preview(self):
        count = self.video_list.count()
        chunk_size = self.spin_chunk.value()
        self.lbl_count.setText(f"{count} video")
        self.btn_merge.setEnabled(count >= 2)
        
        if count == 0:
            self.preview_label.setText("Chưa có video nào. Bấm '+ Thêm Video' để bắt đầu.")
            return
        
        # Build preview text
        all_files = [self.video_list.item(i).text() for i in range(count)]
        chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
        
        lines = [f"📊 Sẽ tạo {len(chunks)} video gộp từ {count} video:"]
        for idx, chunk in enumerate(chunks):
            names = [os.path.basename(f) for f in chunk]
            lines.append(f"  🎥 Phần {idx+1}: {' + '.join(names)}")
        
        self.preview_label.setText("\n".join(lines))
    
    def _get_video_duration(self, video_path, ffmpeg_bin="ffmpeg"):
        """Get video duration in seconds using ffprobe"""
        try:
            ffprobe = ffmpeg_bin.replace("ffmpeg", "ffprobe")
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return float(result.stdout.strip())
        except:
            return 0
    
    def on_merge(self):
        count = self.video_list.count()
        if count < 2:
            return
        
        chunk_size = self.spin_chunk.value()
        all_files = [self.video_list.item(i).text() for i in range(count)]
        chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
        
        # Get extras settings
        logo_path = self.txt_logo.text().strip()
        intro_path = self.txt_intro.text().strip()
        outro_path = self.txt_outro.text().strip()
        has_logo = self.chk_logo.isChecked() and bool(logo_path) and os.path.exists(logo_path)
        has_intro = bool(intro_path) and os.path.exists(intro_path)
        has_outro = bool(outro_path) and os.path.exists(outro_path)
        has_extras = has_logo or has_intro or has_outro
        
        logo_pos_data = self._logo_pos_data
        
        # Per-track music settings
        intro_settings = {
            "use_sec": self.spin_intro_sec.value(),
            "fade_in": self.spin_intro_fade.value(),
            "fade_out": self.spin_intro_fadeout.value(),
            "vol": self.spin_intro_vol.value(),
        }
        outro_settings = {
            "use_sec": self.spin_outro_sec.value(),
            "fade_in": self.spin_outro_fadein.value(),
            "fade_out": self.spin_outro_fadeout.value(),
            "vol": self.spin_outro_vol.value(),
        }
        
        self.btn_merge.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
        
        merged_files = []
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            self.status_label.setText(f"🔄 Đang ghép phần {idx+1}/{total_chunks}...")
            QtWidgets.QApplication.processEvents()
            
            if len(chunk) <= 1:
                self.progress.setValue(int((idx + 1) / total_chunks * 100))
                continue
            
            # Determine output name
            first_base = os.path.splitext(chunk[0])[0]
            if first_base.endswith("_final"):
                first_base = first_base[:-6]
            last_base = os.path.splitext(os.path.basename(chunk[-1]))[0]
            if last_base.endswith("_final"):
                last_base = last_base[:-6]
            
            out_dir = os.path.dirname(chunk[0])
            merged_out = os.path.join(out_dir, f"{os.path.basename(first_base)}_to_{last_base}_merged.mp4")
            
            # ── Step 1: Concat videos ──
            if has_extras:
                concat_temp = merged_out + ".temp_concat.mp4"
            else:
                concat_temp = merged_out
            
            concat_txt = merged_out + ".concat.txt"
            with open(concat_txt, "w", encoding="utf-8") as f:
                for vid in chunk:
                    escaped = vid.replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            
            cmd_concat = [
                ffmpeg_bin, "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_txt,
                "-c", "copy",
                concat_temp
            ]
            
            print(f"🔄 Concat: {' + '.join([os.path.basename(v) for v in chunk])}")
            result = subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            
            if os.path.exists(concat_txt):
                os.remove(concat_txt)
            
            if result.returncode != 0 or not os.path.exists(concat_temp):
                print(f"❌ Concat failed: {result.stderr[-200:] if result.stderr else 'unknown'}")
                self.progress.setValue(int((idx + 1) / total_chunks * 100))
                continue
            
            # ── Step 2: Apply logo + intro/outro music if any ──
            if has_extras:
                self.status_label.setText(f"🎬 Đang chèn logo/nhạc phần {idx+1}/{total_chunks}...")
                QtWidgets.QApplication.processEvents()
                
                video_dur = self._get_video_duration(concat_temp, ffmpeg_bin)
                
                # Build complex FFmpeg filter
                inputs = ["-i", concat_temp]
                filter_parts = []
                audio_mix_parts = []
                input_idx = 1  # 0 = concat video
                
                # Logo overlay
                if has_logo:
                    inputs.extend(["-i", logo_path])
                    x_ratio = logo_pos_data.get("x_ratio", 0.95)
                    y_ratio = logo_pos_data.get("y_ratio", 0.02)
                    size_pct = logo_pos_data.get("size_pct", 15)
                    opacity_val = logo_pos_data.get("opacity", 80) / 100.0
                    
                    # Scale logo to size_pct% of video width, then position by ratio
                    logo_scale = f"[{input_idx}]scale=iw*{size_pct}/100:-1"
                    if opacity_val < 1.0:
                        logo_scale += f",format=rgba,colorchannelmixer=aa={opacity_val}"
                    logo_scale += "[logo]"
                    filter_parts.append(logo_scale)
                    
                    # Position: x_ratio * main_W, y_ratio * main_H
                    overlay_x = f"W*{x_ratio:.4f}"
                    overlay_y = f"H*{y_ratio:.4f}"
                    filter_parts.append(f"[0:v][logo]overlay={overlay_x}:{overlay_y}[vout]")
                    input_idx += 1
                    video_label = "[vout]"
                else:
                    video_label = "[0:v]"
                
                # Audio: start with original audio
                audio_label = "[0:a]"
                
                # Intro music
                if has_intro:
                    inputs.extend(["-i", intro_path])
                    i_vol = intro_settings["vol"]
                    i_fi = intro_settings["fade_in"]
                    i_fo = intro_settings["fade_out"]
                    i_sec = intro_settings["use_sec"]
                    
                    intro_filter = f"[{input_idx}]"
                    # Trim if use_sec > 0
                    if i_sec > 0:
                        intro_filter += f"atrim=0:{i_sec},asetpts=PTS-STARTPTS,"
                        fade_out_start = max(0, i_sec - i_fo)
                    else:
                        # Get full duration for fade out timing
                        intro_full_dur = self._get_audio_duration(intro_path)
                        fade_out_start = max(0, intro_full_dur - i_fo)
                    
                    intro_filter += f"volume={i_vol}"
                    if i_fi > 0:
                        intro_filter += f",afade=t=in:st=0:d={i_fi}"
                    if i_fo > 0:
                        intro_filter += f",afade=t=out:st={fade_out_start}:d={i_fo}"
                    intro_filter += "[intro_aud]"
                    filter_parts.append(intro_filter)
                    audio_mix_parts.append("[intro_aud]")
                    input_idx += 1
                
                # Outro music
                if has_outro:
                    inputs.extend(["-i", outro_path])
                    o_vol = outro_settings["vol"]
                    o_fi = outro_settings["fade_in"]
                    o_fo = outro_settings["fade_out"]
                    o_sec = outro_settings["use_sec"]
                    
                    outro_dur = self._get_audio_duration(outro_path)
                    use_dur = o_sec if o_sec > 0 else outro_dur
                    
                    outro_filter = f"[{input_idx}]"
                    # Trim if use_sec > 0
                    if o_sec > 0:
                        outro_filter += f"atrim=0:{o_sec},asetpts=PTS-STARTPTS,"
                    
                    outro_filter += f"volume={o_vol}"
                    if o_fi > 0:
                        outro_filter += f",afade=t=in:st=0:d={o_fi}"
                    if o_fo > 0:
                        fade_out_start = max(0, use_dur - o_fo)
                        outro_filter += f",afade=t=out:st={fade_out_start}:d={o_fo}"
                    outro_filter += "[outro_raw]"
                    filter_parts.append(outro_filter)
                    
                    # Delay outro to play at end of video
                    if video_dur > 0:
                        outro_start = max(0, video_dur - use_dur)
                        filter_parts.append(f"[outro_raw]adelay={int(outro_start * 1000)}|{int(outro_start * 1000)}[outro_aud]")
                    else:
                        filter_parts.append("[outro_raw]acopy[outro_aud]")
                    
                    audio_mix_parts.append("[outro_aud]")
                    input_idx += 1
                
                # Mix audio
                if audio_mix_parts:
                    mix_inputs = audio_label + "".join(audio_mix_parts)
                    n_inputs = 1 + len(audio_mix_parts)
                    filter_parts.append(f"{mix_inputs}amix=inputs={n_inputs}:duration=first:dropout_transition=2[aout]")
                    audio_label = "[aout]"
                
                # Build full filter
                filter_complex = ";".join(filter_parts)
                
                cmd_extras = [ffmpeg_bin, "-y"] + inputs
                if filter_complex:
                    cmd_extras.extend(["-filter_complex", filter_complex])
                cmd_extras.extend([
                    "-map", video_label.strip("[]") if video_label == "[0:v]" else video_label,
                    "-map", audio_label,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    merged_out
                ])
                
                # Fix map labels for non-filtered case
                cmd_extras_final = []
                for arg in cmd_extras:
                    if arg == "[0:v]":
                        arg = "0:v"
                    cmd_extras_final.append(arg)
                
                print(f"🎬 Applying extras: logo={has_logo}, intro={has_intro}, outro={has_outro}")
                result2 = subprocess.run(cmd_extras_final, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                
                # Clean temp
                if os.path.exists(concat_temp):
                    os.remove(concat_temp)
                
                if result2.returncode != 0:
                    print(f"❌ Extras failed: {result2.stderr[-300:] if result2.stderr else 'unknown'}")
                    self.progress.setValue(int((idx + 1) / total_chunks * 100))
                    continue
            
            if os.path.exists(merged_out):
                merged_files.append(merged_out)
                print(f"✅ Merged: {os.path.basename(merged_out)}")
            
            self.progress.setValue(int((idx + 1) / total_chunks * 100))
            QtWidgets.QApplication.processEvents()
        
        self.btn_merge.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.progress.setValue(100)
        
        if merged_files:
            self.status_label.setText(f"✅ Hoàn thành! Tạo {len(merged_files)} video gộp.")
            msg = "\n".join([f"• {os.path.basename(f)}" for f in merged_files])
            QtWidgets.QMessageBox.information(
                self, "✅ Gộp xong!",
                f"Đã tạo {len(merged_files)} video gộp:\n\n{msg}"
            )
        else:
            self.status_label.setText("❌ Không có video nào được ghép.")


# ============= MAIN WINDOW =============

class BatchWindow(QtWidgets.QWidget):
    """Main batch processing window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tool6 Batch - Video OCR Queue")
        self.resize(1400, 900)
        
        self.queue = []
        self.video_cards = []
        self.is_portrait_mode = True
        self.process_mode = ProcessMode.SRT_REMOVESUB
        
        # OCR Settings (defaults)
        self.ocr_settings = {
            "vse_mode": "accurate",
            "tts_api_url": "http://192.168.1.75:8002/api2/convert-tts",
            "kokoro_voice": "af_bella",
            "tts_speed": 1.0,
            "orig_video_speed": 1.0,
            "uvr_vocal_remove": False,
            "uvr_strength": 100,
            "vol_bgm": 2.0,
            "sync_mode": "stretch_video",
            "vol_tts": 2.0,
            "target_lang": "Tiếng Việt",
            "ocr_prompt": "Bạn là một chuyên gia dịch thuật hãy dịch cho tôi đoạn srt sau qua tiếng việt : ",
            "removesub_method": "ai", # 'ai' (LAMA - Best Quality) or 'delogo' (Fast Clone Stamp)
        }
        
        self.current_index = 0
        
        # === Auto-clean temp WAV sync files on startup ===
        self._cleanup_temp_files()
        self.ocr_worker = None
        self.inpaint_worker = None
        
        # ROI clipboard for copy/paste
        self.clipboard_roi = None
        self.ocr_workers = {} # card -> worker
        self.inpaint_workers = {} # card -> worker
        
        self.downloader_window = None  # Reference to downloader window
        
        self.downloader_window = None  # Reference to downloader window
        
        self.init_ui()
        self.load_settings()

    def closeEvent(self, event):
        """Save settings on exit"""
        self.save_settings()
        super().closeEvent(event)
    
    def _cleanup_temp_files(self):
        """Clean up temp WAV sync files and VSR temp mp4 files to prevent disk bloat"""
        import shutil
        import tempfile
        import glob
        
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".temp_wav_sync")
        if os.path.isdir(temp_dir):
            try:
                size_bytes = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, dn, filenames in os.walk(temp_dir)
                    for f in filenames
                )
                size_gb = size_bytes / (1024**3)
                if size_gb > 0.01:
                    print(f"🧹 Cleaning temp files: {temp_dir} ({size_gb:.1f} GB)")
                shutil.rmtree(temp_dir, ignore_errors=True)
                if size_gb > 0.01:
                    print(f"✅ Freed {size_gb:.1f} GB disk space")
            except Exception as e:
                print(f"⚠️ Could not clean temp files: {e}")
                
        # Clean stray .mp4 temp files in /tmp created by VSR
        tmp_pattern = os.path.join(tempfile.gettempdir(), "tmp*.mp4")
        for f in glob.glob(tmp_pattern):
            try:
                os.remove(f)
            except Exception as e:
                pass
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        top_layout = QtWidgets.QHBoxLayout()
        
        self.btn_add = QtWidgets.QPushButton("➕ Add Videos")
        self.btn_add.clicked.connect(self.on_add_videos)
        self.btn_add.setStyleSheet("font-size: 14px; padding: 10px; background: #28a745; color: white; font-weight: bold;")
        top_layout.addWidget(self.btn_add)
        
        self.btn_settings = QtWidgets.QPushButton("⚙️ Settings")
        self.btn_settings.clicked.connect(self.on_settings)
        self.btn_settings.setStyleSheet("font-size: 14px; padding: 10px;")
        top_layout.addWidget(self.btn_settings)
        
        self.btn_start = QtWidgets.QPushButton("▶️ Start Queue")
        self.btn_start.clicked.connect(self.on_start_queue)
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("font-size: 14px; padding: 10px; background: #007bff; color: white; font-weight: bold;")
        top_layout.addWidget(self.btn_start)
        
        self.btn_layout = QtWidgets.QPushButton("🔄 Landscape Mode")
        self.btn_layout.setCheckable(True)
        self.btn_layout.setChecked(True)
        self.btn_layout.clicked.connect(self.on_toggle_layout)
        self.btn_layout.setStyleSheet("font-size: 14px; padding: 10px;")
        top_layout.addWidget(self.btn_layout)
        
        self.btn_clear = QtWidgets.QPushButton("🗑️ Clear All")
        self.btn_clear.clicked.connect(self.on_clear_all)
        self.btn_clear.setStyleSheet("font-size: 14px; padding: 10px;")
        top_layout.addWidget(self.btn_clear)
        
        
        top_layout.addWidget(QtWidgets.QLabel("|"))  # Separator
        
        self.btn_save_queue = QtWidgets.QPushButton("💾 Save Queue")
        self.btn_save_queue.clicked.connect(self.on_save_queue)
        self.btn_save_queue.setStyleSheet("font-size: 14px; padding: 10px;")
        top_layout.addWidget(self.btn_save_queue)
        
        self.btn_load_queue = QtWidgets.QPushButton("📂 Load Queue")
        self.btn_load_queue.clicked.connect(self.on_load_queue)
        self.btn_load_queue.setStyleSheet("font-size: 14px; padding: 10px;")
        top_layout.addWidget(self.btn_load_queue)
        
        top_layout.addWidget(QtWidgets.QLabel("|"))  # Separator
        
        self.btn_downloader = QtWidgets.QPushButton("⬇️ Downloader")
        self.btn_downloader.clicked.connect(self.on_open_downloader)
        self.btn_downloader.setStyleSheet("font-size: 14px; padding: 10px; background: #6f42c1; color: white; font-weight: bold;")
        top_layout.addWidget(self.btn_downloader)
        
        self.btn_sub_settings = QtWidgets.QPushButton("⚙️ Subtitle Settings")
        self.btn_sub_settings.clicked.connect(self.open_subtitle_settings)
        self.btn_sub_settings.setStyleSheet("font-size: 14px; padding: 10px; background: #17a2b8; color: white; font-weight: bold;")
        top_layout.addWidget(self.btn_sub_settings)
        
        self.btn_merge_dialog = QtWidgets.QPushButton("🔗 Gộp Video")
        self.btn_merge_dialog.clicked.connect(self.on_open_merge_dialog)
        self.btn_merge_dialog.setStyleSheet("font-size: 14px; padding: 10px; background: #e67e22; color: white; font-weight: bold;")
        top_layout.addWidget(self.btn_merge_dialog)

        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Merge Settings (only for TTS_MERGE / BURN_ONLY)
        self.merge_widget = QtWidgets.QWidget()
        merge_layout = QtWidgets.QHBoxLayout(self.merge_widget)
        merge_layout.setContentsMargins(0, 0, 0, 0)
        self.chkMergeEpisodes = QtWidgets.QCheckBox("🔄 Ghép Video sau khi xử lý (Dành cho Batch)")
        self.chkMergeEpisodes.setChecked(bool(self.ocr_settings.get("merge_episodes", False)))
        self.chkMergeEpisodes.stateChanged.connect(lambda: self.ocr_settings.update({"merge_episodes": self.chkMergeEpisodes.isChecked()}))
        merge_layout.addWidget(self.chkMergeEpisodes)
        
        merge_layout.addWidget(QtWidgets.QLabel("Số tập mỗi nhóm:"))
        self.spinMergeCount = QtWidgets.QSpinBox()
        self.spinMergeCount.setRange(2, 50)
        self.spinMergeCount.setValue(int(self.ocr_settings.get("merge_count", 4)))
        self.spinMergeCount.valueChanged.connect(lambda v: self.ocr_settings.update({"merge_count": v}))
        merge_layout.addWidget(self.spinMergeCount)
        merge_layout.addStretch()
        layout.addWidget(self.merge_widget)
        self._update_merge_visibility()
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(10)
        self.grid_widget.setLayout(self.grid_layout)
        
        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll)
        
        progress_layout = QtWidgets.QHBoxLayout()
        progress_layout.addWidget(QtWidgets.QLabel("Overall Progress:"))
        
        self.overall_progress = QtWidgets.QProgressBar()
        progress_layout.addWidget(self.overall_progress)
        
        self.status_label = QtWidgets.QLabel("Ready - Mode: SRT Only")
        progress_layout.addWidget(self.status_label)
        
        layout.addLayout(progress_layout)
        self.setLayout(layout)
    
    def on_settings(self):
        dialog = SettingsDialog(settings=self.ocr_settings, current_mode=self.process_mode, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_mode = dialog.get_mode()
            print(f"\n🔧 [SETTINGS] Mode selected: {new_mode} (was: {self.process_mode})")
            self.process_mode = new_mode
            self.ocr_settings.update(dialog.get_settings())
            
            # If switched to TTS_MERGE or BURN_ONLY, try to auto-detect SRTs for items without SRT
            if new_mode in (ProcessMode.TTS_MERGE, ProcessMode.TTS_MERGE_MULTI, ProcessMode.BURN_ONLY):
                for i, item in enumerate(self.queue):
                    # Only auto-detect if srt_path is NOT already set (preserve Multi-Lang assignments)
                    if not item.srt_path:
                        new_srt = item.auto_detect_srt()
                        if new_srt:
                            item.srt_path = new_srt
                    if i < len(self.video_cards):
                        self.video_cards[i].update_srt_display()
            
            self.update_mode_display()
            self.update_buttons()
            self._update_merge_visibility()
            
            # Toggle logo panel on all cards
            self._toggle_logo_panels()
    
    def update_mode_display(self):
        """Update status label to show current mode"""
        if self.process_mode == ProcessMode.SRT_REMOVESUB:
            mode_text = "🧼 SRT + RemoveSub"
        elif self.process_mode == ProcessMode.REMOVE_LOGO:
            mode_text = "🚫 Remove Logo"
        elif self.process_mode == ProcessMode.SRT_ONLY:
            mode_text = "📝 SRT Only"
        elif self.process_mode == ProcessMode.BURN_ONLY:
            mode_text = "🔥 Burn Subtitles Only"
        elif self.process_mode == ProcessMode.STT_EXTRACT:
            mode_text = "🎤 Speech-to-Text"
        elif self.process_mode == ProcessMode.TTS_MERGE_MULTI:
            mode_text = "🌍 Multi-Lang TTS + Merge"
        else:
            mode_text = "🎬 TTS + Merge"
        self.status_label.setText(f"Mode: {mode_text}")

    def _update_merge_visibility(self):
        """Only show merge option for TTS_MERGE and BURN_ONLY modes"""
        show = self.process_mode in (ProcessMode.TTS_MERGE, ProcessMode.TTS_MERGE_MULTI, ProcessMode.BURN_ONLY)
        self.merge_widget.setVisible(show)

    def _toggle_logo_panels(self):
        """Show/hide logo time range panels on all cards based on mode"""
        show = (self.process_mode == ProcessMode.REMOVE_LOGO)
        for card in self.video_cards:
            if show:
                card.show_logo_panel()
            else:
                card.hide_logo_panel()
    
    def on_toggle_layout(self, checked):
        self.is_portrait_mode = checked
        if checked:
            self.btn_layout.setText("🔄 Landscape Mode")
        else:
            self.btn_layout.setText("🔄 Portrait Mode")
        self.rebuild_all_cards()
    
    def on_add_videos(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Videos", "", "Videos (*.mp4 *.mkv *.avi *.mov)"
        )
        
        if not files:
            return
        
        # Collect all items first
        items_to_add = []
        for file_path in files:
            print(f"\n🔍 [ADD] process_mode = {self.process_mode}")
            if self.process_mode == ProcessMode.TTS_MERGE_MULTI:
                dummy_item = VideoQueueItem(file_path)
                all_srts = dummy_item.auto_detect_all_srts()
                print(f"🔍 [MULTI] Found {len(all_srts)} SRTs: {[s.get('lang') for s in all_srts]}")
                if not all_srts:
                    all_srts = [{"srt_path": None, "lang": None}]
            else:
                all_srts = [{"srt_path": None, "lang": None}]
                
            for srt_info in all_srts:
                item = VideoQueueItem(file_path)
                
                if srt_info.get("srt_path"):
                    item.srt_path = srt_info["srt_path"]
                    print(f"✅ Auto-detected SRT: {os.path.basename(item.srt_path)}")
                else:
                    if self.process_mode in (ProcessMode.TTS_MERGE, ProcessMode.TTS_MERGE_MULTI, ProcessMode.BURN_ONLY):
                        fallback_srt = item.auto_detect_srt()
                        if fallback_srt:
                            print(f"✅ Auto-detected SRT: {os.path.basename(fallback_srt)}")
                        else:
                            print(f"⚠️ No SRT found for {os.path.basename(file_path)}")
                
                if srt_info.get("lang"):
                    item.target_lang = srt_info["lang"]
                
                if self.process_mode in (ProcessMode.TTS_MERGE, ProcessMode.TTS_MERGE_MULTI, ProcessMode.BURN_ONLY):
                    roi = item.auto_load_roi_from_segments()
                    if roi:
                        print(f"✅ Auto-loaded ROI from segments: {roi}")
                
                items_to_add.append(item)
        
        # Batch add items with GUI refresh
        total = len(items_to_add)
        print(f"\n📦 Adding {total} items to queue...")
        self.status_label.setText(f"Adding {total} items...")
        QtWidgets.QApplication.processEvents()
        
        for i, item in enumerate(items_to_add):
            self.queue.append(item)
            
            card = VideoCard(item, is_portrait=self.is_portrait_mode)
            card.roi_changed.connect(self.on_roi_changed)
            card.remove_requested.connect(self.on_remove_video)
            card.apply_all_requested.connect(self.on_apply_roi_to_all_from_card)
            self.video_cards.append(card)
            
            cols = 3 if self.is_portrait_mode else 2
            idx = len(self.video_cards) - 1
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col)
            
            # Process events every 5 items to keep GUI responsive
            if (i + 1) % 5 == 0:
                self.status_label.setText(f"Adding items... {i+1}/{total}")
                QtWidgets.QApplication.processEvents()
        
        self.update_mode_display()
        self.update_buttons()

    
    def on_roi_changed(self, item, x, y, w, h):
        print(f"ROI set for {os.path.basename(item.video_path)}: {x},{y},{w},{h}")
        self.update_buttons()
    
    def on_remove_video(self, item):
        if item in self.queue:
            idx = self.queue.index(item)
            self.queue.remove(item)
            
            card = self.video_cards[idx]
            
            # Stop any workers for this card
            if card in self.ocr_workers:
                w = self.ocr_workers[card]
                if w.isRunning(): w.stop(); w.wait()
                del self.ocr_workers[card]
            if card in self.inpaint_workers:
                w = self.inpaint_workers[card]
                if w.isRunning(): w.stop(); w.wait()
                del self.inpaint_workers[card]
            # Safety for singular workers if they match this item
            if self.ocr_worker and self.ocr_worker.isRunning():
                # No easy way to check if ocr_worker is for this item without storing it, 
                # but stopping is safer than crash. 
                # Actually, workers are usually sequential in batch.
                pass 
                
            card.cleanup()
            self.video_cards.remove(card)
            self.grid_layout.removeWidget(card)
            card.deleteLater()
            
            self.rebuild_grid()
            self.update_buttons()

    def on_clear_all(self):
        """Clear all videos from queue"""
        # Stop all workers first
        for worker_dict in [self.ocr_workers, self.inpaint_workers]:
            for card, worker in list(worker_dict.items()):
                if worker.isRunning():
                    worker.stop()
                    worker.wait()
            worker_dict.clear()
        
        # Stop singular workers if any
        for w in [self.ocr_worker, self.inpaint_worker, getattr(self, 'merge_worker', None)]:
            if w and hasattr(w, 'isRunning') and w.isRunning():
                if hasattr(w, 'stop'): w.stop()
                w.wait()
                
        self.queue = []
        self.rebuild_grid()
        self.update_buttons()

    def load_settings(self):
        """Load settings from registry"""
        settings = QSettings("Tool6", "BatchProcessor")
        
        # OCR Settings
        self.ocr_settings["frame_step"] = int(settings.value("frame_step", 2))
        self.ocr_settings["min_conf"] = float(settings.value("min_conf", 0.5))
        self.ocr_settings["sim_threshold"] = int(settings.value("sim_threshold", 50))
        # Important: recommended default 0.4s now
        self.ocr_settings["min_duration"] = float(settings.value("min_duration", 0.6))
        self.ocr_settings["grace"] = float(settings.value("grace", 0.0))
        self.ocr_settings["vse_mode"] = settings.value("vse_mode", "accurate")
        self.ocr_settings["workers"] = int(settings.value("workers", 4))
        self.ocr_settings["removesub_method"] = settings.value("removesub_method", "ai")
        
        # TTS Settings
        self.ocr_settings["tts_api_url"] = settings.value("tts_api_url", "http://192.168.1.75:8002/api2/convert-tts")
        self.ocr_settings["kokoro_voice"] = str(settings.value("kokoro_voice", "af_bella"))
        self.ocr_settings["edge_voice"] = str(settings.value("edge_voice", "vi-VN-HoaiMyNeural"))
        self.ocr_settings["tts_speed"] = float(settings.value("tts_speed", 1.2))
        self.ocr_settings["orig_video_speed"] = float(settings.value("orig_video_speed", 1.0))
        self.ocr_settings["uvr_vocal_remove"] = bool(settings.value("uvr_vocal_remove", False, type=bool))
        self.ocr_settings["uvr_strength"] = int(settings.value("uvr_strength", 100))
        # BGM files (stored as JSON list)
        import json
        bgm_raw = settings.value("bgm_files", "[]")
        try:
            self.ocr_settings["bgm_files"] = json.loads(bgm_raw) if isinstance(bgm_raw, str) else []
        except:
            self.ocr_settings["bgm_files"] = []
        self.ocr_settings["bgm_random"] = bool(settings.value("bgm_random", True, type=bool))
        self.ocr_settings["vol_bgm"] = float(settings.value("vol_bgm", 2.0))
        self.ocr_settings["vol_bgm_import"] = float(settings.value("vol_bgm_import", 1.0))
        
        sync_mode_val = settings.value("sync_mode", settings.value("dynamic_retime", True))
        if sync_mode_val is True: sync_mode_val = "stretch_video"
        elif sync_mode_val is False: sync_mode_val = "overlap"
        self.ocr_settings["sync_mode"] = str(sync_mode_val)
        self.ocr_settings["vol_tts"] = float(settings.value("vol_tts", 2.0))
        
        # Prompt Settings
        default_prompt = "Bạn là một chuyên gia dịch thuật hãy dịch cho tôi đoạn srt sau qua tiếng việt : "
        saved_prompt = settings.value("ocr_prompt", default_prompt)
        if not isinstance(saved_prompt, str):
            saved_prompt = default_prompt
        self.ocr_settings["ocr_prompt"] = saved_prompt
        
        # Load Target Language
        self.ocr_settings["target_lang"] = str(settings.value("target_lang", "Tiếng Việt"))
        
        # Restore process mode
        saved_mode = settings.value("process_mode", "srt_removesub")
        try:
            self.process_mode = ProcessMode(saved_mode)
        except (ValueError, KeyError):
            self.process_mode = ProcessMode.SRT_REMOVESUB
        print(f"\n📦 [LOAD] Restored process_mode = {self.process_mode}")
        
        # Merge Settings
        self.ocr_settings["merge_episodes"] = bool(settings.value("merge_episodes", False, type=bool))
        self.ocr_settings["merge_count"] = int(settings.value("merge_count", 4))
        
        # Window state
        if settings.value("maximized", False, type=bool):
            self.showMaximized()
            
    def save_settings(self):
        """Save settings to registry"""
        settings = QSettings("Tool6", "BatchProcessor")
        
        # OCR Settings
        settings.setValue("frame_step", self.ocr_settings["frame_step"])
        settings.setValue("min_conf", self.ocr_settings["min_conf"])
        settings.setValue("sim_threshold", self.ocr_settings["sim_threshold"])
        settings.setValue("min_duration", self.ocr_settings["min_duration"])
        settings.setValue("grace", self.ocr_settings["grace"])
        settings.setValue("workers", self.ocr_settings["workers"])
        settings.setValue("auto_translate", self.ocr_settings.get("auto_translate", True))
        settings.setValue("gemini_api_key", self.ocr_settings.get("gemini_api_key", ""))
        settings.setValue("gemini_model", self.ocr_settings.get("gemini_model", "gemini-3-pro-preview"))
        settings.setValue("removesub_method", self.ocr_settings.get("removesub_method", "ai"))
        
        # TTS Settings
        settings.setValue("tts_api_url", self.ocr_settings.get("tts_api_url", ""))
        settings.setValue("kokoro_voice", self.ocr_settings.get("kokoro_voice", "af_bella"))
        settings.setValue("edge_voice", self.ocr_settings.get("edge_voice", "vi-VN-HoaiMyNeural"))
        settings.setValue("tts_speed", self.ocr_settings.get("tts_speed", 1.2))
        settings.setValue("orig_video_speed", self.ocr_settings.get("orig_video_speed", 1.0))
        settings.setValue("uvr_vocal_remove", self.ocr_settings.get("uvr_vocal_remove", False))
        settings.setValue("uvr_strength", self.ocr_settings.get("uvr_strength", 100))
        # BGM files (stored as JSON list)
        import json
        settings.setValue("bgm_files", json.dumps(self.ocr_settings.get("bgm_files", [])))
        settings.setValue("bgm_random", self.ocr_settings.get("bgm_random", True))
        settings.setValue("vol_bgm", self.ocr_settings.get("vol_bgm", 2.0))
        settings.setValue("vol_bgm_import", self.ocr_settings.get("vol_bgm_import", 1.0))
        settings.setValue("sync_mode", self.ocr_settings.get("sync_mode", "stretch_video"))
        settings.setValue("vol_tts", self.ocr_settings.get("vol_tts", 2.0))
        
        # Prompt Settings
        settings.setValue("ocr_prompt", self.ocr_settings.get("ocr_prompt", ""))
        settings.setValue("target_lang", self.ocr_settings.get("target_lang", "Tiếng Việt"))
        
        # Merge Settings
        settings.setValue("merge_episodes", self.ocr_settings.get("merge_episodes", False))
        settings.setValue("merge_count", self.ocr_settings.get("merge_count", 4))
        
        # Window state
        settings.setValue("maximized", self.isMaximized())
        
        # Save process mode
        settings.setValue("process_mode", self.process_mode.value)

    def on_open_downloader(self):
        """Open the downloader GUI in a singleton manner"""
        if self.downloader_window is None:
            import download8movie_gui
            self.downloader_window = download8movie_gui.MainWindow()
            
        if not self.downloader_window.isVisible():
            self.downloader_window.show()
        else:
            self.downloader_window.raise_()
            self.downloader_window.activateWindow()
    
    def on_open_merge_dialog(self):
        """Open the merge dialog"""
        if not hasattr(self, '_merge_dialog') or self._merge_dialog is None:
            self._merge_dialog = MergeDialog(self)
        
        if not self._merge_dialog.isVisible():
            self._merge_dialog.show()
        else:
            self._merge_dialog.raise_()
            self._merge_dialog.activateWindow()
    
    def rebuild_grid(self):
        cols = 3 if self.is_portrait_mode else 2
        for i, card in enumerate(self.video_cards):
            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(card, row, col)
    
    def rebuild_all_cards(self):
        for card in self.video_cards:
            self.grid_layout.removeWidget(card)
            card.cleanup()
            card.deleteLater()
        
        self.video_cards.clear()
        
        for item in self.queue:
            card = VideoCard(item, is_portrait=self.is_portrait_mode)
            card.roi_changed.connect(self.on_roi_changed)
            card.remove_requested.connect(self.on_remove_video)
            card.apply_all_requested.connect(self.on_apply_roi_to_all_from_card)
            self.video_cards.append(card)
        
        self.rebuild_grid()
    
    def on_apply_roi_to_all_from_card(self, source_item):
        if not source_item.roi or source_item.roi == [0, 0, 0, 0]:
            QtWidgets.QMessageBox.warning(self, "No ROI", "Vui lòng vẽ vùng chọn ROI trên video này trước!")
            return
            
        count = 0
        for i, item in enumerate(self.queue):
            if item != source_item:
                item.roi = source_item.roi.copy()
                item.status = VideoStatus.ROI_SET
                self.video_cards[i].preview.roi = item.roi
                self.video_cards[i].preview.update()
                self.video_cards[i].update_status()
                count += 1
                
        if count > 0:
            self.update_buttons()
            QtWidgets.QMessageBox.information(self, "Đã sao chép ROI", f"Đã áp dụng vùng chọn cho {count} video khác!")
        else:
            QtWidgets.QMessageBox.information(self, "Info", "Không có video nào khác trong hàng đợi để áp dụng.")
            
    def update_buttons(self):
        # STT, TTS_MERGE, BURN_ONLY modes don't require ROI
        if self.process_mode in (ProcessMode.STT_EXTRACT, ProcessMode.TTS_MERGE, ProcessMode.TTS_MERGE_MULTI, ProcessMode.BURN_ONLY):
            self.btn_start.setEnabled(len(self.queue) > 0)
        else:
            all_have_roi = all(item.status == VideoStatus.ROI_SET for item in self.queue)
            self.btn_start.setEnabled(len(self.queue) > 0 and all_have_roi)
        
        total = len(self.queue)
        with_roi = sum(1 for item in self.queue if item.status == VideoStatus.ROI_SET)
        
        if self.process_mode == ProcessMode.SRT_ONLY:
            mode_text = "SRT Only"
        elif self.process_mode == ProcessMode.SRT_REMOVESUB:
            mode_text = "SRT + RemoveSub"
        elif self.process_mode == ProcessMode.REMOVE_LOGO:
            mode_text = "🚫 Remove Logo"
        elif self.process_mode == ProcessMode.STT_EXTRACT:
            mode_text = "🎤 Speech-to-Text"
        elif self.process_mode == ProcessMode.BURN_ONLY:
            mode_text = "🔥 Burn Sub"
        else:
            mode_text = "TTS + Merge"
            
        self.status_label.setText(f"{with_roi}/{total} ready - Mode: {mode_text}")
    
    # ============= QUEUE PROCESSING =============
    
    def on_start_queue(self):
        """Start processing queue"""
        # Check if modes require ROI
        if self.process_mode in (ProcessMode.SRT_REMOVESUB, ProcessMode.SRT_ONLY, ProcessMode.REMOVE_LOGO):
            # Check for videos without ROI
            videos_without_roi = []
            for i, item in enumerate(self.queue):
                if item.roi == [0, 0, 0, 0]:
                    videos_without_roi.append(os.path.basename(item.video_path))
            
            if videos_without_roi:
                roi_label = "vùng logo" if self.process_mode == ProcessMode.REMOVE_LOGO else "vùng subtitle"
                msg = (
                    f"⚠️ {len(videos_without_roi)} video(s) chưa chọn {roi_label}!\n\n"
                    f"Videos: {', '.join(videos_without_roi[:5])}"
                    + (f"... và {len(videos_without_roi) - 5} video khác" if len(videos_without_roi) > 5 else "")
                    + f"\n\nVui lòng click vào video và vẽ {roi_label} trước khi start!"
                )
                QtWidgets.QMessageBox.warning(self, "⚠️ Chưa chọn ROI", msg)
                return
        
        self.current_index = 0
        self.btn_start.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.process_next_video()

    
    def process_next_video(self):
        """Process next video in queue"""
        # Cleanup any lingering temporary files before starting next video
        self._cleanup_temp_files()
        
        if self.current_index >= len(self.queue):
            if self.ocr_settings.get("merge_episodes", False):
                self.merge_completed_videos()
            else:
                self.btn_start.setEnabled(True)
                self.btn_add.setEnabled(True)
                self.overall_progress.setValue(100)
                QtWidgets.QMessageBox.information(
                    self, "✅ Queue Complete",
                    f"All {len(self.queue)} videos processed successfully!"
                )
            return
        
        item = self.queue[self.current_index]
        card = self.video_cards[self.current_index]
        
        item.status = VideoStatus.PROCESSING
        card.update_status()
        
        progress = int((self.current_index / len(self.queue)) * 100)
        self.overall_progress.setValue(progress)
        

        
        # STT_EXTRACT mode: Speech-to-Text (no ROI needed)
        if self.process_mode == ProcessMode.STT_EXTRACT:
            self.start_stt(item, card)
            return

        # TTS_MERGE or BURN_ONLY mode: TTS + Burn subtitles / Burn Subtitles
        if self.process_mode in [ProcessMode.TTS_MERGE, ProcessMode.TTS_MERGE_MULTI, ProcessMode.BURN_ONLY]:
            self.process_tts_merge(item, card)
            return
        
        # REMOVE_LOGO mode: Skip OCR, just delogo the ROI on all frames
        if self.process_mode == ProcessMode.REMOVE_LOGO:
            self.start_remove_logo(item, card)
            return
        
        # Start OCR
        self.start_ocr(item, card)

    def start_stt(self, item, card):
        """Start Speech-to-Text for a single video using WhisperX"""
        card.status_label.setText("🎤 STT: Đang tách phụ đề...")
        card.status_label.setStyleSheet("padding: 5px; background: #e8d5f5; border-radius: 3px; font-size: 10px;")
        card.progress_bar.setVisible(True)
        card.progress_bar.setValue(0)

        self.stt_worker = STTWorker(
            video_path=item.video_path,
            language="en",
            model="large-v3"
        )
        self.stt_worker.progress.connect(lambda v: card.progress_bar.setValue(v))
        self.stt_worker.finished.connect(lambda srt_path: self.on_stt_finished(srt_path, item, card))
        self.stt_worker.error.connect(lambda err: self.on_stt_error(err, item, card))
        self.stt_worker.start()

    def on_stt_finished(self, srt_path, item, card):
        """Handle STT completion for batch processing"""
        item.srt_path = srt_path
        item.output_srt = srt_path
        item.status = VideoStatus.DONE
        card.update_status()
        card.update_srt_display()
        print(f"✅ STT finished: {os.path.basename(srt_path)}")

        self.current_index += 1
        self.process_next_video()

    def on_stt_error(self, error_msg, item, card):
        """Handle STT error for batch processing"""
        item.status = VideoStatus.ERROR
        item.error_msg = error_msg
        card.update_status()
        print(f"❌ STT Error: {error_msg}")

        self.current_index += 1
        self.process_next_video()


    def merge_completed_videos(self):
        """Merge videos into chunks of N"""
        import subprocess
        import os
        
        chunk_size = self.ocr_settings.get("merge_count", 4)
        
        # Collect all final output videos that exist
        final_videos = []
        for item in self.queue:
            if item.status == VideoStatus.DONE:
                video_base = os.path.splitext(item.video_path)[0]
                final_path = video_base + "_final.mp4"
                if os.path.exists(final_path):
                    final_videos.append(final_path)
                    
        if not final_videos:
            self.btn_start.setEnabled(True)
            self.btn_add.setEnabled(True)
            QtWidgets.QMessageBox.information(self, "No Videos", "Không có file nào để ghép.")
            return
            
        chunks = [final_videos[i:i + chunk_size] for i in range(0, len(final_videos), chunk_size)]
        
        self.overall_progress.setValue(100)
        self.status_label.setText(f"Đang ghép {len(final_videos)} video thành {len(chunks)} phần...")
        QtWidgets.QApplication.processEvents()
        
        for idx, chunk in enumerate(chunks):
            if len(chunk) <= 1:
                continue # No need to merge a single video
                
            first_base = os.path.splitext(chunk[0])[0]
            merged_out = first_base + f"_merged_part{idx+1}.mp4"
            concat_txt = first_base + "_concat.txt"
            
            with open(concat_txt, "w", encoding="utf-8") as f:
                for vid in chunk:
                    # Escape path for ffmpeg concat
                    escaped_vid = vid.replace("'", "'\\''")
                    f.write(f"file '{escaped_vid}'\n")
                    
            print(f"🔄 Merging chunk {idx+1} to {merged_out}...")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_txt,
                "-c", "copy",
                merged_out
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(concat_txt):
                os.remove(concat_txt)
                
        self.btn_start.setEnabled(True)
        self.btn_add.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "✅ Batch Complete",
            f"Đã xử lý {len(self.queue)} video và ghép thành {len(chunks)} phần!"
        )

    def start_ocr(self, item, card):
        """Step 1: start OCR Engine (PaddleOCR)"""
        if sip.isdeleted(card): return
        
        item.status = VideoStatus.PROCESSING
        card.status_label.setText("🔄 Extracting Subtitles (PaddleOCR)...")
        card.status_label.setStyleSheet("padding: 5px; background: #cce5ff; border-radius: 3px; font-size: 10px;")
        card.progress_bar.setVisible(True)
        card.progress_bar.setValue(0)
        
        print(f"🔬 Using PaddleOCR Engine for {item.video_path}")
        from tool6 import OCRWorker
        
        # Áp dụng triệt để VSE Mode từ Giao diện xuống thẳng Lõi OCRWorker
        vse_mode = self.ocr_settings.get("vse_mode", "accurate")
        if vse_mode == "ultra":
            # ~15 FPS (Quét gần như MỌI frame - không sót sub ngắn)
            eff_frame_step = 2
            eff_min_dur = 0.05      # Chấp nhận sub cực ngắn (50ms)
            eff_min_conf = 0.45     # Hạ ngưỡng conf để bắt sub mờ/nhanh
            eff_grace = 0.3         # Grace 300ms - tránh đóng segment khi OCR miss 1-2 frame
        elif vse_mode == "accurate":
            # ~6 FPS (Quét cẩn thận, đủ tốt cho 99% trường hợp)
            eff_frame_step = 5      
            eff_min_dur = 0.1       
            eff_min_conf = 0.55
            eff_grace = 0.15        # Grace nhỏ cho accurate
        elif vse_mode == "auto":
            # ~3 FPS (Dành cho video Subtitle hiện và giữ nguyên đủ lâu)
            eff_frame_step = 10
            eff_min_dur = 0.3
            eff_min_conf = 0.6
            eff_grace = 0.0
        else: # "fast"
            # ~1.5 FPS (Quét cực nhanh)
            eff_frame_step = 20
            eff_min_dur = 0.5
            eff_min_conf = 0.7
            eff_grace = 0.0
            
        self.ocr_worker = OCRWorker(
            video_path=item.video_path,
            roi=item.roi,
            frame_step=eff_frame_step,
            min_conf=eff_min_conf,
            sim_threshold=self.ocr_settings.get("sim_threshold", 50),
            min_duration=eff_min_dur,
            grace_sec=eff_grace,
            normalize_simplified=self.ocr_settings.get("normalize", True),
            num_workers=3,
            parent=self
        )
        
        self.ocr_worker.progress.connect(lambda p: card.set_progress(p, "ocr") if not sip.isdeleted(card) else None)
        self.ocr_worker.finished.connect(lambda srt, segs: self.on_ocr_done(item, card, srt, segs) if not sip.isdeleted(card) else None)
        self.ocr_worker.error.connect(lambda err: self.on_ocr_error(item, card, err) if not sip.isdeleted(card) else None)
        
        print(f"🔄 Processing {item.video_path}...")
        self.ocr_worker.start()
    
    def on_ocr_done(self, item, card, srt_path, segments):
        """Handle OCR completion"""
        if sip.isdeleted(card): return
        item.output_srt = srt_path
        item.segments = segments
        print(f"✅ OCR done: {srt_path} ({len(segments)} segments)")
        
        # Save segments to JSON for later RemoveSub Only mode
        import json
        segments_path = srt_path.replace(".srt", ".segments.json")
        # Convert tuples to lists for JSON serialization
        segments_serializable = []
        for seg in segments:
            seg_copy = seg.copy()
            if "box" in seg_copy and seg_copy["box"]:
                seg_copy["box"] = list(seg_copy["box"])
            segments_serializable.append(seg_copy)
        
        with open(segments_path, 'w', encoding='utf-8') as f:
            json.dump(segments_serializable, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved segments: {segments_path}")
        
        # FIX: Export Prompt + SRT to TXT file
        try:
            prompt = self.ocr_settings.get("ocr_prompt", "")
            if prompt:
                txt_path = srt_path.replace(".srt", ".txt")
                
                # Check if SRT file exists and read content
                if os.path.exists(srt_path):
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        srt_content = f.read()
                    
                    # Write Prompt + SRT
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(prompt + "\n\n" + srt_content)
                        
                    print(f"📄 Saved prompt TXT: {txt_path}")
        except Exception as e:
            print(f"⚠️ Failed to save prompt txt: {e}")
        
        self.finish_ocr_step(item, card, segments)



    def finish_ocr_step(self, item, card, segments):
        if sip.isdeleted(card): return
        
        # Nếu đang chọn mode SRT_ONLY thì hoàn tất ở đây, bỏ qua inpaint và TTS
        if self.process_mode == ProcessMode.SRT_ONLY:
            item.status = VideoStatus.DONE
            item.output_video = "" # SRT only doesn't produce an output video
            
            # Cập nhật UI an toàn bằng cách ghi trực tiếp
            card.status_label.setText("✅ Chỉ OCR: Hoàn thành SRT + TXT")
            card.status_label.setStyleSheet("padding: 5px; background: #d4edda; border-radius: 3px; font-size: 10px;")
            card.progress_bar.setVisible(False)
            card.update_status()
            
            print(f"✅ Hoàn tất item {item.video_path} (SRT Only Mode).")
            
            self.current_index += 1
            progress = int((self.current_index / len(self.queue)) * 100)
            self.overall_progress.setValue(progress)
            
            # Chạy tiếp file tiếp theo trong hàng đợi
            QtCore.QTimer.singleShot(100, self.process_next_video)
            return
            
        self.start_inpaint(item, card, segments)
    

    
    def open_subtitle_settings(self):
        """Open the advanced subtitle settings dialog"""
        from subtitle_settings_dialog import SubtitleSettingsDialog
        
        # Try to find a video to load for preview
        sample_video = None
        if len(self.queue) > 0:
            sample_video = self.queue[0].video_path
        
        dialog = SubtitleSettingsDialog(video_path=sample_video, parent=self)
        dialog.exec_()
    
    def process_tts_merge(self, item, card):
        """Process TTS + Merge mode - generate audio and burn subtitles"""
        import os
        
        video_base = os.path.splitext(item.video_path)[0]
        
        print(f"\n{'='*60}")
        print(f"🎬 [PROCESS] Video: {os.path.basename(item.video_path)}")
        print(f"🎬 [PROCESS] Pre-assigned srt_path: {item.srt_path}")
        print(f"🎬 [PROCESS] target_lang: {getattr(item, 'target_lang', None)}")
        print(f"🎬 [PROCESS] process_mode: {self.process_mode}")
        print(f"{'='*60}")
        
        # Only auto-detect if srt_path is not already set (e.g. from Multi-Lang mode)
        if not item.srt_path:
            new_srt = item.auto_detect_srt()
            if new_srt:
                item.srt_path = new_srt
                card.update_srt_display()
                print(f"🔍 [AUTO-DETECT] Found: {new_srt}")
            
        srt_path = item.srt_path
        

        
        if not srt_path:
            # Show dialog to manually select SRT file
            srt_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 
                f"Select SRT for {os.path.basename(item.video_path)}",
                os.path.dirname(item.video_path),
                "Subtitle Files (*.srt)"
            )
            
            if not srt_path:
                # User cancelled - skip this video
                print(f"⏭️ Skipped (no SRT selected)")
                item.status = VideoStatus.ERROR
                item.error_msg = "No SRT file selected"
                card.update_status()
                
                self.current_index += 1
                self.process_next_video()
                return

        
        print(f"📝 Found SRT: {srt_path}")
        
        # Output path - append SRT suffix if present (e.g. _hi, _ar)
        suffix = ""
        if srt_path:
            srt_name = os.path.splitext(os.path.basename(srt_path))[0]
            vid_name = os.path.basename(video_base)
            base_without_nosub = vid_name.replace("_nosub", "")
            if srt_name.startswith(base_without_nosub) and len(srt_name) > len(base_without_nosub):
                suffix = srt_name[len(base_without_nosub):]
            elif srt_name.startswith(vid_name) and len(srt_name) > len(vid_name):
                suffix = srt_name[len(vid_name):]
                
        output_path = video_base + suffix + "_final.mp4"
        
        # Create MergeWorker
        # Get TTS settings from ocr_settings or per-item overrides
        tts_api_url = self.ocr_settings.get("tts_api_url", "http://192.168.1.75:8002/api2/convert-tts")
        kokoro_voice = getattr(item, "kokoro_voice", None) or self.ocr_settings.get("kokoro_voice", "af_bella")
        tts_speed = self.ocr_settings.get("tts_speed", 1.0)
        orig_video_speed = self.ocr_settings.get("orig_video_speed", 1.0)
        uvr_vocal_remove = self.ocr_settings.get("uvr_vocal_remove", False)
        uvr_strength = self.ocr_settings.get("uvr_strength", 100)
        
        # Auto-pick MiniMax voice if target_lang is set but voice is not
        minimax_voice = getattr(item, "minimax_voice", None)
        if not minimax_voice and getattr(item, "target_lang", None):
            try:
                from minimax_bridge import get_minimax_voices
                mm_voices = get_minimax_voices()
                lang_name = item.target_lang
                if lang_name in mm_voices and len(mm_voices[lang_name]) > 0:
                    minimax_voice = mm_voices[lang_name][0].get("id")
            except Exception:
                pass
        
        if not minimax_voice:
            minimax_voice = self.ocr_settings.get("minimax_voice", "Vietnamese_Serene_Man")
        
        # Auto-pick Edge TTS voice if tts_api_url starts with "edge"
        if tts_api_url.strip().lower().startswith("edge") and ":" not in tts_api_url:
            # No voice specified in URL, auto-pick based on target_lang
            target_lang = getattr(item, "target_lang", None)
            if target_lang:
                from edge_tts_bridge import get_default_voice_for_lang
                edge_voice = get_default_voice_for_lang(target_lang)
            else:
                edge_voice = self.ocr_settings.get("edge_voice", "vi-VN-HoaiMyNeural")
            tts_api_url = f"edge:{edge_voice}"
            print(f"🆓 Edge TTS: auto-picked voice '{edge_voice}' for lang '{target_lang}'")
        
        # Pass ROI so subtitle appears in correct position
        roi_top = item.roi[1] if item.roi else None  # y position
        roi_bottom = (item.roi[1] + item.roi[3]) if item.roi else None  # y + height
        
        print(f"🎯 ROI for subtitle: top={roi_top}, bottom={roi_bottom}")
        print(f"🔊 TTS API: {tts_api_url} (Kokoro Voice: {kokoro_voice}), TTS Speed: {tts_speed}, Video Speed: {orig_video_speed}")
        
        self.merge_worker = MergeWorker(
            video=item.video_path,
            srt=srt_path,
            out=output_path,
            speed=tts_speed,
            orig_video_speed=orig_video_speed,
            uvr_enabled=uvr_vocal_remove,
            uvr_strength=uvr_strength,
            vol_bgm=self.ocr_settings.get("vol_bgm", 2.0),
            vol_tts=self.ocr_settings.get("vol_tts", 2.0),
            margin_v=60,  # Fallback if no ROI
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            tts_api_url=tts_api_url,
            kokoro_voice=kokoro_voice,
            sync_mode=self.ocr_settings.get("sync_mode", "stretch_video"),
            sub_overrides=item.sub_overrides,
            burn_sub_only=(self.process_mode == ProcessMode.BURN_ONLY),
            bgm_files=self.ocr_settings.get("bgm_files", []),
            bgm_random=self.ocr_settings.get("bgm_random", True),
            vol_bgm_import=self.ocr_settings.get("vol_bgm_import", 1.0),
            speaker_mapping=getattr(item, "speaker_mapping", None),
            speaker_json_path=getattr(item, "speaker_json_path", None),
            line_mapping=getattr(item, "line_mapping", None),
            minimax_voice=minimax_voice,
            minimax_model=self.ocr_settings.get("minimax_model", "speech-2.8-hd"),
            line_emotion=getattr(item, "line_emotion", None),
            target_lang=getattr(item, "target_lang", None)
        )


        
        # Connect signals
        self.merge_worker.log.connect(lambda msg: print(msg))
        self.merge_worker.progress.connect(lambda p: self.on_merge_progress(item, card, p) if not sip.isdeleted(card) else None)
        self.merge_worker.finished.connect(lambda out: self.on_merge_done(item, card, out) if not sip.isdeleted(card) else None)
        self.merge_worker.error.connect(lambda err: self.on_merge_error(item, card, err) if not sip.isdeleted(card) else None)
        
        # Start worker
        item.status = VideoStatus.PROCESSING
        card.update_status()
        self.merge_worker.start()
    
    def on_merge_progress(self, item, card, pct):
        if sip.isdeleted(card): return
        card.progress_bar.setVisible(True)
        card.progress_bar.setValue(pct)
        
        total = len(self.queue)
        if total > 0:
            overall = int((self.current_index * 100 + pct) / total)
            self.overall_progress.setValue(overall)
            
    def on_merge_done(self, item, card, output_path):
        """Handle TTS+Merge completion"""
        if sip.isdeleted(card): return
        print(f"✅ TTS+Merge done: {output_path}")
        
        item.status = VideoStatus.DONE
        item.output_srt = output_path  # Store final video path
        card.update_status()
        
        self.current_index += 1
        progress = int((self.current_index / len(self.queue)) * 100)
        self.overall_progress.setValue(progress)
        
        self.process_next_video()
    
    def on_merge_error(self, item, card, error_msg):
        """Handle TTS+Merge error"""
        if sip.isdeleted(card): return
        print(f"❌ TTS+Merge error: {error_msg}")
        
        item.status = VideoStatus.ERROR
        item.error_msg = error_msg
        card.update_status()
        
        reply = QtWidgets.QMessageBox.question(
            self, "Error",
            f"TTS+Merge failed for {os.path.basename(item.video_path)}:\n{error_msg}\n\nContinue with next video?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.current_index += 1
            self.process_next_video()
        else:
            self.btn_start.setEnabled(True)
            self.btn_add.setEnabled(True)
    
    def start_remove_logo(self, item, card):
        """Remove Logo mode: skip OCR, remove static logo from entire video"""
        if sip.isdeleted(card): return
        
        input_video = item.video_path
        base, ext = os.path.splitext(input_video)
        output_video = base + "_nologo" + ext
        
        x, y, w, h = item.roi
        
        # Get video info
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            item.status = VideoStatus.ERROR
            item.error_msg = "Cannot open video"
            card.update_status()
            self.current_index += 1
            self.process_next_video()
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        V_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        V_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        method = self.ocr_settings.get("removesub_method", "ai")
        
        # Use per-video logo time range from slider
        logo_start_sec = item.logo_start_sec
        logo_end_sec = item.logo_end_sec
        
        # If end is 0, use entire video
        if logo_end_sec <= 0:
            logo_end_sec = video_duration
        # Clamp to video duration
        logo_start_sec = max(0.0, min(logo_start_sec, video_duration))
        logo_end_sec = max(logo_start_sec, min(logo_end_sec, video_duration))
        
        logo_duration = logo_end_sec - logo_start_sec
        print(f"🚫 Remove Logo: {os.path.basename(input_video)}")
        print(f"   Method: {method} | ROI: [{x},{y},{w},{h}]")
        print(f"   Time range: {self._fmt_time(logo_start_sec)} → {self._fmt_time(logo_end_sec)} ({logo_duration:.1f}s / {video_duration:.1f}s total)")
        
        if method == "ai":
            # ===== AI Inpainting (VSR - STTN/LAMA) =====
            card.status_label.setText(f"🚫 Logo AI: {self._fmt_time(logo_start_sec)}→{self._fmt_time(logo_end_sec)}")
            card.status_label.setStyleSheet("padding: 5px; background: #d1ecf1; border-radius: 3px; font-size: 10px;")
            card.progress_bar.setVisible(True)
            card.progress_bar.setValue(0)
            
            # Build fake segments ONLY within the logo time range
            chunk_duration = 10.0  # 10s per segment
            fake_segments = []
            t = logo_start_sec
            while t < logo_end_sec:
                end_t = min(t + chunk_duration, logo_end_sec)
                fake_segments.append({
                    "start": t,
                    "end": end_t,
                    "text": "[LOGO]",
                    "box": (float(x), float(y), float(x + w), float(y + h)),
                    "stable_box": (float(x), float(y), float(x + w), float(y + h)),
                })
                t = end_t
            
            print(f"   Created {len(fake_segments)} segments for {logo_duration:.1f}s (skipping {video_duration-logo_duration:.1f}s)")
            
            from tool6 import InpaintWorker
            self.inpaint_worker = InpaintWorker(
                input_video=input_video,
                output_video=output_video,
                segments=fake_segments
            )
            
            self.inpaint_worker.progress.connect(lambda p: self._on_logo_progress(card, p) if not sip.isdeleted(card) else None)
            self.inpaint_worker.finished.connect(lambda out: self._on_logo_done(item, card, out) if not sip.isdeleted(card) else None)
            self.inpaint_worker.error.connect(lambda err: self._on_logo_error(item, card, err) if not sip.isdeleted(card) else None)
            self.inpaint_worker.start()
        else:
            # ===== FFmpeg Delogo (fast, lower quality) =====
            card.status_label.setText(f"🚫 Logo FFmpeg: {self._fmt_time(logo_start_sec)}→{self._fmt_time(logo_end_sec)}")
            card.status_label.setStyleSheet("padding: 5px; background: #fff3cd; border-radius: 3px; font-size: 10px;")
            card.progress_bar.setVisible(True)
            card.progress_bar.setValue(0)
            
            # Enforce bounds - delogo filter REQUIRES the box NOT to touch any video edge
            margin = 4
            x = max(margin, x)
            y = max(margin, y)
            if x + w >= V_W - margin:
                w = V_W - x - margin
            if y + h >= V_H - margin:
                h = V_H - y - margin
            # Enforce even coordinates (for YUV420)
            if x % 2 != 0: x += 1
            if y % 2 != 0: y += 1
            if w % 2 != 0: w -= 1
            if h % 2 != 0: h -= 1
            
            if w <= 4 or h <= 4:
                item.status = VideoStatus.ERROR
                item.error_msg = f"Invalid ROI after clamping: {x},{y},{w},{h} (video: {V_W}x{V_H})"
                card.update_status()
                self.current_index += 1
                self.process_next_video()
                return
            
            ffmpeg_bin = "ffmpeg"
            if os.path.exists("/usr/bin/ffmpeg"):
                ffmpeg_bin = "/usr/bin/ffmpeg"
            
            # Build delogo filter with time range using enable parameter
            if logo_start_sec > 0 or logo_end_sec < video_duration:
                vf = f"delogo=x={x}:y={y}:w={w}:h={h}:enable='between(t,{logo_start_sec:.2f},{logo_end_sec:.2f})'"
            else:
                vf = f"delogo=x={x}:y={y}:w={w}:h={h}"
            
            cmd = [
                ffmpeg_bin, "-y",
                "-i", input_video,
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "superfast",
                "-crf", "22",
                "-c:a", "copy",
                output_video
            ]
            
            print(f"⚡ Running: {' '.join(cmd)}")
            
            self._logo_worker = LogoRemoveWorker(cmd, output_video, video_duration, ffmpeg_bin, input_video, x, y, w, h)
            self._logo_worker.progress.connect(lambda p: self._on_logo_progress(card, p) if not sip.isdeleted(card) else None)
            self._logo_worker.finished.connect(lambda out: self._on_logo_done(item, card, out) if not sip.isdeleted(card) else None)
            self._logo_worker.error.connect(lambda err: self._on_logo_error(item, card, err) if not sip.isdeleted(card) else None)
            self._logo_worker.start()
    
    @staticmethod
    def _parse_time_str(s):
        """Parse MM:SS or HH:MM:SS string to seconds"""
        try:
            parts = s.strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            else:
                return float(parts[0])
        except:
            return 0.0
    
    @staticmethod
    def _fmt_time(sec):
        """Format seconds to MM:SS"""
        m = int(sec) // 60
        s = int(sec) % 60
        return f"{m:02d}:{s:02d}"
    
    def _on_logo_progress(self, card, pct):
        if sip.isdeleted(card): return
        card.progress_bar.setValue(pct)
        card.status_label.setText(f"🚫 Removing Logo: {pct}%")
    
    def _on_logo_done(self, item, card, output_path):
        if sip.isdeleted(card): return
        print(f"✅ Logo removed: {output_path}")
        
        item.status = VideoStatus.DONE
        item.output_srt = output_path
        card.update_status()
        
        self.current_index += 1
        progress = int((self.current_index / len(self.queue)) * 100)
        self.overall_progress.setValue(progress)
        self.process_next_video()
    
    def _on_logo_error(self, item, card, error_msg):
        if sip.isdeleted(card): return
        print(f"❌ Logo remove error: {error_msg}")
        
        item.status = VideoStatus.ERROR
        item.error_msg = error_msg
        card.update_status()
        
        reply = QtWidgets.QMessageBox.question(
            self, "Error",
            f"Logo removal failed for {os.path.basename(item.video_path)}:\n{error_msg}\n\nContinue with next video?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.current_index += 1
            self.process_next_video()
        else:
            self.btn_start.setEnabled(True)
            self.btn_add.setEnabled(True)
    
    def start_inpaint(self, item, card, segments):
        """Start inpainting for video"""
        input_video = item.video_path
        output_video = input_video.replace(".mp4", "_nosub.mp4")
        
        print(f"🧼 Starting RemoveSub: {output_video}...")
        
        # Decide which worker to use based on settings
        method = self.ocr_settings.get("removesub_method", "ai")
        
        if method == "delogo":
            print(f"🚀 Starting Fast Delogo for {input_video}")
            self.inpaint_worker = DelogoWorker(
                input_video=input_video,
                output_video=output_video,
                segments=segments,
                roi=item.roi
            )
        else: # Default to "ai"
            print(f"🤖 Starting AI Inpainting for {input_video}")
            self.inpaint_worker = InpaintWorker(
                input_video=input_video,
                output_video=output_video,
                segments=segments
            )
        
        self.inpaint_worker.progress.connect(lambda p: card.set_progress(p, "inpaint") if not sip.isdeleted(card) else None)
        self.inpaint_worker.finished.connect(lambda out: self.on_inpaint_done(item, card, out) if not sip.isdeleted(card) else None)
        self.inpaint_worker.error.connect(lambda err: self.on_inpaint_error(item, card, err) if not sip.isdeleted(card) else None)
        
        self.inpaint_worker.start()
    
    def on_inpaint_done(self, item, card, output_video):
        """Handle inpaint completion"""
        if sip.isdeleted(card): return
        print(f"✅ RemoveSub done: {output_video}")
        
        item.status = VideoStatus.DONE
        item.output_srt = output_video
        card.update_status()
        
        self.current_index += 1
        self.process_next_video()
    
    def on_ocr_error(self, item, card, error_msg):
        """Handle OCR error"""
        if sip.isdeleted(card): return
        print(f"❌ OCR error: {error_msg}")
        
        item.status = VideoStatus.ERROR
        item.error_msg = error_msg
        card.update_status()
        
        reply = QtWidgets.QMessageBox.question(
            self, "Error",
            f"OCR failed for {os.path.basename(item.video_path)}:\n{error_msg}\n\nContinue with next video?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.current_index += 1
            self.process_next_video()
        else:
            self.btn_start.setEnabled(True)
            self.btn_add.setEnabled(True)
    
    def on_inpaint_error(self, item, card, error_msg):
        """Handle inpaint error"""
        if sip.isdeleted(card): return
        print(f"❌ Inpaint error: {error_msg}")
        
        item.status = VideoStatus.ERROR
        item.error_msg = error_msg
        card.update_status()
        
        reply = QtWidgets.QMessageBox.question(
            self, "Error",
            f"RemoveSub failed for {os.path.basename(item.video_path)}:\n{error_msg}\n\nContinue with next video?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.current_index += 1
            self.process_next_video()
        else:
            self.btn_start.setEnabled(True)
            self.btn_add.setEnabled(True)
    
    def on_save_queue(self):
        """Save queue to JSON file"""
        if not self.queue:
            QtWidgets.QMessageBox.warning(self, "Empty Queue", "No videos to save!")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Queue", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        import json
        data = {
            "videos": [
                {
                    "path": item.video_path,
                    "roi": item.roi,
                    "status": item.status.value,
                }
                for item in self.queue
            ],
            "settings": self.ocr_settings,
            "process_mode": self.process_mode.value,
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        QtWidgets.QMessageBox.information(self, "Saved", f"Queue saved to {file_path}")
    
    def on_load_queue(self):
        """Load queue from JSON file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Queue", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Clear current queue
        for card in self.video_cards:
            card.cleanup()
            self.grid_layout.removeWidget(card)
            card.deleteLater()
        
        self.queue.clear()
        self.video_cards.clear()
        
        # Load settings
        if "settings" in data:
            self.ocr_settings = data["settings"]
        
        if "process_mode" in data:
            self.process_mode = ProcessMode(data["process_mode"])
        
        # Load videos
        for video_data in data["videos"]:
            if not os.path.exists(video_data["path"]):
                print(f"⚠️  Video not found: {video_data['path']}")
                continue
            
            item = VideoQueueItem(video_data["path"])
            if video_data.get("roi"):
                item.roi = video_data["roi"]
                item.status = VideoStatus.ROI_SET
            
            self.queue.append(item)
            
            card = VideoCard(item, is_portrait=self.is_portrait_mode)
            card.roi_changed.connect(self.on_roi_changed)
            card.remove_requested.connect(self.on_remove_video)
            card.apply_all_requested.connect(self.on_apply_roi_to_all_from_card)
            self.video_cards.append(card)
            
            # Update preview with ROI
            if item.roi:
                card.preview.roi = item.roi
                card.preview.update()
                card.update_status()
            
            cols = 3 if self.is_portrait_mode else 2
            idx = len(self.video_cards) - 1
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col)
        
        self.update_buttons()
        QtWidgets.QMessageBox.information(
            self, "Loaded", f"Loaded {len(self.queue)} videos from queue"
        )
    
    def on_apply_roi_to_all(self):
        """Apply clipboard ROI to all videos"""
        if not self.clipboard_roi:
            QtWidgets.QMessageBox.warning(
                self, "No ROI", "Copy ROI from a video first!"
            )
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, "Apply ROI to All",
            f"Apply ROI {self.clipboard_roi} to all {len(self.queue)} videos?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            for item, card in zip(self.queue, self.video_cards):
                item.roi = self.clipboard_roi.copy()
                item.status = VideoStatus.ROI_SET
                card.preview.roi = item.roi
                card.preview.update()
                card.update_status()
            
            self.update_buttons()
            QtWidgets.QMessageBox.information(
                self, "Done", f"Applied ROI to {len(self.queue)} videos!"
            )
    
    def on_clear_all(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All", "Remove all videos from queue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            for card in self.video_cards:
                card.cleanup()
                self.grid_layout.removeWidget(card)
                card.deleteLater()
            
            self.queue.clear()
            self.video_cards.clear()
            self.update_buttons()


def main():
    import os
    os.environ["LC_NUMERIC"] = "C"
    app = QtWidgets.QApplication(sys.argv)
    from PyQt5.QtCore import QLocale
    QLocale.setDefault(QLocale.c())
    app.setStyle("Fusion")
    
    window = BatchWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
