# video_inpainting.py
# -*- coding: utf-8 -*-
"""
Subtitle Removal with multiple methods:
- soft_blur: OpenCV blur (fast, decent)
- ai_inpaint: LaMa single-image inpainting (good quality)
- sttn_inpaint: STTN video inpainting (best temporal coherence)
- inpaint: OpenCV inpaint
- solid: Black fill

Features:
- Time padding to avoid 0.1s flashes
- Stable box per segment
- Edge-aware mask
- Distance-falloff blur (NO hard edges)
- Directional blur (subtitle horizontal)
- Temporal blending (prev + next frame)
"""

import os
# Skip slow model connectivity check for faster startup
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import subprocess





# ===================== CONFIG =====================
TIME_PAD_FRAMES = 4        # giãn thời gian ± N frame (tăng từ 3)
EDGE_LOW = 50              # giảm để detect edge tốt hơn (từ 60)
EDGE_HIGH = 130            # giảm (từ 140)
EDGE_DILATE = 6            # tăng để mask rộng hơn (từ 4)
PAD_RATIO = 0.04           # 4% chiều cao video (tăng từ 3%)

DIST_MAX_RATIO = 0.12      # phạm vi blur lan ra rộng hơn (từ 0.08)
ALPHA_BLUR = 51            # blur alpha mượt hơn (từ 31)
DIR_BLUR_X = 91            # blur ngang mạnh hơn (từ 61)
DIR_BLUR_Y = 31            # blur dọc (từ 21)


# ===================== MAIN =====================
def remove_subtitle_video(
    input_video: str,
    output_video: str,
    segments: list,
    method: str = "soft_blur",     # "soft_blur" | "ai_inpaint" | "sttn_inpaint" | "inpaint" | "solid"
    inpaint_radius: int = 3,
    box_pad_px: int = 10,
    mask_dilate_px: int = 12,
    horizontal_coverage: float = 0.8,  # 0.0-1.0, percentage of video width to cover
    ffmpeg_path: str | None = None,
    progress_cb=None,
):
    if not segments:
        _copy_video(ffmpeg_path, input_video, output_video)
        return output_video

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # ========= PREPROCESS SEGMENTS =========
    pad_sec = TIME_PAD_FRAMES / fps
    for s in segments:
        s["start"] = max(0.0, s["start"] - pad_sec)
        s["end"] += pad_sec
        s["stable_box"] = s.get("box")

    segments.sort(key=lambda s: (s["start"], s["end"]))

    # ========= VIDEO OUT =========
    tmp_noaudio = os.path.splitext(output_video)[0] + "__noaudio.mp4"
    out = cv2.VideoWriter(
        tmp_noaudio,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    seg_i = 0
    box_pad = max(box_pad_px, int(H * PAD_RATIO))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (mask_dilate_px, mask_dilate_px),
    )

    # ===== VSR Native Video Inpainter =====
    if method in ["sttn_inpaint", "ai_inpaint", "propainter", "lama"]:
        try:
            from vsr_bridge import remove_subtitle_vsr_native, VSR_LOADED
        except ImportError:
            VSR_LOADED = False

        if VSR_LOADED:
            cap.release()
            out.release()
            
            vsr_method = "sttn"
            if method == "propainter": vsr_method = "propainter"
            elif method == "lama" or method == "ai_inpaint": vsr_method = "lama"
            elif method == "sttn_inpaint": vsr_method = "sttn"
            
            try:
                return remove_subtitle_vsr_native(
                    input_video=input_video,
                    output_video=output_video,
                    segments=segments,
                    fps=fps,
                    total_frames=total,
                    method=vsr_method,
                    box_pad_px=box_pad_px,
                    horizontal_coverage=horizontal_coverage,
                    progress_cb=progress_cb,
                    ffmpeg_path=ffmpeg_path
                )
            except Exception as e:
                print(f"❌ VSR Native Engine failed: {e}. Falling back to soft_blur...")
                method = "soft_blur"
                cap = cv2.VideoCapture(input_video)
                tmp_noaudio = os.path.splitext(output_video)[0] + "__noaudio.mp4"
                out = cv2.VideoWriter(tmp_noaudio, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        else:
            print("⚠️ VSR Bridge not loaded. Falling back to soft_blur...")
            method = "soft_blur"

    prev_frame = None

    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        # ===== Peek next frame (temporal) =====
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ok2, next_frame = cap.read()
        if not ok2:
            next_frame = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        t = idx / fps

        while seg_i < len(segments) and t > segments[seg_i]["end"]:
            seg_i += 1

        active = None
        if seg_i < len(segments):
            s = segments[seg_i]
            if s["start"] <= t <= s["end"]:
                active = s

        if active and active.get("stable_box"):
            mask = _build_text_mask(
                frame,
                active["stable_box"],
                box_pad,
                kernel,
                horizontal_coverage,
            )

            if method == "soft_blur":
                frame = _reupload_blur(frame, mask, prev_frame, next_frame)
            elif method in ["ai_inpaint", "sttn_inpaint", "lama", "propainter"]: 
                # Fallback if VSR not triggered properly
                pass
            elif method == "inpaint":
                frame = cv2.inpaint(
                    frame,
                    mask.astype(np.uint8),
                    int(inpaint_radius),
                    cv2.INPAINT_TELEA,
                )
            elif method == "solid":
                frame[mask > 10] = (0, 0, 0)

        out.write(frame)
        prev_frame = frame.copy()

        if progress_cb and idx % 30 == 0:
            progress_cb(int(idx * 100 / total))

    cap.release()
    out.release()

    _mux_audio(ffmpeg_path, tmp_noaudio, input_video, output_video)

    if progress_cb:
        progress_cb(100)

    return output_video


def _remove_subtitle_sttn(
    input_video, output_video, segments, cap, fps, W, H, total,
    box_pad, kernel, horizontal_coverage, ffmpeg_path, progress_cb):
    """
    STTN batch processing: read all frames, generate masks, run STTN, write output.
    STTN processes frames in batches for temporal coherence.
    """
    print("🤖 Starting STTN video inpainting (batch mode)...")
    
    # Initialize STTN
    try:
        from sttn_inpainting import STTNInpainter
        sttn = STTNInpainter(device='cuda')
    except Exception:
        from sttn_inpainting import STTNInpainter
        sttn = STTNInpainter(device='cpu')
        print("⚠️  STTN running on CPU (slower)")
    
    # Phase 1: Read all frames and generate per-frame masks
    print("📖 Reading video frames...")
    all_frames = []
    all_masks = []
    seg_i = 0
    
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        
        t = idx / fps
        
        while seg_i < len(segments) and t > segments[seg_i]["end"]:
            seg_i += 1
        
        active = None
        if seg_i < len(segments):
            s = segments[seg_i]
            if s["start"] <= t <= s["end"]:
                active = s
        
        all_frames.append(frame)
        
        if active and active.get("stable_box"):
            mask = _build_text_mask(
                frame, active["stable_box"], box_pad, kernel, horizontal_coverage)
            mask_bin = (mask > 10).astype(np.uint8) * 255
            all_masks.append(mask_bin)
        else:
            all_masks.append(np.zeros((H, W), dtype=np.uint8))
        
        if progress_cb and idx % 100 == 0:
            progress_cb(int(idx * 30 / total))  # Phase 1 = 0-30%
    
    cap.release()
    
    # Check if any masks have content
    has_mask = any(m.max() > 0 for m in all_masks)
    if not has_mask:
        print("⚠️  No subtitle masks generated, copying video as-is")
        _copy_video(ffmpeg_path, input_video, output_video)
        return output_video
    
    # Phase 2: Run STTN inpainting
    print(f"🧠 Running STTN on {len(all_frames)} frames...")
    
    def sttn_progress(pct):
        if progress_cb:
            progress_cb(30 + int(pct * 0.5))  # Phase 2 = 30-80%
    
    result_frames = sttn.inpaint_video_segment(all_frames, all_masks, progress_cb=sttn_progress)
    
    # Phase 3: Write output video
    print("💾 Writing output video...")
    tmp_noaudio = os.path.splitext(output_video)[0] + "__noaudio.mp4"
    out = cv2.VideoWriter(
        tmp_noaudio,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    
    for idx, frame in enumerate(result_frames):
        out.write(frame)
        if progress_cb and idx % 100 == 0:
            progress_cb(80 + int(idx * 20 / len(result_frames)))  # Phase 3 = 80-100%
    
    out.release()
    
    # Mux audio
    _mux_audio(ffmpeg_path, tmp_noaudio, input_video, output_video)
    
    if progress_cb:
        progress_cb(100)
    
    print(f"✅ STTN inpainting complete: {output_video}")
    return output_video


# ===================== MASK =====================
def _build_text_mask(frame, box, pad, kernel, horizontal_coverage=0.8):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # AGGRESSIVE horizontal expansion to catch side text in multi-language subtitles
    # Strategy: Expand to cover most of the video width in subtitle region
    # This ensures we catch text on both sides (e.g., Chinese/Japanese characters)
    
    # Calculate center of detected box
    center_x = (x1 + x2) // 2
    
    # Expand horizontally based on horizontal_coverage parameter
    target_width = int(W * horizontal_coverage)
    half_width = target_width // 2
    
    x1 = max(0, center_x - half_width)
    y1 = max(0, y1 - pad)
    x2 = min(W - 1, center_x + half_width)
    y2 = min(H - 1, y2 + pad)

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, EDGE_LOW, EDGE_HIGH)
    edges = cv2.dilate(edges, None, iterations=EDGE_DILATE)

    mask = np.zeros((H, W), np.uint8)
    mask[y1:y2, x1:x2] = 255
    mask[y1:y2, x1:x2] = cv2.bitwise_or(
        mask[y1:y2, x1:x2], edges
    )

    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask.astype(np.float32)


# ===================== REUPLOAD BLUR =====================
def _reupload_blur(frame, mask, prev_frame=None, next_frame=None):
    H, W = frame.shape[:2]

    # ===== Distance falloff =====
    bin_mask = (mask > 10).astype(np.uint8)
    dist = cv2.distanceTransform(1 - bin_mask, cv2.DIST_L2, 5)

    max_dist = max(12, int(H * DIST_MAX_RATIO))
    alpha = np.clip(dist / max_dist, 0, 1)
    alpha = 1.0 - alpha
    alpha = cv2.GaussianBlur(alpha, (ALPHA_BLUR, ALPHA_BLUR), 0)

    # ===== Multi-stage blur (progressive) =====
    # Stage 1: Heavy directional blur
    blur1 = cv2.GaussianBlur(frame, (DIR_BLUR_X, DIR_BLUR_Y), 0)
    
    # Stage 2: Softer isotropic blur
    blur2 = cv2.GaussianBlur(frame, (41, 41), 0)
    
    # Blend two blur stages
    blur = cv2.addWeighted(blur1, 0.7, blur2, 0.3, 0)

    # ===== Enhanced temporal blend (median filter) =====
    if prev_frame is not None and next_frame is not None:
        # Stack frames for median
        frames_stack = np.stack([
            prev_frame.astype(np.float32),
            frame.astype(np.float32),
            next_frame.astype(np.float32)
        ], axis=0)
        
        # Median filter across time (removes outliers)
        temporal = np.median(frames_stack, axis=0)
        
        # Blend blur with temporal median
        blur = blur.astype(np.float32) * 0.5 + temporal * 0.5
    elif prev_frame is not None or next_frame is not None:
        # Fallback: use available frame
        ref = prev_frame if prev_frame is not None else next_frame
        temporal = ref.astype(np.float32)
        blur = blur.astype(np.float32) * 0.6 + temporal * 0.4

    base = frame.astype(np.float32)
    out = base.copy()

    # ===== Alpha blending with feathering =====
    for c in range(3):
        out[:, :, c] = base[:, :, c] * (1 - alpha) + blur[:, :, c] * alpha

    return out.astype(np.uint8)


# ===================== FFMPEG =====================
def _copy_video(ffmpeg_path, src, dst):
    if ffmpeg_path:
        subprocess.run([ffmpeg_path, "-y", "-i", src, "-c", "copy", dst], check=True)
    else:
        import shutil
        shutil.copy2(src, dst)


def _mux_audio(ffmpeg_path, video_noaudio, audio_src, out):
    if not ffmpeg_path:
        os.replace(video_noaudio, out)
        return

    subprocess.run([
        ffmpeg_path, "-y",
        "-i", video_noaudio,
        "-i", audio_src,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out
    ], check=True)

    try:
        os.remove(video_noaudio)
    except Exception:
        pass
