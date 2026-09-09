# vsr_bridge.py
import sys
import os
import cv2

vsr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vsr")
if vsr_dir not in sys.path:
    sys.path.insert(0, vsr_dir)

try:
    from backend.main import SubtitleRemover, SubtitleDetect
    import backend.config as vsr_config
    VSR_LOADED = True
except Exception as e:
    print(f"Error loading VSR backend: {e}")
    VSR_LOADED = False


class MockDetector:
    def __init__(self, sub_list):
        self.sub_list = sub_list
    
    def find_subtitle_frame_no(self, sub_remover=None):
        return self.sub_list
        
    def find_continuous_ranges_with_same_mask(self, subtitle_frame_no_box_dict):
        return SubtitleDetect.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        
    def get_scene_div_frame_no(self, v_path):
        return SubtitleDetect.get_scene_div_frame_no(v_path)
        
    def split_range_by_scene(self, intervals, points):
        return SubtitleDetect.split_range_by_scene(intervals, points)
        
    def filter_and_merge_intervals(self, intervals):
        return SubtitleDetect.filter_and_merge_intervals(intervals)

if VSR_LOADED:
    class AutosubSubtitleRemover(SubtitleRemover):
        def __init__(self, input_video, output_video, segments, fps, total_frames, box_pad=10, horizontal_coverage=0.85):
            super().__init__(input_video, sub_area=None, gui_mode=False)
            
            self.horizontal_coverage = horizontal_coverage
            # CRITICAL FIX: The parent __init__ calls importlib.reload(config),
            # which resets STTN_SKIP_DETECTION back to True! We MUST override it here
            # so it doesn't fall back to processing the full screen.
            import backend.main as vsr_main
            vsr_main.config.STTN_SKIP_DETECTION = False
            vsr_main.config.LAMA_SUPER_FAST = False
            
            self.video_out_name_noaudio = os.path.splitext(output_video)[0] + "__noaudio.mp4"
            # VSR main uses `video_temp_file.name` to write, and then replaces it?
            # Actually, `self.video_writer = cv2.VideoWriter(self.video_temp_file.name...`
            # When it finishes processing, does it move it? I will check later. 
            
            self.autosub_segments = segments
            # Replace the real detector with our mock so we don't do OCR again!
            sub_list = self._build_sub_list(fps, total_frames, box_pad)
            self.sub_detector = MockDetector(sub_list)
            
        def _build_sub_list(self, fps, total_frames, box_pad):
            # CRITICAL FIX: Limit continuous range length to prevent OOM crash.
            # STTN engine reads ALL frames in a continuous range into RAM at once.
            # For videos with constant text (logo/watermark), this can mean loading
            # ALL frames (e.g., 156GB for 15min 1080p video), crashing the system.
            # We insert 2-frame gaps every MAX_CONTINUOUS_FRAMES to force batch processing.
            MAX_CONTINUOUS_FRAMES = 300  # ~10s at 30fps, ~1.7GB RAM per batch
            
            sub_list = {}
            consecutive_count = 0
            
            for i in range(1, int(total_frames) + 1):
                t = (i - 1) / fps
                active = None
                for s in self.autosub_segments:
                    if s["start"] <= t <= s["end"]:
                        active = s
                        break
                
                if active and active.get("stable_box"):
                    consecutive_count += 1
                    
                    # Insert gap to break continuous ranges and prevent OOM
                    if consecutive_count > MAX_CONTINUOUS_FRAMES:
                        # Skip 2 frames to create a gap (STTN treats gaps as range boundaries)
                        consecutive_count = 0
                        continue
                    
                    x1, y1, x2, y2 = map(int, active["stable_box"])
                    W, H = self.frame_width, self.frame_height
                    pad = max(box_pad, int(H * 0.045))  # 4.5% chiều cao - phủ rộng hơn viền/shadow chữ
                    
                    center_x = (x1 + x2) // 2
                    
                    if self.horizontal_coverage and self.horizontal_coverage > 0:
                        target_width = int(W * self.horizontal_coverage)
                        half_width = target_width // 2
                        xmin = max(0, center_x - half_width)
                        xmax = min(W - 1, center_x + half_width)
                    else:
                        h_pad = pad * 3  # Tăng pad ngang cho an toàn
                        xmin = max(0, x1 - h_pad)
                        xmax = min(W - 1, x2 + h_pad)
                        
                    ymin = max(0, y1 - pad)
                    ymax = min(H - 1, y2 + pad)
                    
                    sub_list[i] = [(xmin, xmax, ymin, ymax)]
                else:
                    consecutive_count = 0
            
            print(f"📊 sub_list: {len(sub_list)} frames / {int(total_frames)} total ({len(sub_list)*100//max(1,int(total_frames))}% coverage)")
            return sub_list
            
        def run_with_progress(self, progress_cb, method="sttn"):
            import backend.main as vsr_main
            # Set algorithm
            if method == "propainter":
                vsr_main.config.MODE = vsr_main.config.InpaintMode.PROPAINTER
            elif method == "lama":
                vsr_main.config.MODE = vsr_main.config.InpaintMode.LAMA
            else:
                vsr_main.config.MODE = vsr_main.config.InpaintMode.STTN
                
            class MockTqdm:
                def __init__(self, total):
                    self.n = 0
                    self.total = total
                def update(self, n=1):
                    self.n += n
                    if progress_cb:
                        # Map to 30-80 range since OCR is 0-30 in tool6
                        pct = 30 + int(self.n * 50 / self.total)
                        progress_cb(pct)
                def close(self): pass

            tbar = MockTqdm(self.frame_count)
            
            try:
                if vsr_config.MODE == vsr_config.InpaintMode.STTN:
                    self.sttn_mode(tbar)
                elif vsr_config.MODE == vsr_config.InpaintMode.LAMA:
                    self.lama_mode(tbar)
                elif vsr_config.MODE == vsr_config.InpaintMode.PROPAINTER:
                    self.propainter_mode(tbar)
            finally:
                # Close writer
                self.video_writer.release()
                self.video_cap.release()
            
            # The video is written to self.video_temp_file.name
            # Return that path so the caller can mux audio
            import shutil
            shutil.copy2(self.video_temp_file.name, self.video_out_name_noaudio)
            
            # Clean up the temp file to avoid filling up the disk
            try:
                self.video_temp_file.close()
                if os.path.exists(self.video_temp_file.name):
                    os.remove(self.video_temp_file.name)
            except Exception as e:
                print(f"Warning: Failed to remove temp file {self.video_temp_file.name}: {e}")
            
            return self.video_out_name_noaudio

def remove_subtitle_vsr_native(input_video, output_video, segments, fps, total_frames, 
                               method="sttn", box_pad_px=10, horizontal_coverage=0.85, progress_cb=None, ffmpeg_path=None):
    if not VSR_LOADED:
        raise RuntimeError("VSR Backend failed to load.")
    
    print(f"🤖 Starting VSR Native mode: {method}")
    remover = AutosubSubtitleRemover(
        input_video, output_video, segments, fps, total_frames, 
        box_pad=box_pad_px, horizontal_coverage=horizontal_coverage
    )
    
    tmp_noaudio = remover.run_with_progress(progress_cb, method=method)
    
    # Mux audio
    if ffmpeg_path:
        import subprocess
        subprocess.run([
            ffmpeg_path, "-y",
            "-i", tmp_noaudio,
            "-i", input_video,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_video
        ], check=True)
        try:
            os.remove(tmp_noaudio)
        except:
            pass
    else:
        import shutil
        shutil.copy2(tmp_noaudio, output_video)
        
    if progress_cb:
        progress_cb(100)
        
    return output_video
