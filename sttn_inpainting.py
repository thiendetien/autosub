# sttn_inpainting.py
# -*- coding: utf-8 -*-
"""
STTN-based video inpainting for subtitle removal.
Uses Spatial-Temporal Transformer Network (ECCV 2020) to process
multiple frames at once for temporally coherent inpainting.

Requires: pretrained checkpoint at checkpoints/sttn.pth
Download from: https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.utils import Stack, ToTorchFormatTensor

# STTN inference parameters
STTN_W, STTN_H = 432, 240  # Default STTN resolution
REF_LENGTH = 10   # Reference frame sampling interval
NEIGHBOR_STRIDE = 5  # Neighbor stride for inference

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


def get_ref_index(neighbor_ids, length):
    """Sample reference frames from the whole video"""
    ref_index = []
    for i in range(0, length, REF_LENGTH):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


class STTNInpainter:
    """
    Wrapper for STTN video inpainting model.
    
    Unlike LaMa (single-image), STTN processes multiple frames together
    using spatial-temporal transformers for better temporal coherence.
    """
    
    def __init__(self, device='cuda', ckpt_path=None):
        """
        Initialize STTN model.
        
        Args:
            device: 'cuda' or 'cpu'
            ckpt_path: Path to sttn.pth checkpoint.
                       Default: checkpoints/sttn.pth relative to this file.
        """
        if ckpt_path is None:
            ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'sttn.pth')
        
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"STTN checkpoint not found: {ckpt_path}\n"
                f"Download from: https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view\n"
                f"Save to: {os.path.join(BASE_DIR, 'checkpoints', 'sttn.pth')}"
            )
        
        self.device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Import and create model
        from model.sttn import InpaintGenerator
        self.model = InpaintGenerator().to(self.device)
        
        # Load checkpoint
        data = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(data['netG'])
        self.model.eval()
        print(f"✅ STTN model loaded from {ckpt_path} on {self.device}")
    
    def inpaint_video_segment(self, frames_bgr, masks, progress_cb=None):
        """
        Inpaint a segment of video frames using STTN.
        
        Args:
            frames_bgr: List of BGR numpy arrays (H, W, 3) uint8
            masks: List of binary masks (H, W) uint8, 255=inpaint, 0=keep
            progress_cb: Optional callback(percent) for progress
            
        Returns:
            List of inpainted BGR numpy arrays (H, W, 3) uint8 (same size as input)
        """
        if not frames_bgr or not masks:
            return frames_bgr
        
        orig_h, orig_w = frames_bgr[0].shape[:2]
        video_length = len(frames_bgr)
        
        # Convert frames to PIL RGB and resize to STTN dimensions
        pil_frames = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_frames.append(pil_img.resize((STTN_W, STTN_H)))
        
        # Convert masks to PIL and resize
        pil_masks = []
        for m in masks:
            mask_bin = (m > 127).astype(np.uint8) * 255
            # Dilate mask slightly (like STTN test.py)
            mask_bin = cv2.dilate(mask_bin, cv2.getStructuringElement(
                cv2.MORPH_CROSS, (3, 3)), iterations=4)
            pil_mask = Image.fromarray(mask_bin)
            pil_masks.append(pil_mask.resize((STTN_W, STTN_H), Image.NEAREST))
        
        # Prepare tensors
        frames_np = [np.array(f).astype(np.uint8) for f in pil_frames]
        feats = _to_tensors(pil_frames).unsqueeze(0) * 2 - 1  # normalize to [-1, 1]
        
        binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in pil_masks]
        masks_tensor = _to_tensors(pil_masks).unsqueeze(0)
        
        feats = feats.to(self.device)
        masks_tensor = masks_tensor.to(self.device)
        
        comp_frames = [None] * video_length
        
        # Encode all frames
        with torch.no_grad():
            encoded = self.model.encoder(
                (feats * (1 - masks_tensor).float()).view(video_length, 3, STTN_H, STTN_W))
            _, c, feat_h, feat_w = encoded.size()
            encoded = encoded.view(1, video_length, c, feat_h, feat_w)
        
        # Complete holes by spatial-temporal transformers
        for f in range(0, video_length, NEIGHBOR_STRIDE):
            neighbor_ids = [i for i in range(
                max(0, f - NEIGHBOR_STRIDE),
                min(video_length, f + NEIGHBOR_STRIDE + 1))]
            ref_ids = get_ref_index(neighbor_ids, video_length)
            
            with torch.no_grad():
                pred_feat = self.model.infer(
                    encoded[0, neighbor_ids + ref_ids, :, :, :],
                    masks_tensor[0, neighbor_ids + ref_ids, :, :, :])
                pred_img = torch.tanh(self.model.decoder(
                    pred_feat[:len(neighbor_ids), :, :, :])).detach()
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(
                        np.uint8) * binary_masks[idx] + frames_np[idx] * (1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5
            
            if progress_cb:
                progress_cb(int(f * 100 / video_length))
        
        # Final composite and resize back to original resolution
        result_frames = []
        for f_idx in range(video_length):
            comp = np.array(comp_frames[f_idx]).astype(
                np.uint8) * binary_masks[f_idx] + frames_np[f_idx] * (1 - binary_masks[f_idx])
            
            # Convert back to BGR
            comp_bgr = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Resize back to original resolution
            if comp_bgr.shape[1] != orig_w or comp_bgr.shape[0] != orig_h:
                comp_bgr = cv2.resize(comp_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Composite: only replace the masked region in the original frame
            mask_orig = masks[f_idx]
            if mask_orig is not None:
                mask_resized = (mask_orig > 127).astype(np.float32)
                # Feather the mask edges for smooth blending
                mask_resized = cv2.GaussianBlur(mask_resized, (21, 21), 0)
                mask_3c = np.stack([mask_resized] * 3, axis=-1)
                
                original = frames_bgr[f_idx].astype(np.float32)
                inpainted = comp_bgr.astype(np.float32)
                blended = original * (1 - mask_3c) + inpainted * mask_3c
                result_frames.append(blended.astype(np.uint8))
            else:
                result_frames.append(frames_bgr[f_idx])
        
        if progress_cb:
            progress_cb(100)
        
        return result_frames


# Check if STTN is available
def is_sttn_available():
    """Check if STTN checkpoint exists"""
    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'sttn.pth')
    return os.path.isfile(ckpt_path)


def test_sttn():
    """Test if STTN can be loaded"""
    if not is_sttn_available():
        print("❌ STTN checkpoint not found")
        print(f"   Download from: https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view")
        print(f"   Save to: {os.path.join(BASE_DIR, 'checkpoints', 'sttn.pth')}")
        return False
    
    try:
        inpainter = STTNInpainter(device='cpu')
        print("✅ STTN available (CPU)")
        
        try:
            inpainter_gpu = STTNInpainter(device='cuda')
            print("✅ STTN available (GPU)")
        except Exception:
            print("⚠️  GPU not available for STTN, will use CPU")
        
        return True
    except Exception as e:
        print(f"❌ STTN load failed: {e}")
        return False


if __name__ == '__main__':
    test_sttn()
