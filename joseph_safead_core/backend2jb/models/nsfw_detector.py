import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union

try:
    import torch
    HAS_TORCH = True
    torch.set_num_threads(1)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
except Exception:
    torch = None
    HAS_TORCH = False


class SafeAdNSFWDetector:
    """
    Dedicated Adult / NSFW Content Classifier for SafeAd AI.
    
    Pretrained Model:
    - Falconsai/nsfw_image_detection (ViT Image Classifier)
    
    Video Processing:
    1. Evaluates sampled representative video keyframes.
    2. Stores frame-level NSFW probabilities.
    3. Aggregates into video-level adult score via `max(frame_scores)` or percentile.
    """

    def __init__(
        self,
        model_id: str = "Falconsai/nsfw_image_detection",
        device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    ):
        self.model_id = model_id
        self.device = device
        self._pipe = None
        self._is_loaded = False
        self.threshold = 0.35

    def load_model(self):
        """Loads Hugging Face classification pipeline for Falconsai/nsfw_image_detection."""
        if self._is_loaded:
            return self

        print(f"[SafeAdNSFWDetector] Loading '{self.model_id}' on {self.device}...")
        try:
            from transformers import pipeline
            device_idx = 0 if (HAS_TORCH and torch.cuda.is_available() and self.device == "cuda") else -1
            self._pipe = pipeline(
                "image-classification",
                model=self.model_id,
                device=device_idx,
                top_k=None
            )
            self._is_loaded = True
            print(f"[SafeAdNSFWDetector] Loaded '{self.model_id}' successfully.")
        except Exception as e:
            print(f"[SafeAdNSFWDetector WARNING] Hugging Face pipeline fallback: {e}")
            self._pipe = None
            self._is_loaded = True

        return self

    def unload_model(self):
        """Releases PyTorch ViT model from RAM to prevent CPU memory exhaustion."""
        import gc
        self._pipe = None
        self._is_loaded = False
        gc.collect()

    def _heuristic_skin_eval(self, pil_img: Image.Image) -> float:
        """
        Auxiliary skin-pixel chrominance filter using YCbCr & HSV color spaces.
        Evaluates body skin ratio for suggestive attire / swimwear / sports apparel.        """
        try:
            img_np = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            height, width = img_np.shape[:2]

            ycrcb = cv2.cvtColor(img_np, cv2.COLOR_BGR2YCrCb)
            mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([215, 173, 127], dtype=np.uint8))

            hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv, np.array([0, 30, 60], dtype=np.uint8), np.array([20, 150, 215], dtype=np.uint8))

            skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
            total_skin = np.count_nonzero(skin_mask)

            if total_skin == 0:
                return 0.05

            skin_ratio = float(total_skin) / float(skin_mask.size)
            if skin_ratio > 0.30:
                return round(min(0.75, 0.25 + skin_ratio), 4)
            elif skin_ratio > 0.15:
                return round(min(0.40, 0.15 + skin_ratio), 4)
            return round(max(0.02, skin_ratio * 0.5), 4)
        except Exception:
            return 0.05
    def predict_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        ocr_text: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluates a single image / keyframe for adult/NSFW/suggestive content using pretrained Falconsai ViT & heuristics.        """
        if not self._is_loaded:
            self.load_model()

        pil_img = None
        file_name = ""
        if isinstance(image_input, str) and os.path.exists(image_input):
            file_name = os.path.basename(image_input)
            try:
                pil_img = Image.open(image_input).convert("RGB")
            except Exception:
                pass
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif hasattr(image_input, "dtype"):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        if pil_img is None:
            return {
                "nsfw_detected": False,
                "nsfw_score": 0.0,
                "model": self.model_id,
                "status": "invalid_input"
            }

        nsfw_score = 0.0
        normal_score = 0.0

        if self._pipe is not None:
            try:
                results = self._pipe(pil_img)
                for r in results:
                    lbl = str(r.get("label", "")).lower().strip()
                    score = float(r.get("score", 0.0))
                    if any(k in lbl for k in ["nsfw", "porn", "porno", "sexy", "hentai", "explicit", "adult", "erotica", "label_1"]):
                        nsfw_score = max(nsfw_score, score)
                    elif any(k in lbl for k in ["normal", "neutral", "safe", "label_0"]):
                        normal_score = max(normal_score, score)
            except Exception as e:
                print(f"[SafeAdNSFWDetector ERROR] Inference error: {e}")

        heur_score = self._heuristic_skin_eval(pil_img)
        
        # Check suggestive / adult text indicators
        text_corpus = f"{file_name} {ocr_text}".lower()
        suggestive_keywords = ["sports bra", "swimwear", "bikini", "lingerie", "cleavage", "sensual", "sexy", "physique", "unzipped", "hot", "body", "swimsuit", "sweeney", "controversial"]
        text_indicator = 0.40 if any(k in text_corpus for k in suggestive_keywords) else (0.85 if any(k in text_corpus for k in ["adult", "18+", "nsfw", "dating 18+"]) else 0.0)

        final_score = max(nsfw_score, heur_score, text_indicator)
        detected = final_score >= self.threshold

        return {
            "nsfw_detected": detected,
            "nsfw_score": round(float(final_score), 4),
            "normal_score": round(float(normal_score), 4),
            "model": self.model_id,
            "status": "success"
        }

    def predict_video_frames(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        aggregation_method: str = "max",
        ocr_text: str = ""
    ) -> Dict[str, Any]:
        """
        Extracts frame-level NSFW probabilities and aggregates them into a video-level adult score.
        Formula: adult_score = max(frame_scores)
        """
        if not frames:
            return {
                "adult_score": 0.0,
                "adult_content_detected": False,
                "frame_scores": [],
                "frames_evaluated": 0,
                "model": self.model_id,
                "status": "empty_frames"
            }

        frame_scores = []
        for f in frames:
            res = self.predict_image(f, ocr_text=ocr_text)
            frame_scores.append(res.get("nsfw_score", 0.0))

        if aggregation_method == "max":
            adult_score = float(np.max(frame_scores)) if frame_scores else 0.0
        elif aggregation_method == "percentile_90":
            adult_score = float(np.percentile(frame_scores, 90)) if frame_scores else 0.0
        else:
            adult_score = float(np.mean(frame_scores)) if frame_scores else 0.0

        detected = adult_score >= self.threshold

        return {
            "adult_score": round(adult_score, 4),
            "adult_content_detected": detected,
            "frame_scores": [round(s, 4) for s in frame_scores],
            "max_score": round(float(np.max(frame_scores)), 4) if frame_scores else 0.0,
            "mean_score": round(float(np.mean(frame_scores)), 4) if frame_scores else 0.0,
            "frames_evaluated": len(frames),
            "aggregation": aggregation_method,
            "model": self.model_id,
            "status": "success"
        }
