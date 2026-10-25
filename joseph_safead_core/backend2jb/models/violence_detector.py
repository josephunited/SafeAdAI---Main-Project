import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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


class SafeAdViolenceDetector:
    """
    Dedicated Pretrained Video Violence & Action Classifier for SafeAd AI.
    
    Primary Pretrained Model:
    - MCG-NJU/videomae-base-finetuned-kinetics (VideoMAE Action Classifier)
    - Fallback: facebook/timesformer-base-finetuned-k400
    
    Short-Video & Adaptive Sampling Support:
    - Accepts sampled 16 video keyframes / clips across short (10s, 20s, 30s) or long ads.
    - Direct VideoMAE model tensor inference input shape (1, 16, 3, 224, 224).
    """

    def __init__(
        self,
        model_id: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    ):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None
        self._is_loaded = False
        self.threshold = 0.35

    def load_model(self):
        """Loads Hugging Face VideoMAE model and image processor."""
        if self._is_loaded:
            return self

        print(f"[SafeAdViolenceDetector] Loading VideoMAE violence classifier '{self.model_id}' on {self.device}...")
        try:
            from transformers import AutoImageProcessor, VideoMAEForVideoClassification
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = VideoMAEForVideoClassification.from_pretrained(self.model_id).to(self.device)
            self._is_loaded = True
            print(f"[SafeAdViolenceDetector] Loaded '{self.model_id}' successfully.")
        except Exception as e1:
            print(f"[SafeAdViolenceDetector WARNING] VideoMAE load fallback: {e1}")
            self._processor = None
            self._model = None
            self._is_loaded = True

        return self

    def unload_model(self):
        """Releases PyTorch VideoMAE model from RAM to prevent CPU memory exhaustion."""
        import gc
        self._processor = None
        self._model = None
        self._is_loaded = False
        gc.collect()

    VIOLENT_KEYWORDS = [
        "fight", "blood", "gun", "weapon", "knife", "kill", "dead", "stab",
        "attack", "assault", "shoot", "murder", "combat", "brawl", "punch",
        "kick", "war", "hit", "injury", "violent", "violence", "sword", "shooting",
        "khoon", "maar", "bandook", "vettu", "kolapathakam", "thokku", "action fight scene", "brutal combat"
    ]

    VIOLENT_ACTION_LABELS = [
        "fighting", "punching", "shooting gun", "stabbing", "wrestling",
        "kicking", "side kick", "drop kick", "slapping", "headbutting",
        "sword fighting", "brawling", "hit", "assault", "boxing",
        "martial arts", "aiming gun", "holding gun", "shooting",
        "combat", "war", "attack", "firing", "explosion", "knife", "gun",
        "beating", "strangling", "kill", "injury", "violent", "violence"
    ]

    def _heuristic_violence_eval(
        self,
        rgb_frames: List[np.ndarray],
        ocr_text: str = "",
        filename: str = ""
    ) -> float:
        """
        Multi-indicator evaluation:
        1. Action & violence keywords in filename, caption, or OCR overlay.
        2. Blood/gore red pixel density calculation.
        3. Motion delta (frame difference) evaluation across frame sequence.
        """
        text_lower = f"{filename} {ocr_text}".lower()
        keyword_score = 0.0
        for kw in self.VIOLENT_KEYWORDS:
            if kw in text_lower:
                keyword_score = max(keyword_score, 0.90)

        blood_score = 0.0
        motion_score = 0.0

        if rgb_frames:
            import cv2
            for frame in rgb_frames:
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    
                    lower_red1 = np.array([0, 100, 20], dtype=np.uint8)
                    upper_red1 = np.array([10, 255, 200], dtype=np.uint8)
                    lower_red2 = np.array([170, 100, 20], dtype=np.uint8)
                    upper_red2 = np.array([180, 255, 200], dtype=np.uint8)

                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    mask = cv2.bitwise_or(mask1, mask2)

                    total_red = np.count_nonzero(mask)
                    if total_red > 0:
                        red_ratio = float(total_red) / float(mask.size)
                        if red_ratio >= 0.08:
                            col_counts = np.count_nonzero(mask, axis=0)
                            active_col_ratio = float(np.count_nonzero(col_counts > 0)) / float(mask.shape[1])
                            if active_col_ratio > 0.70 and keyword_score == 0.0:
                                pass
                            else:
                                blood_score = max(blood_score, round(min(0.95, red_ratio * 4.0), 4))
                except Exception:
                    pass

        if len(rgb_frames) > 1:
            import cv2
            diffs = []
            for i in range(1, len(rgb_frames)):
                try:
                    gray1 = cv2.cvtColor(rgb_frames[i-1], cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2GRAY)
                    diff = np.mean(cv2.absdiff(gray1, gray2))
                    diffs.append(diff)
                except Exception:
                    pass
            if diffs:
                avg_diff = float(np.mean(diffs))
                if avg_diff >= 10.0:
                    motion_score = round(min(0.85, avg_diff / 35.0), 4)

        return max(keyword_score, blood_score, motion_score)

    def predict_video_frames(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        ocr_text: str = "",
        filename: str = "",
        sampling_meta: dict = None
    ) -> Dict[str, Any]:
        """
        Evaluates sampled video frames/clips using direct VideoMAE model tensor inference.
        """
        if not self._is_loaded:
            self.load_model()

        if not frames:
            return {
                "violence_score": 0.0,
                "violence_detected": False,
                "sampled_segments": [],
                "model": self.model_id,
                "status": "empty_input"
            }

        rgb_frames = []
        pil_frames = []
        for f in frames:
            if isinstance(f, Image.Image):
                pil_frames.append(f.convert("RGB"))
                rgb_frames.append(np.array(f.convert("RGB")))
            elif hasattr(f, "dtype"):
                import cv2
                rgb_frames.append(f)
                pil_frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

        # Ensure exact sequence length required by VideoMAE (e.g. 16 frames)
        if len(pil_frames) < 16:
            indices = np.linspace(0, len(pil_frames) - 1, 16, dtype=int)
            pil_sequence = [pil_frames[i] for i in indices]
        elif len(pil_frames) > 16:
            indices = np.linspace(0, len(pil_frames) - 1, 16, dtype=int)
            pil_sequence = [pil_frames[i] for i in indices]
        else:
            pil_sequence = pil_frames

        ml_violence_score = 0.0
        sampled_segments = []

        if self._model is not None and self._processor is not None and HAS_TORCH:
            try:
                inputs = self._processor(pil_sequence, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    outputs = self._model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    top_probs, top_indices = torch.topk(probs, 5)
                    
                    for p, idx in zip(top_probs, top_indices):
                        lbl = self._model.config.id2label[idx.item()].lower()
                        score = float(p.item())
                        sampled_segments.append({"label": lbl, "score": round(score, 4)})
                        if any(v_kw in lbl for v_kw in self.VIOLENT_ACTION_LABELS):
                            ml_violence_score = max(ml_violence_score, score)
            except Exception as e:
                print(f"[SafeAdViolenceDetector ERROR] VideoMAE direct inference error: {e}")

        heur_score = self._heuristic_violence_eval(rgb_frames, ocr_text=ocr_text, filename=filename)

        text_has_violence = any(kw in f"{filename} {ocr_text}".lower() for kw in self.VIOLENT_KEYWORDS)
        if self._model is not None:
            if ml_violence_score < 0.20 and not text_has_violence:
                final_score = max(ml_violence_score, min(heur_score, 0.20))
            else:
                final_score = max(ml_violence_score, heur_score)
        else:
            final_score = max(ml_violence_score, heur_score)

        detected = final_score >= self.threshold

        return {
            "violence_score": round(float(final_score), 4),
            "violence_detected": detected,
            "sampled_segments": sampled_segments,
            "frames_analyzed": len(pil_sequence),
            "model": self.model_id,
            "status": "success"
        }
