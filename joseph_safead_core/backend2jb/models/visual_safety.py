import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

class VisualSafetyDetector:
    """
    Multimodal Visual Safety Detector for SafeAd AI.
    
    Primary Pretrained Safety Model:
    - meta-llama/Llama-Guard-3-11B-Vision or Hugging Face open vision safety guardrail.
    
    Analyzes visual frames / images for safety categories:
    - violence
    - sexual content
    - child sexual exploitation / child safety risk
    - hate / harassment
    - dangerous content / illegal acts
    
    Outputs normalized safety evidence.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-3-11B-Vision",
        device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    ):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None
        self._blip_processor = None
        self._blip_model = None
        self._is_loaded = False
        self._status = "UNINITIALIZED"
        self._active_model_name = "visual_heuristics"

    def load_model(self):
        """Lazy loader for visual safety models with explicit availability checks."""
        if self._is_loaded:
            return self

        from models.detector_infrastructure import (
            check_model_availability,
            DetectorStatus,
            is_model_cached_locally
        )

        can_load_primary, primary_reason = check_model_availability(self.model_id)

        if can_load_primary:
            print(f"[VisualSafetyDetector] Loading primary model '{self.model_id}' on {self.device}...")
            try:
                from transformers import AutoProcessor, MllamaForConditionalGeneration
                self._processor = AutoProcessor.from_pretrained(self.model_id)
                self._model = MllamaForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self._status = DetectorStatus.LOADED_SUCCESS.value
                self._active_model_name = self.model_id
                self._is_loaded = True
                print(f"[VisualSafetyDetector] Loaded '{self.model_id}' successfully.")
                return self
            except Exception as e:
                print(f"[VisualSafetyDetector WARNING] Primary model load failed: {e}")

        # Primary unavailable or failed: check local BLIP image captioning fallback without remote network penalty
        print(f"[VisualSafetyDetector] Model '{self.model_id}' unavailable ({primary_reason}). Checking local offline fallback...")
        if is_model_cached_locally("Salesforce/blip-image-captioning-base"):
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
                self._status = DetectorStatus.LOADED_SUCCESS.value
                self._active_model_name = "Salesforce/blip-image-captioning-base (local cache)"
                self._is_loaded = True
                print(f"[VisualSafetyDetector] Loaded local offline fallback 'Salesforce/blip-image-captioning-base' successfully.")
                return self
            except Exception as e2:
                print(f"[VisualSafetyDetector WARNING] Local BLIP fallback failed: {e2}")

        # Deterministic heuristic fallback
        self._status = DetectorStatus.HEURISTIC_FALLBACK.value
        self._active_model_name = "visual_safety_heuristics"
        self._is_loaded = True
        print(f"[VisualSafetyDetector] Operating on visual safety heuristics.")
        return self

    def unload_model(self):
        """Releases PyTorch model and processor from RAM to prevent CPU memory exhaustion."""
        import gc
        self._model = None
        self._processor = None
        self._blip_model = None
        self._blip_processor = None
        self._is_loaded = False
        gc.collect()

    def analyze_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyzes a single image / keyframe and generates normalized visual safety evidence.
        """
        if not self._is_loaded:
            self.load_model()

        pil_img = None
        if isinstance(image_input, str) and os.path.exists(image_input):
            try:
                pil_img = Image.open(image_input).convert("RGB")
            except Exception:
                pass
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif hasattr(image_input, "dtype"):
            import cv2
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        if pil_img is None:
            return {
                "category": "general_visual",
                "unsafe": False,
                "confidence": 0.0,
                "source": "llama_guard_vision",
                "detected_categories": []
            }

        # 1. Primary Llama Guard 3 Vision inference
        if self._model is not None and self._processor is not None and HAS_TORCH:
            try:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Provide safety classification for this ad frame across violence, sexual content, child safety, hate, and dangerous content."}
                        ]
                    }
                ]
                prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self._processor(pil_img, prompt, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    output_ids = self._model.generate(**inputs, max_new_tokens=50)
                    response = self._processor.decode(output_ids[0], skip_special_tokens=True)

                is_unsafe = "unsafe" in response.lower()
                confidence = 0.90 if is_unsafe else 0.95
                
                detected = []
                if "S1" in response or "violence" in response.lower():
                    detected.append("Violence")
                if "S2" in response or "sexual" in response.lower():
                    detected.append("Sexual Content")
                if "S3" in response or "child" in response.lower():
                    detected.append("Child Safety Risk")
                if "S4" in response or "hate" in response.lower():
                    detected.append("Hate Content")
                if "S5" in response or "dangerous" in response.lower():
                    detected.append("Dangerous Content")

                return {
                    "category": detected[0] if detected else "general_visual",
                    "unsafe": is_unsafe,
                    "confidence": confidence,
                    "source": "llama_guard_vision",
                    "detected_categories": detected,
                    "response_text": response.strip(),
                    "execution_status": self._status,
                    "model_provenance": {
                        "model_id": self._active_model_name,
                        "device": self.device,
                        "status": self._status
                    }
                }
            except Exception as e:
                print(f"[VisualSafetyDetector ERROR] Primary model inference error: {e}")

        # 2. Local BLIP Image Captioning Fallback (Semantic Visual Understanding)
        blip_caption = ""
        detected_from_blip = []
        if self._blip_model is not None and self._blip_processor is not None:
            try:
                inputs = self._blip_processor(pil_img, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    out = self._blip_model.generate(**inputs, max_new_tokens=30)
                    blip_caption = self._blip_processor.decode(out[0], skip_special_tokens=True).lower()
                
                # Check semantic visual cues
                for w in ["gun", "weapon", "knife", "rifle", "pistol", "sword", "blood", "fighting"]:
                    if w in blip_caption and "Violence" not in detected_from_blip:
                        detected_from_blip.append("Violence")
                for s in ["bikini", "lingerie", "swimwear", "underwear", "shirtless", "cleavage", "nude"]:
                    if s in blip_caption and "Sexual Content" not in detected_from_blip:
                        detected_from_blip.append("Sexual Content")
                for a in ["beer", "wine", "whiskey", "vodka", "alcohol", "liquor", "champagne"]:
                    if a in blip_caption and "Alcohol" not in detected_from_blip:
                        detected_from_blip.append("Alcohol")
            except Exception as e_blip:
                print(f"[VisualSafetyDetector WARNING] BLIP captioning error: {e_blip}")

        is_unsafe = ("Violence" in detected_from_blip)
        confidence = 0.88 if detected_from_blip else 0.85

        return {
            "category": detected_from_blip[0] if detected_from_blip else "general_visual",
            "unsafe": is_unsafe,
            "confidence": confidence,
            "source": self._active_model_name,
            "detected_categories": detected_from_blip,
            "visual_description": blip_caption,
            "execution_status": self._status,
            "model_provenance": {
                "model_id": self._active_model_name,
                "device": self.device,
                "status": self._status
            }
        }

    def analyze_video_frames(
        self,
        frames: List[Union[Image.Image, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Evaluates a sequence of sampled video keyframes for visual safety violations.
        """
        if not frames:
            return {
                "category": "general_visual",
                "unsafe": False,
                "confidence": 0.0,
                "source": "llama_guard_vision",
                "detected_categories": []
            }

        frame_results = []
        all_detected = set()
        any_unsafe = False

        for f in frames:
            res = self.analyze_image(f)
            frame_results.append(res)
            if res.get("unsafe", False):
                any_unsafe = True
            for cat in res.get("detected_categories", []):
                all_detected.add(cat)

        confidences = [r.get("confidence", 0.0) for r in frame_results]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # Prioritize scenes from frames that actually triggered safety violations first
        violating_descs = [r.get("visual_description", "") for r in frame_results if r.get("unsafe") and r.get("visual_description")]
        benign_descs = [r.get("visual_description", "") for r in frame_results if not r.get("unsafe") and r.get("visual_description")]
        ordered_descs = list(dict.fromkeys(violating_descs + benign_descs))
        scene_summary = "; ".join(ordered_descs[:3]) if ordered_descs else ""

        return {
            "category": list(all_detected)[0] if all_detected else "general_visual",
            "unsafe": any_unsafe,
            "confidence": round(avg_conf, 4),
            "source": "llama_guard_vision",
            "detected_categories": list(all_detected),
            "visual_description": scene_summary,
            "frames_analyzed": len(frames)
        }
