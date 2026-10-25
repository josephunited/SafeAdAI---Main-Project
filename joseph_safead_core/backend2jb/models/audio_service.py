import os
import time
from typing import Dict, Any, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from processing.audio_extraction import extract_audio_to_wav, cleanup_audio_file, ensure_ffmpeg_on_path

class SafeAdAudioService:
    """
    OpenAI Whisper Speech Transcription Service for SafeAd AI.
    
    Architecture:
    1. Extracts audio track from video.
    2. Detects audio presence (audio_available = True/False).
    3. Transcribes speech using OpenAI Whisper (`openai/whisper-small` or `whisper-base`).
    4. Passes speech transcript to text safety detector to generate normalized audio safety evidence.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._whisper_model = None
        self._pipe = None
        self._is_loaded = False
        ensure_ffmpeg_on_path()

    def load_model(self):
        """Loads OpenAI Whisper model lazily."""
        if self._is_loaded:
            return self

        ensure_ffmpeg_on_path()
        print(f"[SafeAdAudioService] Loading OpenAI Whisper '{self.model_name}' on {self.device}...")
        try:
            import whisper
            self._whisper_model = whisper.load_model(self.model_name, device=self.device)
            self._is_loaded = True
            print(f"[SafeAdAudioService] Loaded Whisper '{self.model_name}' successfully.")
        except Exception as e1:
            print(f"[SafeAdAudioService WARNING] OpenAI Whisper primary load fallback: {e1}")
            try:
                from transformers import pipeline
                device_idx = 0 if (HAS_TORCH and torch.cuda.is_available() and self.device == "cuda") else -1
                self._pipe = pipeline(
                    "automatic-speech-recognition",
                    model=f"openai/whisper-{self.model_name}",
                    device=device_idx
                )
                self._is_loaded = True
                print(f"[SafeAdAudioService] Loaded Hugging Face Whisper pipeline fallback.")
            except Exception as e2:
                print(f"[SafeAdAudioService WARNING] Whisper speech model offline: {e2}")
                self._whisper_model = None
                self._pipe = None
                self._is_loaded = True

        return self

    def unload_model(self):
        """Releases Whisper speech model from RAM to prevent CPU memory exhaustion."""
        import gc
        self._whisper_model = None
        self._pipe = None
        self._is_loaded = False
        gc.collect()

    def transcribe_wav(self, wav_path: str) -> Dict[str, Any]:
        """Transcribes a WAV audio file using Whisper."""
        if not wav_path or not os.path.exists(wav_path):
            return {"transcript": "", "language": "N/A", "status": "no_file"}

        ensure_ffmpeg_on_path()

        if not self._is_loaded:
            self.load_model()

        transcript = ""
        language = "English"

        if self._whisper_model is not None:
            try:
                res = self._whisper_model.transcribe(wav_path, fp16=(self.device == "cuda"))
                transcript = res.get("text", "").strip()
                language = res.get("language", "en")
            except Exception as e:
                print(f"[SafeAdAudioService ERROR] Whisper transcription error: {e}")
        elif self._pipe is not None:
            try:
                res = self._pipe(wav_path)
                if isinstance(res, dict):
                    transcript = res.get("text", "").strip()
            except Exception as e:
                print(f"[SafeAdAudioService ERROR] Transformers Whisper pipeline error: {e}")

        return {
            "transcript": transcript,
            "language": str(language).capitalize(),
            "status": "success" if transcript else "silence"
        }

    def process_video_audio(
        self,
        video_path: str,
        text_safety_detector = None
    ) -> Dict[str, Any]:
        """
        End-to-end audio processing for video advertisements.
        Returns normalized audio safety evidence dictionary.
        """
        ensure_ffmpeg_on_path()
        extraction_res = extract_audio_to_wav(video_path)
        
        if not extraction_res.get("audio_available", False):
            return {
                "audio_available": False,
                "transcript": "",
                "audio_safety": {
                    "violence": 0.0,
                    "adult": 0.0,
                    "child_safety": 0.0,
                    "overall_risk": 0.0
                },
                "status": extraction_res.get("message", "No audio track detected.")
            }

        wav_path = extraction_res.get("wav_path")
        stt_res = self.transcribe_wav(wav_path)
        cleanup_audio_file(wav_path)

        transcript = stt_res.get("transcript", "")

        audio_safety = {
            "violence": 0.0,
            "adult": 0.0,
            "child_safety": 0.0,
            "overall_risk": 0.0
        }

        if text_safety_detector is not None and transcript:
            safety_res = text_safety_detector.predict_text(transcript)
            audio_safety = {
                "violence": safety_res.get("violent", 0.0),
                "adult": safety_res.get("sexual", 0.0),
                "child_safety": safety_res.get("child_safety", 0.0),
                "overall_risk": safety_res.get("overall_text_risk", 0.0)
            }

        return {
            "audio_available": True,
            "transcript": transcript,
            "language": stt_res.get("language", "English"),
            "audio_safety": audio_safety,
            "status": "success"
        }
