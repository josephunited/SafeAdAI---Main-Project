import os
from typing import Optional
import requests
from backend.app.core.config import settings

class AIClientService:
    @staticmethod
    def process_advertisement(file_path: str, title: str = "", caption: str = "", pipeline_mode: Optional[str] = "learned_rf") -> dict:
        """
        Executes AI inference either via local `ai.pipeline` or remote Colab endpoint.
        """
        colab_url = settings.COLAB_API_URL
        
        # 1. Remote Colab API call if configured
        if colab_url and colab_url.strip():
            target_url = colab_url.strip().rstrip('/')
            print(f"[AIClientService] Checking Remote Colab Server health at: {target_url}...")
            
            # Fast 1.5s pre-flight health check to ensure Colab server is reachable
            is_colab_online = False
            try:
                h_res = requests.get(f"{target_url}/health", timeout=1.5)
                if h_res.status_code == 200:
                    is_colab_online = True
            except Exception as h_err:
                print(f"[AIClientService WARNING] Remote Colab ping failed ({h_err}). Falling back instantly to local AI pipeline...")

            if is_colab_online:
                try:
                    import mimetypes
                    mime_type, _ = mimetypes.guess_type(file_path)
                    mime_type = mime_type or 'application/octet-stream'
                    
                    headers = {"ngrok-skip-browser-warning": "true"}
                    with open(file_path, "rb") as f:
                        files = {"file": (os.path.basename(file_path), f, mime_type)}
                        data = {"title": title, "caption": caption}
                        response = requests.post(
                            f"{target_url}/predict",
                            headers=headers,
                            files=files,
                            data=data,
                            timeout=60
                        )
                        if response.status_code == 200:
                            print(f"[AIClientService] Remote Colab GPU Inference complete!")
                            return response.json()
                        else:
                            print(f"[AIClientService WARNING] Remote Colab responded with status {response.status_code}: {response.text}")
                except Exception as e:
                    print(f"[AI Client] Remote Colab request failed: {e}. Falling back to local pipeline...")
                
        # 2. Local AI Pipeline Execution
        try:
            try:
                from ai.pipeline import run_safead_inference
            except (ImportError, ModuleNotFoundError):
                import sys
                jb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "joseph_safead_core", "backend2jb"))
                if jb_dir not in sys.path:
                    sys.path.insert(0, jb_dir)
                try:
                    from ai.pipeline import run_safead_inference
                except (ImportError, ModuleNotFoundError):
                    from backend2jb.ai.pipeline import run_safead_inference
            return run_safead_inference(file_path, title, caption, pipeline_mode=pipeline_mode)
        except Exception as e:
            print(f"[AI Client ERROR] Local AI Pipeline execution error: {e}")
            return {
                "classification": "UNSAFE_FOR_ALL",
                "risk_score": None,
                "risk_score_available": False,
                "risk_category": "error_fallback",
                "explanation": f"AI Processing failure: {e}",
                "age_restriction": None,
                "action": "REJECT",
                "publishable": False,
                "violations": ["Processing Error"]
            }
