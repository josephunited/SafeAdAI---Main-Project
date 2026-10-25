import requests
import os

SAFEAD_API_URL = os.getenv("SAFEAD_API_URL", "http://localhost:8002/api/v1")

class SafeAdClient:
    @staticmethod
    def moderate_ad(file, title, caption, pipeline_mode: str = "learned_rf"):
        url = f"{SAFEAD_API_URL}/advertisements/check"
        try:
            files = {"file": (file.name, file.getvalue(), file.type)}
            data = {"title": title, "caption": caption, "user_id": 1, "pipeline_mode": pipeline_mode}
            r = requests.post(url, files=files, data=data, timeout=600)
            if r.status_code == 200:
                return r.json(), None
            return None, r.text
        except Exception as e:
            return None, str(e)
            
    @staticmethod
    def get_pending_reviews():
        url = f"{SAFEAD_API_URL}/admin/pending"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json(), None
            return None, r.text
        except Exception as e:
            return None, str(e)
            
    @staticmethod
    def override_ad(ad_id, action, notes):
        url = f"{SAFEAD_API_URL}/admin/override"
        try:
            data = {"ad_id": ad_id, "action": action, "notes": notes}
            r = requests.post(url, json=data, timeout=10)
            if r.status_code == 200:
                return r.json(), None
            return None, r.text
        except Exception as e:
            return None, str(e)

    @staticmethod
    def get_user_feed(age_group):
        url = f"{SAFEAD_API_URL}/advertisements/user_feed"
        try:
            r = requests.get(url, params={"user_age_group": age_group}, timeout=10)
            if r.status_code == 200:
                return r.json(), None
            return None, r.text
        except Exception as e:
            return None, str(e)
