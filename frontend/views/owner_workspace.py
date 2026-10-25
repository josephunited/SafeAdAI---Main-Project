import streamlit as st
import os
import sys
import time
import requests
from datetime import datetime

# Inject backend2jb into sys.path to allow Joseph's true modules to load exactly as they do in his branch
jb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "joseph_safead_core/backend2jb"))
if jb_dir not in sys.path:
    sys.path.insert(0, jb_dir)

# Try importing the local fallback inference directly as done in his UI
try:
    from ai.pipeline import run_safead_inference
except ImportError:
    run_safead_inference = None

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002/api/v1")

def check_backend_online():
    try:
        resp = requests.get("http://127.0.0.1:8002/health", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False


def owner_dashboard():
    # Insert custom styling exactly from Joseph's branch
    st.markdown("""
        <style>
        .badge-reject {
            background-color: #ef4444;
            color: #ffffff;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .badge-hitl {
            background-color: #8b5cf6;
            color: #ffffff;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .badge-18 {
            background-color: #f59e0b;
            color: #ffffff;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .badge-14 {
            background-color: #3b82f6;
            color: #ffffff;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .badge-approve {
            background-color: #10b981;
            color: #ffffff;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }
        .matrix-card {
            background-color: transparent;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    try:
        from frontend.styles.instagram_theme import render_instagram_logo
        render_instagram_logo(subtitle="CREATOR STUDIO")
    except Exception:
        pass
        
    st.header("📢 Brand Owner Advertisement Safety Portal")
    
    if "owner_action_msg" in st.session_state and st.session_state["owner_action_msg"]:
        msg = st.session_state["owner_action_msg"]
        if "Pending" in msg:
            st.info(msg)
        elif "Unsafe" in msg:
            st.error(msg)
        else:
            st.success(msg)
        st.session_state["owner_action_msg"] = None
        
    col_up, col_res = st.columns([1, 1.2])
    
    with col_up:
        st.subheader("SECTION 1 — Upload Advertisement Creative")
        
        ad_title = st.text_input("Creative Title", placeholder="e.g. Organic Apple Juice / Casino Slot Promo / Paisa Double Scheme")
        ad_caption = st.text_area("Ad Copy / Caption", placeholder="e.g. Double your money guaranteed / Fresh organic apples")
        uploaded_file = st.file_uploader("Select Image or Video Media File", type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_size_kb = round(len(file_bytes) / 1024.0, 1)
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            media_type = "video" if file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"] else "image"
            
            st.markdown("#### Media File Details")
            st.write(f"- **Filename**: `{uploaded_file.name}`")
            st.write(f"- **Media Type**: `{media_type.upper()} ({file_ext})`")
            st.write(f"- **File Size**: `{file_size_kb} KB` ({len(file_bytes)} bytes)")
            st.write(f"- **Upload Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
            
            # PRIVACY REQUIREMENT: Do NOT render media player or image on Brand Owner page
            
            # Moderation Pipeline Mode Configuration (Phase 8E Learned RF Integration)
            st.markdown("#### ⚙️ Moderation Engine Configuration")
            mode_options = [
                "Learned Random Forest (Default Experimental)",
                "Safety-Prioritized Moderation (Learned RF + HITL Uncertainty Gate)",
                "Candidate B (Enhanced Alcohol Heuristic)",
                "Production Baseline (Frozen Heuristic)",
            ]
            selected_mode_label = st.selectbox(
                "Select Moderation Pipeline Mode",
                mode_options,
                index=0,
                help="Learned Random Forest is the new Phase 8E default experimental fusion engine. Heuristic baseline and Candidate B remain accessible."
            )
            mode_map = {
                "Learned Random Forest (Default Experimental)": "learned_rf",
                "Safety-Prioritized Moderation (Learned RF + HITL Uncertainty Gate)": "learned_rf_hitl",
                "Candidate B (Enhanced Alcohol Heuristic)": "candidate_b",
                "Production Baseline (Frozen Heuristic)": "production",
            }
            active_pipeline_mode = mode_map.get(selected_mode_label, "learned_rf")
            
            st.markdown("---")
            if st.button("🚀 ANALYZE ADVERTISEMENT", type="primary", use_container_width=True):
                if not ad_title:
                    st.error("Please provide a Creative Title before starting analysis.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    progress_bar.progress(15)
                    status_text.info("[1/4] Validating File & Extracting Media Keyframes...")
                    
                    progress_bar.progress(40)
                    status_text.info("[2/4] Running EasyOCR & Llama Guard Visual Safety...")
                    
                    progress_bar.progress(70)
                    status_text.info("[3/4] VideoMAE Violence, Falconsai NSFW, & Whisper Speech Transcription...")
                    
                    progress_bar.progress(90)
                    status_text.info("[4/4] Executing Two-Stage Policy Engine & Ad Risk Analysis...")
                    
                    uploads_dir = os.path.join(jb_dir, "uploads")
                    temp_path = os.path.join(uploads_dir, f"upload_{int(time.time())}_{uploaded_file.name}")
                    os.makedirs(uploads_dir, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                        
                    report = None
                    if check_backend_online():
                        try:
                            files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "video/mp4")}
                            data = {"title": ad_title, "caption": ad_caption, "user_id": 1}
                            resp = requests.post(f"{BACKEND_URL}/advertisements/upload", files=files, data=data, timeout=30)
                            if resp.status_code == 200:
                                ad_data = resp.json()
                                st.session_state["current_ad_id"] = ad_data["id"]
                                
                                analyze_resp = requests.post(
                                    f"{BACKEND_URL}/advertisements/{ad_data['id']}/analyze",
                                    params={"pipeline_mode": active_pipeline_mode},
                                    timeout=600
                                )
                                if analyze_resp.status_code == 200:
                                    report = analyze_resp.json()
                        except Exception as backend_err:
                            print(f"[Owner Workspace] Backend API call error: {backend_err}")

                    if not report:
                        if run_safead_inference:
                            try:
                                report = run_safead_inference(
                                    temp_path,
                                    ad_title,
                                    ad_caption,
                                    pipeline_mode=active_pipeline_mode
                                )
                                st.session_state["current_ad_id"] = report.get("ad_id", 101)
                                st.session_state["last_temp_path"] = temp_path
                                st.session_state["last_ad_title"] = ad_title
                                st.session_state["last_ad_caption"] = ad_caption
                            except Exception as infer_err:
                                st.error(f"Inference processing error: {infer_err}")
                                report = {
                                    "classification": "REQUIRES_HUMAN_REVIEW",
                                    "publication_action": "HUMAN_REVIEW",
                                    "risk_score": 50.0,
                                    "confidence": 0.5,
                                    "explanation": f"Automated processing interrupted: {infer_err}. Routed to Human Review.",
                                    "detected_categories": ["Processing Interrupted"],
                                    "policy_stage": "Fail-Safe Human Review"
                                }
                        else:
                            st.error("Inference module not found.")
                        
                    progress_bar.progress(100)
                    status_text.success("Analysis Complete!")
                    st.session_state["moderation_report"] = report
                    st.session_state["confirm_delete"] = False
                    
    with col_res:
        st.subheader("SECTION 2 — SafeAd AI Analysis Result")
        if "moderation_report" in st.session_state and st.session_state["moderation_report"]:
            rep = st.session_state["moderation_report"]
            
            classification = rep.get("classification", "UNSAFE_FOR_ALL")
            action = rep.get("publication_action", "REJECT")
            risk_score = rep.get("risk_score", 0.0)
            confidence = rep.get("confidence", 0.85)
            requires_hitl = rep.get("requires_human_review", False) or action == "HUMAN_REVIEW"
            ad_id = st.session_state.get("current_ad_id", rep.get("ad_id", 1))
            
            # Prominent Classification Badge Display
            if classification == "UNSAFE_FOR_ALL" or action == "REJECT":
                st.markdown("<div class='badge-reject'>UNSAFE FOR ALL — REJECT DO NOT PUBLISH</div>", unsafe_allow_html=True)
            elif classification == "SAFE_18_PLUS":
                st.markdown("<div class='badge-18'>18+ — AGE RESTRICTED</div>", unsafe_allow_html=True)
            elif classification == "SAFE_14_PLUS":
                st.markdown("<div class='badge-14'>14+ — AGE RESTRICTED</div>", unsafe_allow_html=True)
            elif classification == "REQUIRES_HUMAN_REVIEW":
                st.markdown("<div class='badge-hitl'>REQUIRES HUMAN REVIEW</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge-approve'>SAFE FOR ALL — APPROVED</div>", unsafe_allow_html=True)
                
            # Engine Stage and Mode indicator
            is_learned_rf = rep.get("pipeline_mode", "") in ["learned_rf", "learned_rf_hitl", "candidate_e", "candidate_e_hitl"] or "LEARNED" in str(rep.get("policy_stage", ""))
            mode_badge = rep.get("pipeline_mode", "learned_rf").upper()
            st.caption(f"🔧 **Moderation Engine**: `{mode_badge}` | **Policy Stage**: `{rep.get('policy_stage', 'Standard Fusion')}`")

            # Metrics Row — Uncalibrated RF vote shares are explicitly labeled
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Risk Score", f"{risk_score:.1f} / 100")
            if is_learned_rf:
                mcol2.metric("RF Vote Share", f"{int(confidence * 100)}% (Uncalibrated)")
            else:
                mcol2.metric("Confidence", f"{int(confidence * 100)}%")
            mcol3.metric("Latency", f"{rep.get('total_processing_time_seconds', 0.0):.2f}s")
            
            if is_learned_rf:
                st.caption("ℹ️ *Note: Random Forest vote share represents tree ensemble consensus and is not a calibrated posterior probability.*")
            
            st.markdown(f"**Explanation**:\n\n{rep.get('explanation', '')}")
            
            # Central SafeAdAssessment Matrix Summary
            st.markdown("### 📊 Safety Evidence Summary")
            ev = rep.get("evidence", {})
            matrix = ev if "visual" in ev else rep.get("assessment_matrix", {})
            
            st.markdown("<div class='matrix-card'>", unsafe_allow_html=True)
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.markdown(f"• **Extracted OCR Text**: *\"{rep.get('extracted_ocr', '') or 'None detected'}\"*")
                st.markdown(f"• **Visual Safety**: `{'UNSAFE' if matrix.get('visual', {}).get('unsafe') else 'SAFE'}`")
                st.markdown(f"• **Adult / NSFW Score**: `{matrix.get('visual', {}).get('adult', 0.0):.2f}`")
            with e_col2:
                st.markdown(f"• **Audio Speech Transcript**: *\"{rep.get('audio_transcript', '') or 'No audio'}\"*")
                st.markdown(f"• **Video Kinetic Violence**: `{matrix.get('video', {}).get('violence', 0.0):.2f}`")
                st.markdown(f"• **Scam / Deceptive Risk**: `{matrix.get('advertisement_risks', {}).get('scam', 0.0):.2f}`")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # BRAND OWNER WORKFLOW ACTIONS
            st.markdown("---")
            st.subheader("SECTION 3 — Choose Brand Owner Action")
            st.info("ℹ️ **Policy Enforcement**: The AI safety classification cannot be manually overridden by the Brand Owner.")
            
            can_publish = classification not in ["UNSAFE_FOR_ALL", "REQUIRES_HUMAN_REVIEW"] and rep.get("publishable", True)
            if not can_publish:
                if classification == "UNSAFE_FOR_ALL":
                    st.error("🚨 **Prohibited Content**: This advertisement was rejected and cannot be published to any user feed.")
                elif classification == "REQUIRES_HUMAN_REVIEW" or not rep.get("publishable", True):
                    st.warning("⏳ **Human Review Required**: Publication is withheld pending human moderator decision. Please send for review below.")

            b_act1, b_act2 = st.columns(2)
            b_act3, b_act4 = st.columns(2)
            
            with b_act1:
                accept_type = "primary" if can_publish else "secondary"
                if st.button("✅ ACCEPT AI DECISION", use_container_width=True, type=accept_type, disabled=not can_publish):
                    if check_backend_online():
                        try:
                            requests.post(f"{BACKEND_URL}/advertisements/{ad_id}/accept-ai", timeout=5)
                        except Exception:
                            pass
                    
                    if "global_ads" not in st.session_state:
                        st.session_state["global_ads"] = []
                        
                    # Find if already added
                    ad_exists = any(a.get("ad_id") == ad_id for a in st.session_state["global_ads"])
                    
                    if not ad_exists and can_publish:
                        feed_ad = {
                            "ad_id": ad_id,
                            "classification": classification,
                            "title": rep.get("title", st.session_state.get("last_ad_title", "Advertisement")),
                            "caption": rep.get("caption", st.session_state.get("last_ad_caption", "")),
                            "file_path": st.session_state.get("last_temp_path", "")
                        }
                        st.session_state["global_ads"].append(feed_ad)

                    st.session_state["owner_action_msg"] = f"✅ Advertisement accepted and moved to the {classification} feed."
                    st.session_state["moderation_report"] = None
                    st.session_state["current_ad_id"] = None
                    st.rerun()
                        
            with b_act2:
                if st.button("🗑️ DELETE ADVERTISEMENT", use_container_width=True):
                    st.session_state["confirm_delete"] = True
                    
            with b_act3:
                if st.button("🔄 UPLOAD ANOTHER ADVERTISEMENT", use_container_width=True):
                    st.session_state["moderation_report"] = None
                    st.session_state["current_ad_id"] = None
                    st.session_state["confirm_delete"] = False
                    st.rerun()
                    
            with b_act4:
                if st.button("📥 SEND FOR HUMAN REVIEW", use_container_width=True, type="primary" if (not can_publish and classification == "REQUIRES_HUMAN_REVIEW") else "secondary"):
                    if check_backend_online():
                        try:
                            requests.post(f"{BACKEND_URL}/advertisements/{ad_id}/human-review", json={"user_id": 1, "reason": "Brand owner requested review"}, timeout=5)
                        except Exception:
                            pass
                            
                    if "pending_human_reviews" not in st.session_state:
                        st.session_state["pending_human_reviews"] = []
                        
                    review_case = {
                        "review_id": int(time.time()),
                        "ad_id": ad_id,
                        "title": rep.get("title", st.session_state.get("last_ad_title", "")),
                        "caption": rep.get("caption", st.session_state.get("last_ad_caption", "")),
                        "file_path": rep.get("file_path", st.session_state.get("last_temp_path", "")),
                        "media_type": "video" if str(st.session_state.get("last_temp_path", "")).lower().endswith(('.mp4', '.avi', '.mov')) else "image",
                        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ai_classification": classification,
                        "risk_score": risk_score,
                        "confidence": confidence,
                        "explanation": rep.get("explanation", ""),
                        "ocr_text": rep.get("extracted_ocr", ""),
                        "audio_transcript": rep.get("audio_transcript", "")
                    }
                    if not any(c["ad_id"] == ad_id for c in st.session_state["pending_human_reviews"]):
                        st.session_state["pending_human_reviews"].append(review_case)
                        
                    st.session_state["owner_action_msg"] = "⏳ Advertisement is Pending for Human Review in the Admin Dashboard."
                    st.session_state["moderation_report"] = None
                    st.session_state["current_ad_id"] = None
                    st.rerun()
                    
            # Confirmation modal for delete action
            if st.session_state.get("confirm_delete", False):
                st.warning("🚨 Are you sure you want to delete this advertisement?")
                col_del1, col_del2 = st.columns(2)
                with col_del1:
                    if st.button("❌ Cancel Delete", use_container_width=True):
                        st.session_state["confirm_delete"] = False
                        st.rerun()
                with col_del2:
                    if st.button("🗑️ Confirm Delete", type="primary", use_container_width=True):
                        if check_backend_online():
                            try:
                                requests.delete(f"{BACKEND_URL}/advertisements/{ad_id}", timeout=5)
                            except Exception:
                                pass
                        st.session_state["moderation_report"] = None
                        st.session_state["current_ad_id"] = None
                        st.session_state["confirm_delete"] = False
                        st.success("Advertisement deleted successfully.")
                        st.rerun()
        else:
            st.info("Upload a creative media file and click **ANALYZE ADVERTISEMENT** to view the safety classification.")
