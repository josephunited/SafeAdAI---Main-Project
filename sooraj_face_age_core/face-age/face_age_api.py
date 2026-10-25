from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import io
import sys
import os

# Add the root directory to sys.path so we can import ai modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.face_age.face_age_pipeline import FaceAgePipeline
pipeline = FaceAgePipeline()

# Fallback for behavioral mockup since it wasn't broken
try:
    from website.classifier import estimate_age_from_behavior
except ImportError:
    pass

app = FastAPI(title="Face Age Estimation API")

@app.post("/api/verify_face")
async def verify_face(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Analyze the image directly with the pristine pipeline
    res = pipeline.analyze(image)
    
    if res.get("faces_detected", 0) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the frame. Please ensure a face is visible.")
        
    face = res["faces"][0]
    age_range = face["age_estimation"].get("age_range", "Unknown")
    norm_group = face.get("normalized_age_group", "UNKNOWN")
    conf = face["age_estimation"].get("confidence", 0.0)
    
    # Standardize category text 
    if norm_group == "LESS THAN 14" or age_range in ["0-2", "3-9"]:
        category = "Less than 14"
    elif norm_group == "14 TO 17" or age_range == "10-19":
        category = "14 to 17"
    else:
        category = "18 and above"

    return {"category": category, "confidence": float(conf)}

@app.post("/api/verify_behavior")
async def verify_behavior(queries: str = Form(...), gk_watch: int = Form(...), ad_watch: int = Form(...)):
    import json
    q_list = [q.strip() for q in queries.split(',')]
    gk_w = [{"duration_watched": gk_watch, "total_duration": 60}]
    ad_w = [{"duration_watched": ad_watch, "total_duration": 100}]
    category, conf = estimate_age_from_behavior(q_list, gk_w, ad_w)
    return {"category": category, "confidence": float(conf)}


@app.post("/api/verify_id_and_face")
async def verify_id_and_face(
    id_file: UploadFile = File(...),
    live_file: UploadFile = File(...),
    manual_dob: str = Form(None)
):
    try:
        from website.classifier import verify_id_card_and_live_face, process_pdf_id_document
    except ImportError:
        raise HTTPException(status_code=500, detail="Classifier not found")
        
    live_contents = await live_file.read()
    live_image = np.array(Image.open(io.BytesIO(live_contents)).convert("RGB"))
    
    id_contents = await id_file.read()
    import os
    id_ext = os.path.splitext(id_file.filename)[1].lower()
    
    id_image = None
    pdf_text = None
    if id_ext == ".pdf":
        id_image, pdf_text = process_pdf_id_document(id_contents)
    else:
        id_image = np.array(Image.open(io.BytesIO(id_contents)).convert("RGB"))
        
    res = verify_id_card_and_live_face(id_image, live_image, manual_dob, pdf_text)
    
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
        
    return convert_types(res)

if __name__ == "__main__":
    uvicorn.run("face_age_api:app", host="0.0.0.0", port=8001, reload=False)