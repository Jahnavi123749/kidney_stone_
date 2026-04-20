import os
import time
import base64
import uuid
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import cv2
import numpy as np
from io import BytesIO

# Import local modules
from detector import KidneyStoneDetector, infer_kidney_side
from vlm_analyzer import VLMAnalyzer
from size_estimator import estimate_size, estimate_volume, estimate_weight, classify_clinical_size
from dicom_reader import extract_dicom_metadata
from report_generator import generate_pdf_report

app = FastAPI(title="KidneyVision API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
detector = KidneyStoneDetector()
vlm_analyzer = VLMAnalyzer()

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# In-memory store
analysis_history = {}  # image_id -> full result dict


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

# Clinical urgency derived purely from stone size — used as a floor value.
# The VLM can only raise urgency, never lower it below this threshold.
SIZE_URGENCY = {
    "small":      "low",       # < 4 mm  — often passes naturally
    "medium":     "moderate",  # 4–6 mm  — may need medication
    "large":      "high",      # 6–10 mm — likely needs intervention
    "very_large": "critical",  # > 10 mm — surgery / PCNL typically required
}

URGENCY_RANK = {"low": 0, "moderate": 1, "high": 2, "critical": 3}


def _classify_pixel(mean_pixel_dim):
    """Rough pixel-based clinical category when no calibration is available."""
    if mean_pixel_dim < 20:
        return "small"
    elif mean_pixel_dim < 35:
        return "medium"
    elif mean_pixel_dim < 60:
        return "large"
    return "very_large"


def _build_detection(i, det, crops_b64, pixels_per_mm, stone_type):
    """Build a single detection dict without VLM."""
    x1, y1, x2, y2 = det["bbox"]
    pixel_w = int(x2 - x1)
    pixel_h = int(y2 - y1)
    mean_pixel_dim = (pixel_w + pixel_h) / 2

    size_info = estimate_size(det["bbox"], pixels_per_mm)
    volume = None
    weight_info = None
    clinical_cat = "unknown"

    if size_info:
        volume = estimate_volume(size_info["width_mm"], size_info["height_mm"])
        weight_info = estimate_weight(volume, stone_type)
        clinical_detail = classify_clinical_size(size_info["mean_diameter_mm"])
        clinical_cat = clinical_detail["category"] if clinical_detail else "unknown"
    else:
        clinical_cat = _classify_pixel(mean_pixel_dim)

    # Urgency starts at size-based value — VLM can only raise it, never lower it
    initial_urgency = SIZE_URGENCY.get(clinical_cat, "low")

    return {
        "stone_id": i + 1,
        "bbox": det["bbox"],
        "confidence": det["confidence"],
        "pixel_width": pixel_w,
        "pixel_height": pixel_h,
        "size_mm": size_info,
        "volume_cm3": volume,
        "weight_mg": round(weight_info["weight_g"] * 1000, 2) if weight_info else None,
        "weight_g": weight_info["weight_g"] if weight_info else None,
        "clinical_category": clinical_cat,
        "urgency": initial_urgency,      # size-based urgency floor
        "vlm_analysis": None,            # filled in later via SSE
        "vlm_status": "pending",         # pending | done | error | skipped
        "crop_b64": crops_b64[i] if i < len(crops_b64) else None,
    }


# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    vlm_analyzer.check_availability()  # re-check on each health call
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "vlm_available": vlm_analyzer.available,
        "timestamp": time.time()
    }


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    pixels_per_mm: Optional[float] = Form(None),
    stone_type: str = Form("calcium_oxalate"),
    use_vlm: bool = Form(True)
):
    """
    Phase 1: Fast detection — returns YOLO results immediately.
    VLM analysis is done separately via /api/vlm-stream/{image_id}.
    """
    start_time = time.time()
    image_id = str(uuid.uuid4())

    contents = await image.read()
    temp_path = f"temp_{image_id}_{image.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        dicom_data = None
        if image.filename.lower().endswith('.dcm'):
            dicom_data = extract_dicom_metadata(temp_path)
            if dicom_data:
                img_data = base64.b64decode(dicom_data["image_png_b64"])
                nparr = np.frombuffer(img_data, np.uint8)
                cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if pixels_per_mm is None and dicom_data.get("pixel_spacing_mm"):
                    pixels_per_mm = 1.0 / dicom_data["pixel_spacing_mm"]
            else:
                cv2_img = cv2.imread(temp_path)
        else:
            cv2_img = cv2.imread(temp_path)

        if cv2_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Auto-calibrate pixels_per_mm if not provided
        # Standard abdominal CT: ~350mm field of view on the height axis
        if pixels_per_mm is None or pixels_per_mm <= 0:
            img_height_px = cv2_img.shape[0]
            pixels_per_mm = img_height_px / 350.0  # 350mm typical abdominal FOV

        # YOLO detection
        raw_detections = detector.detect(cv2_img, conf=conf_threshold)

        # Annotated image + crops
        annotated_b64 = detector.draw_annotations(cv2_img, raw_detections)
        crops_b64 = detector.crop_stones(cv2_img, raw_detections)

        # Build detections without VLM
        processed = [
            _build_detection(i, det, crops_b64, pixels_per_mm, stone_type)
            for i, det in enumerate(raw_detections)
        ]

        # Compute sizes for summary
        sizes = [d["size_mm"]["mean_diameter_mm"] for d in processed if d["size_mm"]]
        largest_mm = round(max(sizes), 2) if sizes else 0

        # Determine which kidney is affected by stone position
        image_width = cv2_img.shape[1]
        kidney_side = infer_kidney_side(raw_detections, image_width)

        summary = {
            "total_stones": len(processed),
            "largest_stone_mm": largest_mm,
            "highest_urgency": "low",
            "kidney_side": kidney_side,
            "recommendation": "Consult a specialist for clinical evaluation.",
        }

        result = {
            "image_id": image_id,
            "annotated_image_base64": annotated_b64,
            "detections": processed,
            "dicom_metadata": dicom_data,
            "summary": summary,
            "narrative": "",
            "processing_time_ms": int((time.time() - start_time) * 1000),
            # Store metadata for SSE phase
            "_crops_b64": crops_b64,
            "_use_vlm": use_vlm,
            "_stone_type": stone_type,
            "_pixels_per_mm": pixels_per_mm,
        }

        analysis_history[image_id] = result

        # Return without the internal metadata
        public_result = {k: v for k, v in result.items() if not k.startswith("_")}
        return public_result

    except Exception as e:
        print(f"Analysis Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Analysis failed", "detail": str(e), "suggestion": "Check image format and model status"}
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/vlm-stream/{image_id}")
async def vlm_stream(image_id: str):
    """
    Phase 2: SSE streaming endpoint.
    Runs VLM for each stone and streams results as they complete.
    Events:
      - { type: "stone", stone_id, vlm_analysis }
      - { type: "narrative", text }
      - { type: "done", summary }
      - { type: "error", message }
    """
    if image_id not in analysis_history:
        raise HTTPException(status_code=404, detail="Analysis not found")

    result = analysis_history[image_id]
    crops_b64 = result.get("_crops_b64", [])
    use_vlm = result.get("_use_vlm", True)
    stone_type = result.get("_stone_type", "calcium_oxalate")
    detections = result["detections"]

    async def event_generator():
        highest_urgency_score = 0
        urgency_labels = {0: "low", 1: "moderate", 2: "high", 3: "critical"}

        if not use_vlm or not vlm_analyzer.available:
            yield f"data: {json.dumps({'type':'error','message':'VLM not available or disabled.'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            return

        for i, stone in enumerate(detections):
            crop = crops_b64[i] if i < len(crops_b64) else None
            if not crop:
                yield f"data: {json.dumps({'type':'stone','stone_id':stone['stone_id'],'vlm_analysis':None,'vlm_status':'error'})}\n\n"
                continue

            # Run VLM in a thread so we don't block FastAPI event loop
            vlm_res = await asyncio.get_event_loop().run_in_executor(
                None,
                vlm_analyzer.analyze_stone,
                crop,
                stone["bbox"],
                stone["confidence"],
                [stone["pixel_width"], stone["pixel_height"]]
            )

            # Update stored detection
            analysis_history[image_id]["detections"][i]["vlm_analysis"] = vlm_res
            analysis_history[image_id]["detections"][i]["vlm_status"] = "done" if (vlm_res and "urgency" in vlm_res) else "error"

            # Also compute weight from vlm size if mm not available
            if vlm_res and "estimated_diameter_mm" in vlm_res and stone["size_mm"] is None:
                try:
                    raw = vlm_res["estimated_diameter_mm"]
                    nums = [float(x) for x in raw.replace("mm","").split("-") if x.strip()]
                    if nums:
                        diameter_mm = sum(nums) / len(nums)
                        vol = (4/3) * 3.14159 * ((diameter_mm/2)**3) / 1000
                        density = {"calcium_oxalate":1.95,"calcium_phosphate":2.2,"uric_acid":1.65,"struvite":1.80,"cystine":1.68}
                        d = density.get(stone_type, 1.95)
                        weight_g = vol * d
                        analysis_history[image_id]["detections"][i]["weight_mg"] = round(weight_g * 1000, 2)
                        analysis_history[image_id]["detections"][i]["weight_g"] = round(weight_g, 4)
                except Exception:
                    pass

            # Urgency: take max of size-based floor and VLM value (never downgrade)
            size_floor = URGENCY_RANK.get(stone.get("urgency", "low"), 0)
            vlm_urgency_score = URGENCY_RANK.get((vlm_res or {}).get("urgency", "").lower(), 0) if vlm_res else 0
            final_urgency_score = max(size_floor, vlm_urgency_score)
            final_urgency = urgency_labels[final_urgency_score]
            analysis_history[image_id]["detections"][i]["urgency"] = final_urgency

            if final_urgency_score > highest_urgency_score:
                highest_urgency_score = final_urgency_score

            # Override clinical category from VLM
            if vlm_res and vlm_res.get("size_category"):
                analysis_history[image_id]["detections"][i]["clinical_category"] = vlm_res["size_category"]

            yield f"data: {json.dumps({'type':'stone','stone_id':stone['stone_id'],'vlm_analysis':vlm_res,'urgency':final_urgency,'weight_mg':analysis_history[image_id]['detections'][i]['weight_mg'],'clinical_category':analysis_history[image_id]['detections'][i]['clinical_category']})}\n\n"
            await asyncio.sleep(0)  # yield control

        # Update summary urgency
        analysis_history[image_id]["summary"]["highest_urgency"] = urgency_labels[highest_urgency_score]

        # Stream narrative token-by-token
        full_narrative = ""
        token_gen = vlm_analyzer.generate_report_narrative_stream(
            analysis_history[image_id]["detections"],
            analysis_history[image_id]["summary"]
        )

        loop = asyncio.get_event_loop()

        def _next_token(gen):
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, _next_token, token_gen)
            if token is None:
                break
            full_narrative += token
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
            await asyncio.sleep(0)  # yield control to event loop

        # Store full narrative and emit summary events
        analysis_history[image_id]["narrative"] = full_narrative
        yield f"data: {json.dumps({'type': 'narrative', 'text': full_narrative})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'summary': analysis_history[image_id]['summary']})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/report/{image_id}")
async def get_report(image_id: str):
    if image_id not in analysis_history:
        raise HTTPException(status_code=404, detail="Report not found")

    result = analysis_history[image_id]
    pdf_content = generate_pdf_report(result, image_id)

    report_path = os.path.join(REPORTS_DIR, f"report_{image_id}.pdf")
    with open(report_path, "wb") as f:
        f.write(pdf_content)

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"KidneyStone_Report_{image_id}.pdf"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
