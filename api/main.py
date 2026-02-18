"""
ShelfSense AI - FastAPI Backend
Retail Shelf Monitoring with Roboflow
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os
from typing import List, Optional, Dict

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
ROBOFLOW_API_KEY = "FVqfVxwUgjQmJHhmr2sR"
MODEL_ENDPOINT = "retail-store-detection-cv-p6zlc/4"
ROBOFLOW_URL = f"https://detect.roboflow.com/{MODEL_ENDPOINT}"

# ────────────────────────────────────────────────
# FastAPI App Setup
# ────────────────────────────────────────────────
app = FastAPI(
    title="ShelfSense AI",
    description="Retail Shelf Monitoring Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────
# Health check
# ────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_ENDPOINT}

# ────────────────────────────────────────────────
# Analyze image endpoint
# ────────────────────────────────────────────────
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        params = {
            "api_key": ROBOFLOW_API_KEY,
            "confidence": 0.4,
            "overlap": 0.3
        }

        resp = requests.post(
            ROBOFLOW_URL,
            params=params,
            data=image_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20
        )

        resp.raise_for_status()
        data = resp.json()

        predictions = data.get("predictions", [])

        details = []
        product_count = 0
        missing_count = 0

        for p in predictions:
            cls = p.get("class", "unknown").lower()
            is_missing = any(word in cls for word in ["missing", "empty", "gap", "hole", "vacant"])

            if is_missing:
                missing_count += 1
                category = "missing"
            else:
                product_count += 1
                category = "product"

            details.append({
                "class": category,
                "x": p.get("x", 0),
                "y": p.get("y", 0),
                "width": p.get("width", 0),
                "height": p.get("height", 0),
                "confidence": p.get("confidence", 0)
            })

        result = {
            "status": "success",
            "summary": {
                "total_products_detected": product_count,
                "total_missing_detected": missing_count
            },
            "details": details,
            "business_mapping": {
                "restock_required": missing_count > 0,
                "severity": "high" if missing_count > 5 else "medium" if missing_count > 2 else "low" if missing_count > 0 else "none"
            }
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))