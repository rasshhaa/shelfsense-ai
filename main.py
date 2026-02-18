"""
ShelfSense AI - FastAPI Backend
Retail Shelf Monitoring with Roboflow
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import uuid
import os
from typing import List, Optional, Dict
import uvicorn
import webbrowser
from threading import Timer

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

# Serve images folder
app.mount("/images", StaticFiles(directory="images"), name="images")

# ────────────────────────────────────────────────
# Serve index.html at root (this is the most important part)
# ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    
    if not os.path.exists(html_path):
        return HTMLResponse(
            content="""
            <h1 style="color:#e74c3c; text-align:center; margin-top:120px; font-family:sans-serif;">
                index.html not found
            </h1>
            <p style="text-align:center; color:#7f8c8d;">
                Make sure the file is named exactly <strong>index.html</strong><br>
                and is in the same folder as main.py
            </p>
            """,
            status_code=500
        )
    
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ────────────────────────────────────────────────
# Health check (optional – you can remove if not needed)
# ────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_ENDPOINT}

# ────────────────────────────────────────────────
# Analyze image endpoint (adapted for frontend)
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

        print(f"→ Processing image: {file.filename} ({len(image_bytes):,} bytes)")

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
        print(f"← Roboflow returned {len(predictions)} predictions")

        # Build response that frontend expects
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

        print(f"→ Analysis complete: {product_count} products, {missing_count} missing")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ────────────────────────────────────────────────
# Auto-open browser when server starts
# ────────────────────────────────────────────────
# New version (only change host + port logic)
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8000))          # Render gives you $PORT

    def open_browser():
        import webbrowser
        webbrowser.open("http://127.0.0.1:8000")

    print("Starting ShelfSense AI Backend...")
    print(f"  UI:        http://0.0.0.0:{port}")
    print("  Docs:      http://0.0.0.0:{port}/docs")

    # Comment out or remove browser auto-open on Render
    # Timer(2.0, open_browser).start()

    uvicorn.run(
        app,
        host="0.0.0.0",           # ← very important
        port=port,                # ← very important
        reload=False,             # ← important on Render
        log_level="info"
    )