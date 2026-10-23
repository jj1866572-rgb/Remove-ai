import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image

app = FastAPI(title="24/7 BG Remover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/tmp/u2netp.onnx"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
session = None

def get_session():
    global session
    if session is None:
        if not os.path.exists(MODEL_PATH):
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

@app.get("/")
@app.get("/api")
def health():
    return {"status": "online", "message": "BG Remover is Ready!"}

@app.post("/")
@app.post("/api")
@app.post("/remove-bg")
@app.post("/api/remove-bg")
async def remove_bg(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    target = image if image is not None else file
    if target is None:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        contents = await target.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = img.size

        # ৩২০x৩২০ প্রি-প্রসেসিং
        img_resized = img.resize((320, 320), Image.BILINEAR)
        img_data = np.array(img_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        # মডেল রান
        ort_session = get_session()
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: img_data})
        pred = outputs[0][0, 0]

        # ১. প্রবাবিলিটি স্কেলিং
        d_min, d_max = pred.min(), pred.max()
        norm = (pred - d_min) / (d_max - d_min + 1e-8)

        # ২. হাই-রেজুলিউশন গ্র্যাডিয়েন্ট তৈরি (খাবরা ভাব দূর করতে)
        mask_320 = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        mask_hires = mask_320.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        alpha_raw = np.array(mask_hires).astype(np.float32) / 255.0

        # ৩. স্মুথস্টেপ কার্ভ (কোনো কর্কশ খাঁজ থাকবে না, সিল্কি স্মুথ এজ)
        alpha = np.clip((alpha_raw - 0.25) / (0.80 - 0.25), 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        # ৪. কালার ডিকন্টামিনেশন (কিনারের মিশে থাকা সাদা রঙ শুষে নেওয়া)
        img_np = np.array(img).astype(np.float32)
        bg_mask = alpha < 0.05
        if np.any(bg_mask):
            bg_color = np.mean(img_np[bg_mask], axis=0)
        else:
            bg_color = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        a = np.expand_dims(alpha, axis=2)
        decontaminated = (img_np - (1.0 - a) * bg_color) / np.maximum(a, 0.25)
        clean_rgb = np.where((a > 0.02) & (a < 0.98), np.clip(decontaminated, 0, 255), img_np)

        # ৫. ট্রান্সপারেন্ট পিএনজি তৈরি
        result = Image.fromarray(clean_rgb.astype(np.uint8), mode="RGB")
        final_mask = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
        result.putalpha(final_mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
