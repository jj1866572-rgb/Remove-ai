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

app = FastAPI(title="24/7 Studio Quality BG Remover API")

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
    return {"status": "online", "message": "Studio Quality BG Remover Ready"}

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

        # ৩২০x৩২০ ইনপুট প্রসেসিং
        img_resized = img.resize((320, 320), Image.Resampling.BILINEAR)
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
        norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

        # ২. ৩২-বিট ফ্লোটিং পয়েন্ট (Mode 'F') দিয়ে হাই-রেজুলিউশন বাইকিউবিক রিসাইজিং
        # (এতে কোনো পিক্সেল ভাঙবে না, খাঁজকাটা ভাব পুরোপুরি বন্ধ হবে)
        mask_float = Image.fromarray(norm, mode="F")
        mask_hi_float = mask_float.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
        norm_hi = np.array(mask_hi_float)

        # ৩. প্রিসিশন ইনওয়ার্ড হারমাইট কার্ভ (Precision Hermite Smoothstep)
        # ০.৫৪ এর নিচে সব ব্যাকগ্রাউন্ড কালার ১০০% জিরো (বাইরে কোনো কালার ছড়াবে না)
        # ০.৬৪ এর ওপরে জামা ও শরীর ১০০% সলিড
        # কোনো কৃত্রিম কাট ছাড়াই স্বয়ংক্রিয়ভাবে সিল্কি-স্মুথ এজ তৈরি হবে
        low_t = 0.54
        high_t = 0.64
        alpha = np.clip((norm_hi - low_t) / (high_t - low_t), 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        # ৪. ট্রান্সপারেন্ট পিএনজি তৈরি
        final_mask = Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L")
        result = img.copy()
        result.putalpha(final_mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
