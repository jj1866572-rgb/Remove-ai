import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

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

        # ১. Sigmoid অ্যাক্টিভেশন (আসল প্রোবাবিলিটি বের করা)
        pred = 1.0 / (1.0 + np.exp(-pred))

        # ২. থ্রেশহোল্ড কন্ট্রাস্ট (বর্ডারের সাদা আভা ছেঁটে ফেলা)
        # ২০% এর নিচে থাকা সব আবছা সাদা পিক্সেলকে ১০০% ট্রান্সপারেন্ট করে দেওয়া
        low_cut = 0.25
        high_cut = 0.85
        pred = np.clip((pred - low_cut) / (high_cut - low_cut), 0, 1)

        mask_arr = (pred * 255).astype(np.uint8)
        mask = Image.fromarray(mask_arr, mode="L").resize((orig_w, orig_h), Image.BILINEAR)

        # ৩. ডিফ্রিঞ্জ ফিল্টার (সীমানা থেকে ১ পিক্সেল সাদা বর্ডার স্বয়ংক্রিয়ভাবে মুছে দেওয়া)
        mask = mask.filter(ImageFilter.MinFilter(3))

        # ট্রান্সপারেন্ট ব্যাকগ্রাউন্ড সেট
        result = img.copy()
        result.putalpha(mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
