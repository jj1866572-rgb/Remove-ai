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

# অ্যান্ড্রয়েড অ্যাপের সংযোগ নিশ্চিত করতে CORS অন করা
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
            # গিটহাব থেকে ৪.৭ মেগাবাইট মডেল নামানোর নিরাপদ রিকোয়েস্ট
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

# সার্ভার সচল আছে কিনা চেক করার রুট
@app.get("/")
@app.get("/api")
def health():
    return {"status": "online", "message": "Background Remover API is ready for Android App!"}

# অ্যাপ থেকে আসা সমস্ত POST রিকোয়েস্ট হ্যান্ডেল করা
@app.post("/")
@app.post("/api")
@app.post("/remove-bg")
@app.post("/api/remove-bg")
async def remove_bg(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    # আপনার অ্যাপের 'image' অথবা সাধারণ 'file' ফিল্ড উভয়ই সাপোর্ট করবে
    target = image if image is not None else file
    
    if target is None:
        raise HTTPException(status_code=400, detail="কোনো ছবি পাওয়া যায়নি!")

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

        # ৪.৭ এমবি u2netp মডেল রান করা
        ort_session = get_session()
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: img_data})
        pred = outputs[0][0, 0]

        # ব্যাকগ্রাউন্ড মাস্কিং
        d_min, d_max = pred.min(), pred.max()
        mask_arr = ((pred - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        mask = Image.fromarray(mask_arr, mode="L").resize((orig_w, orig_h), Image.BILINEAR)

        # ট্রান্সপারেন্ট PNG ইমেজ রিটার্ন করা
        result = img.copy()
        result.putalpha(mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
