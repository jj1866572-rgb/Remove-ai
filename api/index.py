import io
import os
import urllib.request
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image

app = FastAPI(title="24/7 BG Remover")

# অ্যান্ড্রয়েড অ্যাপ ও ব্রাউজার থেকে সহজে রিকোয়েস্ট আসার পারমিশন (CORS)
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
            # গিটহাবের ব্লক এড়াতে ব্রাউজার হেডার যুক্ত করা হয়েছে
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Background Remover</title>
        <style>
            body { font-family: sans-serif; background: #0f172a; color: white; text-align: center; padding: 20px; }
            .card { background: #1e293b; max-width: 400px; margin: 20px auto; padding: 25px; border-radius: 16px; }
            h2 { margin-bottom: 20px; font-size: 22px; color: #38bdf8; }
            input[type="file"] { margin: 15px 0; width: 100%; color: #94a3b8; }
            button { background: #2563eb; color: white; border: none; padding: 14px; border-radius: 8px; font-weight: bold; width: 100%; font-size: 16px; }
            #status { margin-top: 15px; font-size: 14px; color: #38bdf8; }
            img { max-width: 100%; margin-top: 15px; border-radius: 8px; background: repeating-conic-gradient(#808080 0% 25%, transparent 0% 50%) 50% / 20px 20px; }
            #dlBtn { display: none; margin-top: 15px; background: #10b981; padding: 12px; border-radius: 8px; color: white; font-weight: bold; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>✨ AI ব্যাকগ্রাউন্ড রিমুভার</h2>
            <input type="file" id="fileInput" accept="image/*">
            <button onclick="removeBg()">ব্যাকগ্রাউন্ড রিমুভ করুন</button>
            <div id="status"></div>
            <img id="preview" style="display:none;">
            <br>
            <a id="dlBtn" download="removed_bg.png">ছবি ডাউনলোড করুন</a>
        </div>

        <script>
        async function removeBg() {
            const input = document.getElementById('fileInput');
            if (!input.files[0]) return alert("একটি ছবি সিলেক্ট করুন!");
            
            const status = document.getElementById('status');
            const preview = document.getElementById('preview');
            const dlBtn = document.getElementById('dlBtn');
            
            status.innerText = "কাজ চলছে... অনুগ্রহ করে ৩-৪ সেকেন্ড অপেক্ষা করুন...";
            preview.style.display = "none";
            dlBtn.style.display = "none";
            
            const formData = new FormData();
            formData.append("file", input.files[0]);
            
            try {
                let res = await fetch("/api/remove-bg", { method: "POST", body: formData });
                if (!res.ok) res = await fetch("/remove-bg", { method: "POST", body: formData });
                if (!res.ok) {
                    const errorDetail = await res.text();
                    throw new Error(errorDetail);
                }
                
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                preview.src = url;
                preview.style.display = "block";
                dlBtn.href = url;
                dlBtn.style.display = "inline-block";
                status.innerText = "সফলভাবে ব্যাকগ্রাউন্ড রিমুভ হয়েছে!";
            } catch (err) {
                status.innerText = "সমস্যা: " + err.message;
            }
        }
        </script>
    </body>
    </html>
    """

@app.post("/remove-bg")
@app.post("/api/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = img.size

        img_resized = img.resize((320, 320), Image.BILINEAR)
        img_data = np.array(img_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        ort_session = get_session()
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: img_data})
        pred = outputs[0][0, 0]

        d_min, d_max = pred.min(), pred.max()
        mask_arr = ((pred - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        mask = Image.fromarray(mask_arr, mode="L").resize((orig_w, orig_h), Image.BILINEAR)

        result = img.copy()
        result.putalpha(mask)

        out_io = io.BytesIO()
        result.save(out_io, format="PNG")
        return Response(content=out_io.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
