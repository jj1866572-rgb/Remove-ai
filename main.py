from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from rembg import remove, new_session
import io

app = FastAPI(title="Background Remover API")

# ৪.৭ মেগাবাইটের হালকা মডেল 'u2netp' লোড করা হচ্ছে
session = new_session("u2netp")

@app.get("/")
def home():
    return {"status": "success", "message": "Background Remover API is running!"}

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    # ইমেজ ফাইল চেক করা
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="দয়া করে একটি সঠিক ইমেজ ফাইল আপলোড করুন।")
    
    # ইমেজ রিড করা
    input_bytes = await file.read()
    
    # ব্যাকগ্রাউন্ড রিমুভ করা
    output_bytes = remove(input_bytes, session=session)
    
    # ট্রান্সপারেন্ট PNG ইমেজ রিটার্ন করা
    return Response(content=output_bytes, media_type="image/png")
