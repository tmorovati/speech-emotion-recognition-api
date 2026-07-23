import os 
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.inference import predict_emotion

app = FastAPI(title="Speech Emotion Recognition API")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Reject anything that isn't a .wav upload
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file")

    # 2. Save the uploaded bytes to a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 3. Run the model, then always clean up the temp file
    try:
        predictions = predict_emotion(tmp_path)
    finally:
        os.remove(tmp_path)

    # 4. Shape the JSON response
    return {
        "filename": file.filename,
        "top_emotion": predictions[0]["label"],
        "confidence": round(predictions[0]["score"], 4),
        "all_scores": predictions,
    }
