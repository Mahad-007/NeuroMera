# Integrating NeuroMera Models Into Your Website

This guide covers how to serve all three emotion models (text, audio, video) as an API and connect them to any frontend.

---

## Architecture

```
┌──────────────┐       HTTP        ┌──────────────────┐
│   Frontend   │  ───────────────> │   FastAPI Server  │
│  (any site)  │  <─────────────── │                   │
│              │     JSON          │  ┌─────────────┐  │
│  - Text box  │                   │  │  BERT       │  │
│  - Audio mic │                   │  │  Wav2Vec2   │  │
│  - Video cam │                   │  │  ViT        │  │
└──────────────┘                   │  └─────────────┘  │
                                   └──────────────────┘
```

Your website sends text/audio/video to the API. The API runs the model and returns the predicted emotion as JSON.

---

## Step 1 - Download Your Trained Models

After training each notebook in Colab, your models are saved on Google Drive:

```
Google Drive/
  NeuroMera_Model/          → text model
  NeuroMera_Audio_Model/    → audio model
  NeuroMera_Video_Model/    → video model
```

Download all three folders and place them in a `models/` directory:

```
neuromera-server/
  app.py
  requirements.txt
  models/
    text/          ← contents of NeuroMera_Model
    audio/         ← contents of NeuroMera_Audio_Model
    video/         ← contents of NeuroMera_Video_Model
```

Each folder should contain `config.json`, `model.safetensors`, and tokenizer/processor files.

---

## Step 2 - Install Dependencies

Create `requirements.txt`:

```
fastapi==0.115.0
uvicorn==0.30.0
python-multipart==0.0.9
transformers==4.44.0
torch==2.4.0
torchaudio==2.4.0
librosa==0.10.2
opencv-python-headless==4.10.0.84
numpy==1.26.4
Pillow==10.4.0
```

Install:

```bash
pip install -r requirements.txt
```

---

## Step 3 - Create the API Server

Create `app.py`:

```python
import os
import tempfile
from collections import Counter

import cv2
import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    ViTForImageClassification,
    ViTImageProcessor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
)

app = FastAPI(title="NeuroMera Emotion API")

# Allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model = None
text_tokenizer = None
audio_model = None
audio_processor = None
video_model = None
video_processor = None
face_cascade = None


# ── Load models once on startup ──────────────────────────
@app.on_event("startup")
def load_models():
    global text_model, text_tokenizer
    global audio_model, audio_processor
    global video_model, video_processor, face_cascade

    print(f"Loading models on {device}...")

    # Text
    text_model = BertForSequenceClassification.from_pretrained("models/text").to(device)
    text_tokenizer = BertTokenizer.from_pretrained("models/text")
    text_model.eval()

    # Audio
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("models/audio").to(device)
    audio_processor = Wav2Vec2Processor.from_pretrained("models/audio")
    audio_model.eval()

    # Video
    video_model = ViTForImageClassification.from_pretrained("models/video").to(device)
    video_processor = ViTImageProcessor.from_pretrained("models/video")
    video_model.eval()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("All models loaded.")


# ── 1. Text endpoint ────────────────────────────────────
class TextRequest(BaseModel):
    text: str

@app.post("/predict/text")
async def predict_text(body: TextRequest):
    inputs = text_tokenizer(
        body.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    emotion = text_model.config.id2label[pred]
    confidence = torch.softmax(outputs.logits, dim=1)[0][pred].item()

    return {"emotion": emotion, "confidence": round(confidence, 4)}


# ── 2. Audio endpoint ───────────────────────────────────
@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=16000)
        inputs = audio_processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = audio_model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        emotion = audio_model.config.id2label[pred]
        confidence = torch.softmax(outputs.logits, dim=1)[0][pred].item()

        return {"emotion": emotion, "confidence": round(confidence, 4)}
    finally:
        os.unlink(tmp_path)


# ── 3. Video endpoint ───────────────────────────────────
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return {"error": "Could not read video"}

        n_frames = 10
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        predictions = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(detected) > 0:
                x, y, w, h = max(detected, key=lambda r: r[2] * r[3])
                face = frame[y : y + h, x : x + w]
            else:
                h, w = frame.shape[:2]
                size = min(h, w)
                y0, x0 = (h - size) // 2, (w - size) // 2
                face = frame[y0 : y0 + size, x0 : x0 + size]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(face)
            inputs = video_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = video_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(video_model.config.id2label[pred])

        cap.release()

        if not predictions:
            return {"error": "No frames processed"}

        emotion = Counter(predictions).most_common(1)[0][0]
        total_frames = len(predictions)
        agreement = predictions.count(emotion) / total_frames

        return {
            "emotion": emotion,
            "confidence": round(agreement, 4),
            "frames_analyzed": total_frames,
        }
    finally:
        os.unlink(tmp_path)


# ── Health check ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}
```

---

## Step 4 - Run the Server

```bash
cd neuromera-server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
# Text
curl -X POST http://localhost:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love you so much"}'

# Audio
curl -X POST http://localhost:8000/predict/audio \
  -F "file=@sample.wav"

# Video
curl -X POST http://localhost:8000/predict/video \
  -F "file=@sample.mp4"
```

Response format (all endpoints):

```json
{
  "emotion": "joy",
  "confidence": 0.9712
}
```

---

## Step 5 - Connect Your Frontend

### Text (any framework)

```javascript
async function predictText(text) {
  const res = await fetch("http://localhost:8000/predict/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return await res.json(); // { emotion: "joy", confidence: 0.97 }
}

// Usage
const result = await predictText("I feel amazing today");
console.log(result.emotion); // "joy"
```

### Audio (from file input or microphone)

```javascript
async function predictAudio(audioFile) {
  const form = new FormData();
  form.append("file", audioFile);

  const res = await fetch("http://localhost:8000/predict/audio", {
    method: "POST",
    body: form,
  });
  return await res.json();
}

// From <input type="file" id="audioInput" accept="audio/*">
const input = document.getElementById("audioInput");
input.addEventListener("change", async (e) => {
  const result = await predictAudio(e.target.files[0]);
  console.log(result.emotion);
});
```

### Video (from file input or webcam recording)

```javascript
async function predictVideo(videoFile) {
  const form = new FormData();
  form.append("file", videoFile);

  const res = await fetch("http://localhost:8000/predict/video", {
    method: "POST",
    body: form,
  });
  return await res.json();
}

// From <input type="file" id="videoInput" accept="video/*">
const input = document.getElementById("videoInput");
input.addEventListener("change", async (e) => {
  const result = await predictVideo(e.target.files[0]);
  console.log(result.emotion);
});
```

### Recording from webcam/mic (browser)

```javascript
// Record audio from microphone
async function recordAudio(seconds = 5) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks = [];

  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();

  await new Promise((r) => setTimeout(r, seconds * 1000));
  recorder.stop();
  stream.getTracks().forEach((t) => t.stop());

  await new Promise((r) => (recorder.onstop = r));
  return new Blob(chunks, { type: "audio/webm" });
}

// Record video from webcam
async function recordVideo(seconds = 5) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  const recorder = new MediaRecorder(stream);
  const chunks = [];

  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();

  await new Promise((r) => setTimeout(r, seconds * 1000));
  recorder.stop();
  stream.getTracks().forEach((t) => t.stop());

  await new Promise((r) => (recorder.onstop = r));
  return new Blob(chunks, { type: "video/webm" });
}
```

---

## Step 6 - Deploy

### Option A: Docker (recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t neuromera-api .
docker run -p 8000:8000 neuromera-api
```

### Option B: Cloud platforms

| Platform | GPU | Free tier | Notes |
|----------|-----|-----------|-------|
| Railway | No | $5 credit | Easiest. Add models to repo or download on startup |
| Render | No | 750 hrs/mo | Good for CPU inference |
| Google Cloud Run | No | Free tier | Serverless, cold starts may be slow |
| AWS EC2 (g4dn) | Yes | No | Best for production with GPU |
| Hugging Face Spaces | Yes (paid) | Free CPU | Built for ML models |

**For CPU deployment:** All three models work on CPU. Text and audio inference take <1 second. Video takes 3-10 seconds depending on length.

**For production with heavy traffic:** Use a GPU instance (AWS g4dn.xlarge or similar) and add request queuing.

---

## Step 7 - Update the frontend URL

When deployed, replace `http://localhost:8000` with your production URL:

```javascript
const API_URL = "https://your-app.railway.app"; // or wherever you deployed

const res = await fetch(`${API_URL}/predict/text`, { ... });
```

---

## API Reference

| Endpoint | Method | Input | Response |
|----------|--------|-------|----------|
| `/predict/text` | POST | `{"text": "..."}` (JSON body) | `{"emotion": "joy", "confidence": 0.97}` |
| `/predict/audio` | POST | `.wav` file (multipart form) | `{"emotion": "anger", "confidence": 0.85}` |
| `/predict/video` | POST | `.mp4` file (multipart form) | `{"emotion": "fear", "confidence": 0.80, "frames_analyzed": 10}` |
| `/health` | GET | none | `{"status": "ok", "device": "cpu"}` |

---

## Folder Structure (final)

```
neuromera-server/
  app.py                  ← FastAPI server (copy from Step 3)
  requirements.txt        ← dependencies (copy from Step 2)
  Dockerfile              ← for Docker deployment (optional)
  models/
    text/                 ← download from Google Drive: NeuroMera_Model/
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
      vocab.txt
      special_tokens_map.json
    audio/                ← download from Google Drive: NeuroMera_Audio_Model/
      config.json
      model.safetensors
      preprocessor_config.json
    video/                ← download from Google Drive: NeuroMera_Video_Model/
      config.json
      model.safetensors
      preprocessor_config.json
```
