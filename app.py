from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from gtts import gTTS
import uuid
import os

app = FastAPI()

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TextInput(BaseModel):
    text: str

@app.post("/generate")
async def generate_summary(text_input: TextInput):
    summary = summarizer(text_input.text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    # Generate audio
    filename = f"{uuid.uuid4()}.mp3"
    tts = gTTS(text=summary, lang="en")
    tts.save(filename)

    return {"summary": summary, "audio_url": f"/audio/{filename}"}

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return {"audio": f"http://your-backend-url.com/{filename}"}
