import sys
import os
import io
import wave
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Ensure we can import shared schemas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from shared.schemas.transcript import FinalTranscript, PartialTranscript

from vad import VAD
from streaming_asr import StreamingASR

app = FastAPI(title="ASR Service")

vad = VAD()
asr = StreamingASR(model_size="small.en")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe", response_model=FinalTranscript)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    contents = await file.read()
    with wave.open(io.BytesIO(contents), 'rb') as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
            raise HTTPException(status_code=400, detail="Must be 16kHz mono 16-bit WAV")
        
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
    result = asr.transcribe_segment(audio_data, is_final_chunk=True)
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = []
    
    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.append(chunk)
            
            combined_audio = np.concatenate(audio_buffer)
            timestamps = vad.detect(combined_audio)
            
            if len(timestamps) > 0:
                partial = asr.transcribe_segment(combined_audio, is_final_chunk=False)
                await websocket.send_text(partial.model_dump_json())
                
    except WebSocketDisconnect:
        if len(audio_buffer) > 0:
            combined_audio = np.concatenate(audio_buffer)
            final = asr.transcribe_segment(combined_audio, is_final_chunk=True)
            await websocket.send_text(final.model_dump_json())
