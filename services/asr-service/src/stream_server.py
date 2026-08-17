import sys
import os
import io
import wave
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Ensure we can import shared schemas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from shared.schemas.transcript import FinalTranscript, PartialTranscript

from vad import VAD
from streaming_asr import StreamingASR

load_dotenv(os.path.join(os.path.dirname(__file__), '../../../../.env'))

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
    speech_detected = False
    silence_chunks = 0
    
    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "stop_recording":
                        if len(audio_buffer) > 0:
                            combined_audio = np.concatenate(audio_buffer)
                            final = asr.transcribe_segment(combined_audio, is_final_chunk=True)
                            await websocket.send_text(final.model_dump_json())
                        audio_buffer = [] # Reset for next
                    elif data.get("type") == "start_recording":
                        audio_buffer = [] # Clear any old audio
                except:
                    pass
            elif "bytes" in message:
                data = message["bytes"]
                # Float32Array from browser (chunk size usually 4096 samples = 256ms)
                chunk = np.frombuffer(data, dtype=np.float32)
                audio_buffer.append(chunk)
                
                combined_audio = np.concatenate(audio_buffer)
                
                # Check last ~1 second for speech (up to 16000 samples)
                check_length = min(len(combined_audio), 16000)
                recent_audio = combined_audio[-check_length:]
                
                # VAD requires at least 512 samples
                if len(recent_audio) >= 512:
                    timestamps = vad.detect(recent_audio)
                    if len(timestamps) > 0:
                        speech_detected = True
                        silence_chunks = 0
                    else:
                        silence_chunks += 1
                
                # 1 chunk = 256ms. 6 chunks = 1.53 seconds of silence
                if speech_detected and silence_chunks >= 6:
                    final = asr.transcribe_segment(combined_audio, is_final_chunk=True)
                    await websocket.send_text(final.model_dump_json())
                    # Reset state for next utterance
                    audio_buffer = []
                    speech_detected = False
                    silence_chunks = 0
                    # Send a special event to frontend if we want to tell it we auto-stopped
                    await websocket.send_text(json.dumps({"type": "vad_stop"}))
                    
                # Send partial every ~1 second (4 chunks)
                elif len(audio_buffer) % 4 == 0:
                    partial = asr.transcribe_segment(combined_audio, is_final_chunk=False)
                    await websocket.send_text(partial.model_dump_json())
                
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect message has been received" not in str(e):
            raise
