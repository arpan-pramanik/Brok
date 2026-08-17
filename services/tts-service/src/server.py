from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import edge_tts

app = FastAPI(title="TTS Service (Edge TTS - Neerja)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE = "en-IN-PrabhatNeural"

@app.get("/health")
async def health():
    return {"status": "ok", "voice": VOICE}

@app.get("/synthesize")
async def synthesize(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
        
    try:
        communicate = edge_tts.Communicate(text.strip(), VOICE)
        
        async def iter_audio():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                    
        return StreamingResponse(iter_audio(), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
