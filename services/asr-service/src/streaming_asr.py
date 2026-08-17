import torch
from faster_whisper import WhisperModel
import os
import sys
import io
import wave
import re
import numpy as np
import httpx

# Ensure we can import shared schemas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from shared.schemas.transcript import FinalTranscript, PartialTranscript

class StreamingASR:
    def __init__(self, model_size="small.en"):
        self.sarvam_key = os.getenv("SARVAM_API_KEY")
        if self.sarvam_key and self.sarvam_key != "your_sarvam_key_here":
            print("ASR Engine initialized with Sarvam AI STT API")
        else:
            self.sarvam_key = None

        # Auto-detect hardware for local fallback
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
            print("Fallback ASR Engine initialized on GPU (CUDA) with model:", model_size)
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            print("Fallback ASR Engine initialized on CPU with model:", model_size)

        # Load Whisper model as local engine / fallback
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=False
        )

    def _preprocess_audio(self, audio_data):
        """Audio Pre-processing: DC offset removal & Peak amplitude normalization (-1dB peak)."""
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # Remove DC Offset (bias)
        audio_data = audio_data - np.mean(audio_data)
        
        # Peak Amplitude Normalization (Scale soft audio to 0.9 peak volume)
        max_peak = np.max(np.abs(audio_data))
        if max_peak > 1e-4:
            audio_data = (audio_data / max_peak) * 0.9
            
        return audio_data

    def _postprocess_text(self, text):
        """Text Post-processing: Strips fillers, fixes repeated stutter words, normalizes spaces."""
        if not text:
            return ""
        
        # Remove vocal fillers (um, uh, ah, er, hmm)
        text = re.sub(r'\b(um|uh|er|ah|hmm)\b', '', text, flags=re.IGNORECASE)
        
        # Fix stuttered word repetitions ("what what" -> "what")
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Capitalize initial letter
        if text:
            text = text[0].upper() + text[1:]
            
        return text

    def _transcribe_sarvam(self, audio_data):
        try:
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                pcm_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                wf.writeframes(pcm_data.tobytes())
            buf.seek(0)

            headers = {"api-subscription-key": self.sarvam_key}
            files = {"file": ("audio.wav", buf, "audio/wav")}
            data = {
                "model": "saaras:v3",
                "language_code": "en-IN",
                "mode": "transcribe"
            }

            with httpx.Client(timeout=15.0) as client:
                resp = client.post("https://api.sarvam.ai/speech-to-text", headers=headers, files=files, data=data)
                if resp.status_code == 200:
                    res_json = resp.json()
                    raw_text = res_json.get("transcript", "")
                    return self._postprocess_text(raw_text)
                else:
                    print(f"Sarvam API error HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Sarvam AI STT error: {e}, using local fallback")
        return None

    def transcribe_segment(self, audio_data, is_final_chunk=True):
        # Preprocess audio (peak volume normalization & DC offset removal)
        audio_data = self._preprocess_audio(audio_data)

        # Try Sarvam STT if key is set
        if self.sarvam_key and is_final_chunk:
            text = self._transcribe_sarvam(audio_data)
            if text is not None and text.strip():
                duration = float(len(audio_data)) / 16000.0 if hasattr(audio_data, "__len__") else 0.0
                return FinalTranscript(
                    text=text,
                    confidence=0.99,
                    duration_seconds=duration,
                    language="en"
                )

        # Local Whisper Engine (or for partials)
        beam_size = 5 if is_final_chunk else 1
        best_of = 5 if is_final_chunk else 1
        
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=beam_size,
            best_of=best_of,
            language="en", 
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=250,
                min_silence_duration_ms=300,
                speech_pad_ms=300
            )
        )
        
        raw_text = "".join([segment.text for segment in segments]).strip()
        text = self._postprocess_text(raw_text)
        
        if is_final_chunk:
            duration = float(len(audio_data)) / 16000.0 if hasattr(audio_data, "__len__") else 0.0
            return FinalTranscript(
                text=text,
                confidence=info.language_probability if info else 1.0,
                duration_seconds=duration,
                language=info.language if info else "en"
            )
        else:
            return PartialTranscript(text=text)
