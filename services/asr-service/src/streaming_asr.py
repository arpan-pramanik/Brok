import sys
import os
import numpy as np
from faster_whisper import WhisperModel

# Ensure we can import shared schemas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from shared.schemas.transcript import PartialTranscript, FinalTranscript

class StreamingASR:
    def __init__(self, model_size="small.en"):
        # RTX 5070 with 8GB VRAM is available, so using cuda and float16
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.accumulated_text = ""
        self.segment_id = 0

    def transcribe_segment(self, audio_data: np.ndarray, is_final_chunk: bool = False):
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=5, 
            language="en", 
            condition_on_previous_text=False
        )
        
        text = "".join(segment.text for segment in segments)
            
        if is_final_chunk:
            self.accumulated_text += text
            final = FinalTranscript(
                text=self.accumulated_text.strip(),
                confidence=0.9,
                duration_seconds=len(audio_data) / 16000.0,
                language="en"
            )
            self.accumulated_text = ""
            return final
        else:
            partial_text = self.accumulated_text + text
            partial = PartialTranscript(
                text=partial_text.strip(),
                is_final=False,
                confidence=0.9,
                segment_id=self.segment_id
            )
            self.segment_id += 1
            return partial
