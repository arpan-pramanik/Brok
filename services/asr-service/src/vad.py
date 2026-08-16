import torch

class VAD:
    def __init__(self):
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        self.get_speech_timestamps, _, self.read_audio, *rest = utils
        self.sampling_rate = 16000

    def detect(self, audio_chunk):
        if not isinstance(audio_chunk, torch.Tensor):
            audio_chunk = torch.tensor(audio_chunk, dtype=torch.float32)
        
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()
            
        speech_timestamps = self.get_speech_timestamps(
            audio_chunk,
            self.model,
            sampling_rate=self.sampling_rate
        )
        
        return speech_timestamps
