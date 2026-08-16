import os
import time
import logging
from shared.schemas.generation import GenerationRequest, GenerationResponse
from prompt_templates import format_prompt
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LocalLLM:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        if not os.path.exists(model_path):
            alt_path = str(Path(__file__).parent.parent.parent.parent / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf")
            if os.path.exists(alt_path):
                model_path = alt_path
        try:
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_threads=os.cpu_count() or 16,
                n_ctx=4096,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to load local model from {model_path}: {e}")
            self.model = None

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.model:
            raise RuntimeError("Local model not loaded")

        prompt = format_prompt(request.query, request.context_chunks, request.source_docs)
        
        start_time = time.time()
        
        response = self.model(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["<|im_end|>", "\n\nQuery:"],
            echo=False
        )
        
        end_time = time.time()
        answer = response['choices'][0]['text'].strip()
        
        return GenerationResponse(
            answer=answer,
            sources=request.source_docs,
            model_used="qwen2.5-3b-instruct",
            generation_time_ms=(end_time - start_time) * 1000,
            fallback_used=False
        )
