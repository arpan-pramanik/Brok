import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from shared.schemas.generation import GenerationRequest, GenerationResponse
from .local_llm import LocalLLM
from .fallback_llm import BedrockFallback

logger = logging.getLogger(__name__)

class CircuitBreakerLLM:
    def __init__(self):
        self.local_llm = LocalLLM()
        self.fallback_llm = BedrockFallback()

    @retry(
        stop=stop_after_attempt(3), # 1 initial + 2 retries
        wait=wait_exponential(multiplier=1, min=1, max=2),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _try_local(self, request: GenerationRequest) -> GenerationResponse:
        logger.info("Attempting local LLM generation")
        return self.local_llm.generate(request)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            return self._try_local(request)
        except Exception as local_err:
            logger.warning(f"Local LLM failed after retries: {local_err}. Falling back to Bedrock.")
            try:
                return self.fallback_llm.generate(request)
            except Exception as bedrock_err:
                logger.error(f"Bedrock fallback also failed: {bedrock_err}.")
                return GenerationResponse(
                    answer="Service temporarily unavailable",
                    sources=[],
                    model_used="error",
                    generation_time_ms=0.0,
                    fallback_used=True
                )
