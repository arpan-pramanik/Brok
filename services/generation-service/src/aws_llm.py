import time
import boto3
from shared.schemas.generation import GenerationRequest, GenerationResponse
from prompt_templates import format_prompt

class AwsNovaLLM:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.model_id = "us.amazon.nova-micro-v1:0"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        prompt = format_prompt(request.query, request.context_chunks, request.source_docs)
        
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        start_time = time.time()
        
        try:
            # Using the Bedrock Converse API for Nova models
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    "maxTokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
            answer = response['output']['message']['content'][0]['text']
            model_used = self.model_id
        except Exception as e:
            answer = f"Service temporarily unavailable: {str(e)}"
            model_used = "error"

        end_time = time.time()

        return GenerationResponse(
            answer=answer,
            sources=request.source_docs,
            model_used=model_used,
            generation_time_ms=(end_time - start_time) * 1000,
            fallback_used=False
        )
