import time
import json
import boto3
from shared.schemas.generation import GenerationRequest, GenerationResponse
from prompt_templates import format_prompt

class BedrockFallback:
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        prompt = format_prompt(request.query, request.context_chunks, request.source_docs)
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })

        start_time = time.time()
        
        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json"
        )
        
        end_time = time.time()
        
        response_body = json.loads(response.get("body").read())
        answer = response_body.get("content")[0].get("text")

        return GenerationResponse(
            answer=answer,
            sources=request.source_docs,
            model_used=self.model_id,
            generation_time_ms=(end_time - start_time) * 1000,
            fallback_used=True
        )
