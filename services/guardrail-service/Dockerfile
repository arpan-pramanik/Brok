FROM python:3.12-slim
WORKDIR /app
COPY services/guardrail-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY services/guardrail-service/src/ /app/src/
EXPOSE 8003
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8003"]
