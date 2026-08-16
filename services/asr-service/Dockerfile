FROM python:3.12-slim
WORKDIR /app
COPY services/asr-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY services/asr-service/src/ /app/src/
EXPOSE 8001
CMD ["uvicorn", "src.stream_server:app", "--host", "0.0.0.0", "--port", "8001"]
