FROM python:3.12-slim

WORKDIR /app

COPY services/tts-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY shared /app/shared

# Copy service code
COPY services/tts-service/src /app/src

ENV PYTHONPATH=/app
EXPOSE 8005

CMD ["python", "src/server.py"]
