FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/generation-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared and src
COPY shared /app/shared
COPY services/generation-service/src /app/src

EXPOSE 8004

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8004"]
