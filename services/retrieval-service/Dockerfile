FROM python:3.12-slim
WORKDIR /app
COPY services/retrieval-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY services/indexing-service/src/ /app/indexing_src/
COPY services/retrieval-service/src/ /app/src/
ENV PYTHONPATH=/app
EXPOSE 8002
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8002"]
