FROM python:3.11-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY src ./src
COPY configs ./configs
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "recsys.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
