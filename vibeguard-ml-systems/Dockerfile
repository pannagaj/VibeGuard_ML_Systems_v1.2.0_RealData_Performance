FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIBEGUARD_CHECKPOINT=/app/artifacts/model.pt
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY artifacts ./artifacts
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "vibeguard.service:app", "--host", "0.0.0.0", "--port", "8000"]
