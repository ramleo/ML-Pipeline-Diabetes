# Stage 1: builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# Stage 2: runtime — lean production image
FROM python:3.11-slim

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code and model artifacts
COPY app.py .
COPY models/ models/

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
