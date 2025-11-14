# Multi-stage build pour réduire la taille finale
FROM python:3.10-slim AS builder

# Installer seulement les outils de build nécessaires
RUN apt-get update && apt-get install -y \
    git \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# CORRECTION: Installer PyTorch CPU séparément, puis les autres packages
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Ensuite installer les autres packages depuis PyPI standard
RUN pip install --no-cache-dir -r requirements.txt

# =======================
# STAGE 2: Runtime (finale)
# =======================
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY timesfm_server.py .

# Create directories
RUN mkdir -p /app/models /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Use Railway's dynamic port
ENV PORT=8080
EXPOSE $PORT

# Run the application
CMD ["python", "timesfm_server.py"]
