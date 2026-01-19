# Irys RLM Production Dockerfile
# Optimized for small EC2 instances (t3.micro/small)

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (for caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Production image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash irys
WORKDIR /app

# Copy application code
COPY --chown=irys:irys src/ /app/src/
COPY --chown=irys:irys pyproject.toml /app/

# Install app in editable mode
RUN pip install --no-cache-dir -e .

# Create temp directory
RUN mkdir -p /tmp/irys && chown irys:irys /tmp/irys

# Switch to non-root user
USER irys

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IRYS_HOST=0.0.0.0 \
    IRYS_PORT=8000 \
    IRYS_TEMP_DIR=/tmp/irys \
    IRYS_LOG_LEVEL=INFO \
    IRYS_LOG_FORMAT=json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "irys.service.api:app", "--host", "0.0.0.0", "--port", "8000"]

