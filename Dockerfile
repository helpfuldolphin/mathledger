# MathLedger Demo v0.2.0
# Governance substrate demo - version-pinned
#
# Build:   docker build -t mathledger-demo .
# Run:     docker run -p 8000:8000 mathledger-demo
# Mounted: docker run -p 8000:8000 -e BASE_PATH=/demo mathledger-demo

FROM python:3.11-slim

# Version pinning - match demo/app.py
LABEL version="0.2.0"
LABEL tag="v0.2.0-demo-lock"
LABEL commit="27a94c8a58139cb10349f6418336c618f528cbab"
LABEL description="MathLedger Governance Demo - UVIL v0 + Trust Classes"

# Install uv for dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies
RUN uv pip install --system -e . || uv pip install --system fastapi uvicorn

# Copy application code
COPY demo/ ./demo/
COPY backend/ ./backend/
COPY governance/ ./governance/
COPY attestation/ ./attestation/
COPY substrate/ ./substrate/
COPY normalization/ ./normalization/
COPY docs/ ./docs/

# Environment configuration
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# BASE_PATH for reverse proxy mounting (default: root)
ENV BASE_PATH=""

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

# Run the demo
CMD ["python", "-m", "uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
