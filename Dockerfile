# MathLedger Demo
# Governance substrate demo - version read from releases/releases.json at runtime
#
# Build:   docker build -t mathledger-demo .
# Run:     docker run -p 8000:8000 mathledger-demo
# Mounted: docker run -p 8000:8000 -e BASE_PATH=/demo mathledger-demo
#
# IMPORTANT: Version is determined by releases/releases.json "current_version" field.
# The demo reads this at startup, so rebuilding the image with updated releases.json
# will automatically update the reported version.

FROM python:3.11-slim

# Version is dynamic from releases.json - no hardcoded LABELs needed
LABEL description="MathLedger Governance Demo - UVIL v0 + Trust Classes"
LABEL maintainer="helpfuldolphin"

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
COPY releases/ ./releases/

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
