FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies declared in pyproject.toml
RUN python - <<'PY'
import tomllib, pathlib
project = tomllib.load(open("pyproject.toml", "rb"))
deps = project.get("project", {}).get("dependencies", [])
pathlib.Path("requirements-p4.txt").write_text("\n".join(deps))
PY

RUN python -m pip install --no-cache-dir --upgrade pip \
  && python -m pip install --no-cache-dir -r requirements-p4.txt

# Bring in the remaining repository contents
COPY . .

# Default entrypoint runs a short P4 harness sanity run
CMD ["python", "scripts/usla_first_light_p4_harness.py", "--cycles", "50", "--output-dir", "results/p4_compose"]
