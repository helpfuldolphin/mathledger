#!/bin/bash

set -e

echo "=== MathLedger Quick Start ==="
echo ""

echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.11+"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Please install Docker Desktop"
    exit 1
fi

echo "Prerequisites OK"
echo ""

echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created"
fi

source .venv/bin/activate
echo "Virtual environment activated"
echo ""

echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -e . > /dev/null 2>&1
echo "Dependencies installed"
echo ""

echo "Starting infrastructure (PostgreSQL + Redis)..."
docker compose up -d postgres redis
sleep 5
echo "Infrastructure started"
echo ""

echo "Setting up database..."
export DATABASE_URL="postgresql://ml:mlpass@localhost:5432/mathledger"
export REDIS_URL="redis://localhost:6379/0"
export PYTHONPATH="$(pwd)"

python scripts/run-migrations.py
echo "Database migrations complete"
echo ""

echo "Running smoke test..."
python -m backend.axiom_engine.derive --system pl --mode baseline --steps 10 --seal
echo "Smoke test complete"
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "  1. Run tests: ml test unit"
echo "  2. Start API: ml env setup && make api"
echo "  3. Start worker: make worker"
echo "  4. View metrics: ml metrics show"
echo ""
echo "For help: ml help"
echo ""
