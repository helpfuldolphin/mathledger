param(
  [switch]$SkipPytest
)

Write-Host "=== MathLedger Sanity ===" -ForegroundColor Cyan
Write-Host "[1/6] Python path"
uv run python -c "import sys; print(sys.executable)"

Write-Host "[2/6] API health"
try {
  $h = (curl.exe -s http://localhost:8000/health) | Out-String
  if (-not $h) { Write-Warning "API not responding. Start it in another tab: uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000" }
  else { Write-Host $h.Trim() }
} catch { Write-Warning "Health check failed: $_" }

Write-Host "[3/6] Active curriculum (PL)"
uv run python -m backend.frontier.print_active --system pl

if (-not $SkipPytest) {
  Write-Host "[4/6] Pytest (quick)"
  uv run pytest -q
} else {
  Write-Host "[4/6] Pytest skipped"
}

Write-Host "[5/6] Mini derive (PL)"
uv run python -m backend.axiom_engine.derive --system pl --steps 5 --depth-max 4 --max-breadth 100 --max-total 300

Write-Host "[6/6] DB proof counts"
docker exec -i infra-postgres-1 psql -U ml -d mathledger -c "SELECT status,prover,COUNT(*) FROM proofs GROUP BY status,prover ORDER BY 1,2;"

Write-Host "=== Done ===" -ForegroundColor Green
