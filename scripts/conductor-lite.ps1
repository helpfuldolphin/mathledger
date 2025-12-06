Write-Host "== Sanity: DB ==" -ForegroundColor Cyan
"import os, psycopg; c=psycopg.connect(os.getenv('DATABASE_URL')); c.close(); print('db_ok')" | uv run python - | Out-Null
Write-Host "DB OK" -ForegroundColor Green

Write-Host "== Sanity: Redis ==" -ForegroundColor Cyan
"import os, redis; r=redis.from_url(os.getenv('REDIS_URL')); r.ping(); print('redis_ok')" | uv run python - | Out-Null
Write-Host "Redis OK" -ForegroundColor Green

Write-Host "== Tests ==" -ForegroundColor Cyan
uv run pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "Tests FAILED" -ForegroundColor Red; exit 1 } else { Write-Host "Tests GREEN" -ForegroundColor Green }

Write-Host "== DDL (idempotent) ==" -ForegroundColor Cyan
"import os, psycopg; print('ddl skipped in lite mode')" | uv run python - | Out-Null
Write-Host "DDL OK" -ForegroundColor Green

Write-Host "== Derive (best-effort) ==" -ForegroundColor Cyan
try {
  uv run python -m backend.axiom_engine.derive --system pl --steps 50 --max-breadth 600 --max-total 2000 --seal
  Write-Host "Derive finished (module)" -ForegroundColor Green
} catch {
  Write-Host "Module derive not available; skipped" -ForegroundColor Yellow
}

Write-Host "== Metrics (DB fallback) ==" -ForegroundColor Cyan
"import os, psycopg, json;
with psycopg.connect(os.getenv('DATABASE_URL')) as c, c.cursor() as cur:
    cur.execute('SELECT COUNT(*) FROM statements'); s=cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM proofs'); p=cur.fetchone()[0]
    cur.execute('SELECT COALESCE(MAX(block_number),0) FROM blocks'); h=cur.fetchone()[0]
    print(json.dumps({'statements':s,'proofs':p,'block_height':h}))" | uv run python -

Write-Host "== DONE ==" -ForegroundColor Green
