$ErrorActionPreference = "Stop"

function Write-Info($m){Write-Host "[INFO] $m" -ForegroundColor Cyan}
function Write-OK($m){Write-Host "[OK]   $m" -ForegroundColor Green}
function Write-Warn($m){Write-Host "[WARN] $m" -ForegroundColor Yellow}
function Write-Err($m){Write-Host "[ERR]  $m" -ForegroundColor Red}

# --- 0) Repo root & env validation (no defaults) ---
$Repo = (Resolve-Path .).Path
if (-not (Test-Path "$Repo\pyproject.toml")) { Write-Err "Run from repo root (pyproject.toml missing)"; exit 1 }

# Require explicit environment configuration
if (-not $env:DATABASE_URL -or $env:DATABASE_URL -eq "") {
  Write-Err "DATABASE_URL environment variable is not set."
  Write-Warn "Set it explicitly before running conductor."
  exit 1
}
if (-not $env:REDIS_URL -or $env:REDIS_URL -eq "") {
  Write-Err "REDIS_URL environment variable is not set."
  Write-Warn "Set it explicitly before running conductor."
  exit 1
}
$env:PYTHONPATH = $Repo

# --- 1) Sanity: DB + Redis ---
Write-Info "Checking DB connectivity..."
@'
import os, psycopg
c = psycopg.connect(os.getenv("DATABASE_URL"))
c.close()
print("db_ok")
