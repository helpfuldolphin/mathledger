param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
function Info($msg){ Write-Host $msg -ForegroundColor Cyan }

function Assert-LastExitCode {
  param(
    [string]$StepDescription
  )
  if ($LASTEXITCODE -ne 0) {
    throw "$StepDescription failed with exit code $LASTEXITCODE"
  }
}

Info "[1/7] Healthcheck (API)"
$api = (curl.exe -s http://localhost:8000/health) | Out-String
if (-not $api) { throw "API not responding. Start it: uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000" }
Write-Host $api.Trim()

Info "[2/7] Print active curriculum (PL)"
uv run python -m backend.frontier.print_active --system pl
Assert-LastExitCode "Print active curriculum"

Info "[3/7] Derivation cycle (PL)"
uv run python -m backend.axiom_engine.derive --system pl --steps 300 --depth-max 4 --max-breadth 500 --max-total 2000
Assert-LastExitCode "Derivation cycle"

Info "[4/7] Append progress (latest run)"
if ($DryRun) {
  Write-Host "(dry) progress append"
} else {
  uv run python -m backend.tools.progress --append-latest
  Assert-LastExitCode "Progress append"
}

Info "[5/7] Export snapshot"
if ($DryRun) {
  Write-Host "(dry) export snapshot"
} else {
  .\export-snapshot.ps1
  Assert-LastExitCode "Snapshot export"
}

Info "[6/7] Metrics canonicalization (Cursor K)"
if ($DryRun) {
  Write-Host "(dry) metrics cartographer collect/report"
} else {
  uv run python tools/metrics_cartographer_cli.py collect
  Assert-LastExitCode "Metrics cartographer collect"

  uv run python backend/metrics_reporter.py
  Assert-LastExitCode "Metrics ASCII reporter"

  uv run python backend/metrics_md_report.py
  Assert-LastExitCode "Metrics markdown reporter"
}

Info "[7/7] DB maintenance"
if ($DryRun) {
  Write-Host "(dry) db maintenance"
} else {
  .\db-maintenance.ps1
  Assert-LastExitCode "DB maintenance"
}

Info "Nightly run complete."
