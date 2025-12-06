param(
  [string]$PolicyPath = "artifacts\policy\policy.bin"
)

$base = "http://127.0.0.1:8001"

function Snapshot {
  try {
    $j = (Invoke-WebRequest "$base/heartbeat.json" -UseBasicParsing).Content | ConvertFrom-Json
    $pph = [int]([double]$j.proofs_per_sec * 3600)
    Write-Host ("[{0:T}] proofs={1:n0}  pph≈{2:n0}  blocks={3:n0}" -f (Get-Date), $j.proofs.success, $pph, $j.blocks.height)
  } catch { Write-Host "[snapshot error]" -ForegroundColor Yellow }
}

function LoopBaseline([int]$secs) {
  $deadline = (Get-Date).AddSeconds($secs)
  $last = (Get-Date).AddSeconds(-999)
  while ((Get-Date) -lt $deadline) {
    uv run python -m backend.axiom_engine.derive --system pl --smoke-pl --seal
    Start-Sleep -Milliseconds 120
    if (((Get-Date) - $last).TotalSeconds -ge 10) { Snapshot; $last = Get-Date }
  }
}

function LoopGuided([int]$secs, [int]$extra, [int]$topk, [double]$eps) {
  $env:ML_GUIDED_EXTRA = "$extra"
  $deadline = (Get-Date).AddSeconds($secs)
  $last = (Get-Date).AddSeconds(-999)
  while ((Get-Date) -lt $deadline) {
    uv run python -m backend.axiom_engine.derive --system pl --smoke-pl `
      --policy $PolicyPath --topk $topk --epsilon $eps --seal
    Start-Sleep -Milliseconds 120
    if (((Get-Date) - $last).TotalSeconds -ge 10) { Snapshot; $last = Get-Date }
  }
}

Set-Location C:\dev\mathledger
Write-Host "=== Demo start ===" -ForegroundColor Cyan
Snapshot

Write-Host "`n--- Stage 1: Baseline (60s) ---" -ForegroundColor Gray
LoopBaseline 60

Write-Host "`n--- Stage 2: Guided (60s, EXTRA=10, TOPK=24, EPS=0.10) ---" -ForegroundColor Gray
LoopGuided 60 10 24 0.10

Write-Host "`n--- Stage 3: Guided (60s, EXTRA=12, TOPK=32, EPS=0.05) ---" -ForegroundColor Gray
LoopGuided 60 12 32 0.05

Write-Host "=== Demo end ===" -ForegroundColor Cyan
Snapshot
