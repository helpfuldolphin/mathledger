param(
  [ValidateSet("Baseline","Guided")] [string]$Mode,
  [string]$Slice = "PL-2",
  [int]$Seconds = 60,
  [string]$PolicyPath = "artifacts\policy\policy.bin",
  [int]$TopK = 12,
  [double]$Eps = 0.2
)

$base = "http://127.0.0.1:8000"
$hdr  = @{ "X-API-Key" = "devkey" }

function Get-ProofsSuccess {
  try {
    $m = (Invoke-WebRequest "$base/metrics" -Headers $hdr -UseBasicParsing).Content | ConvertFrom-Json
    return [int]$m.proofs.success
  } catch { return 0 }
}

# ensure output dir
$destDir = "artifacts\wpv5"
if (!(Test-Path $destDir)) { New-Item -ItemType Directory -Force $destDir | Out-Null }

$start      = Get-Date
$startCount = Get-ProofsSuccess
$deadline   = $start.AddSeconds($Seconds)

while ((Get-Date) -lt $deadline) {
  if ($Mode -eq "Guided") {
    uv run python -m backend.axiom_engine.derive --system pl --smoke-pl --policy "$PolicyPath" --topk $TopK --epsilon $Eps --seal | Out-Null
  } else {
    uv run python -m backend.axiom_engine.derive --system pl --smoke-pl --seal | Out-Null
  }
}

$endCount = Get-ProofsSuccess
$delta    = [int]($endCount - $startCount)
$elapsed  = ((Get-Date) - $start).TotalSeconds
if ($elapsed -le 0) { $elapsed = 1 }
$pph      = ($delta / $elapsed) * 3600.0
$pphFmt   = ("{0:F2}" -f $pph)

# choose csv path and fields (no ternary)
if ($Mode -eq "Guided") {
  $csvPath    = Join-Path $destDir "guided_runs.csv"
  $policyHash = "live"
  $beamKField = $TopK
} else {
  $csvPath    = Join-Path $destDir "baseline_runs.csv"
  $policyHash = "—"
  $beamKField = ""
}

if (!(Test-Path $csvPath)) {
  'run_id,policy_hash,slice,cpu_hours,beam_k,proofs_per_hour,verify_p50_ms,depth,abstain_pct,block_root' | Set-Content -Encoding utf8 $csvPath
}

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$line  = "$runId,$policyHash,$Slice,0.0,$beamKField,$pphFmt,,,,"
Add-Content -Encoding utf8 $csvPath $line

Write-Host ("{0}: {1}/h over {2:N1}s (Δ={3})" -f $Mode, $pphFmt, $elapsed, $delta)
