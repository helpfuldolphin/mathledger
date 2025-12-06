param([int]$Seconds = 60)
$deadline = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $deadline) {
  uv run python -m backend.axiom_engine.derive --system pl --smoke-pl --seal *> $null
}
