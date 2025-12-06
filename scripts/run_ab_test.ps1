# MathLedger A/B Testing Script
# Runs Baseline (no policy) and Guided (with policy) experiments

param(
    [string]$Mode = "Both",  # "Baseline", "Guided", or "Both"
    [string]$Slice = "PL-1",
    [double]$CPUh = 1.0,
    [int]$TopK = 16,
    [double]$Eps = 0.1,
    [int]$DepthMax = 4,
    [string]$PolicyHash = ""
)

$ErrorActionPreference = "Stop"

# Function to write timestamped log message
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
}

# Function to run derivation and capture metrics
function Run-Derivation {
    param(
        [string]$SystemSlug,
        [string]$Slice,
        [int]$Steps,
        [int]$DepthMax,
        [int]$MaxBreadth,
        [int]$MaxTotal,
        [string]$PolicyHash = "",
        [string]$ExperimentType
    )

    $startTime = Get-Date
    $runId = Get-Date -Format "yyyyMMdd-HHmmss"

    Write-Log "Starting $ExperimentType experiment: $runId"
    Write-Log "Slice: $Slice, Steps: $Steps, DepthMax: $DepthMax"

    # Build derive command
    $deriveArgs = @(
        "-m", "backend.axiom_engine.derive"
        "--system", $SystemSlug
        "--steps", $Steps
        "--depth-max", $DepthMax
        "--max-breadth", $MaxBreadth
        "--max-total", $MaxTotal
        "--seal"
    )

    # Add policy if provided
    if ($PolicyHash -ne "") {
        $deriveArgs += @("--policy-hash", $PolicyHash)
    }

    # Run derivation
    $deriveResult = & uv run python @deriveArgs 2>&1
    $exitCode = $LASTEXITCODE

    $endTime = Get-Date
    $duration = $endTime - $startTime
    $cpuHours = [math]::Round($duration.TotalHours, 4)

    if ($exitCode -eq 0) {
        Write-Log "$ExperimentType experiment completed successfully in $($duration.TotalSeconds) seconds"

        # Get API data
        try {
            $headers = @{ "X-API-Key" = "devkey" }
            $metrics = Invoke-RestMethod -Uri "http://localhost:8000/metrics" -Headers $headers -ErrorAction Stop
            $block = Invoke-RestMethod -Uri "http://localhost:8000/blocks/latest" -Headers $headers -ErrorAction Stop

            $proofsPerHour = [math]::Round(($metrics.proofs.success / $cpuHours), 2)
            $blockRootShort = $block.merkle_root.Substring(0, 8)
            $abstainPct = [math]::Round((($metrics.proofs.failure / ($metrics.proofs.success + $metrics.proofs.failure)) * 100), 1)

            # Extract max depth from derive output
            $maxDepth = $DepthMax
            $depthLine = $deriveResult | Where-Object { $_ -match "max.*depth.*(\d+)" }
            if ($depthLine) {
                $maxDepth = [int]($depthLine -replace ".*max.*depth.*(\d+).*", '$1')
            }

            $result = @{
                run_id = $runId
                policy_hash = $PolicyHash
                slice = $Slice
                cpu_hours = $cpuHours
                beam_k = $TopK
                proofs_per_hour = $proofsPerHour
                median_verify_ms = 0  # Placeholder
                depth = $maxDepth
                abstain_pct = $abstainPct
                block_root_short = $blockRootShort
            }

            return $result

        } catch {
            Write-Log "API data fetch failed: $_" "WARN"
            # Return basic result without API data
            return @{
                run_id = $runId
                policy_hash = $PolicyHash
                slice = $Slice
                cpu_hours = $cpuHours
                beam_k = $TopK
                proofs_per_hour = 0
                median_verify_ms = 0
                depth = $DepthMax
                abstain_pct = 0
                block_root_short = "unknown"
            }
        }
    } else {
        Write-Log "$ExperimentType experiment failed with exit code $exitCode" "ERROR"
        Write-Log "Output: $deriveResult" "ERROR"
        return $null
    }
}

# Function to append result to CSV
function Add-ToCSV {
    param(
        [string]$FilePath,
        [hashtable]$Result
    )

    if ($Result -ne $null) {
        $csvLine = "$($Result.run_id),$($Result.policy_hash),$($Result.slice),$($Result.cpu_hours),$($Result.beam_k),$($Result.proofs_per_hour),$($Result.median_verify_ms),$($Result.depth),$($Result.abstain_pct),$($Result.block_root_short)"
        Add-Content -Path $FilePath -Value $csvLine
        Write-Log "Result appended to $FilePath"
    }
}

# Main execution
try {
    Write-Log "Starting MathLedger A/B Test"
    Write-Log "Parameters: Slice=$Slice, CPUh=$CPUh, TopK=$TopK, Eps=$Eps, PolicyHash=$PolicyHash"

    # Calculate steps based on CPU hours (rough estimate: 1 step ≈ 0.01 CPU hours)
    $steps = [math]::Round($CPUh * 100)

    # Create artifacts directory
    if (-not (Test-Path "artifacts/wpv5")) {
        New-Item -ItemType Directory -Path "artifacts/wpv5" -Force | Out-Null
    }

    # Initialize CSV files if they don't exist
    $baselineCsv = "artifacts/wpv5/baseline_runs.csv"
    $guidedCsv = "artifacts/wpv5/guided_runs.csv"

    if (-not (Test-Path $baselineCsv)) {
        $header = "run_id,policy_hash,slice,cpu_hours,beam_k,proofs_per_hour,median_verify_ms,depth,abstain_pct,block_root_short"
        Set-Content -Path $baselineCsv -Value $header
    }

    if (-not (Test-Path $guidedCsv)) {
        $header = "run_id,policy_hash,slice,cpu_hours,beam_k,proofs_per_hour,median_verify_ms,depth,abstain_pct,block_root_short"
        Set-Content -Path $guidedCsv -Value $header
    }

    # Run experiments based on mode
    if ($Mode -eq "Baseline" -or $Mode -eq "Both") {
        Write-Log "=== RUNNING BASELINE EXPERIMENT ==="
        $baselineResult = Run-Derivation -SystemSlug "pl" -Slice $Slice -Steps $steps -DepthMax $DepthMax -MaxBreadth 1000 -MaxTotal 5000 -ExperimentType "Baseline"
        Add-ToCSV -FilePath $baselineCsv -Result $baselineResult
    }

    if ($Mode -eq "Guided" -or $Mode -eq "Both") {
        Write-Log "=== RUNNING GUIDED EXPERIMENT ==="
        $guidedResult = Run-Derivation -SystemSlug "pl" -Slice $Slice -Steps $steps -DepthMax $DepthMax -MaxBreadth 1000 -MaxTotal 5000 -PolicyHash $PolicyHash -ExperimentType "Guided"
        Add-ToCSV -FilePath $guidedCsv -Result $guidedResult
    }

    Write-Log "A/B Test completed successfully"
    Write-Log "Baseline CSV: $baselineCsv"
    Write-Log "Guided CSV: $guidedCsv"

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
