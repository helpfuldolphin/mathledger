#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Documentation Governance Watchdog - First Light Integration

.DESCRIPTION
    Runs documentation governance drift radar in watchdog mode to detect:
    - Uplift claims without "integrated-run pending" disclaimer
    - TDA enforcement claims before runner wiring complete
    - Contradictions to Phase I-II disclaimers

.PARAMETER Mode
    Scan mode: full-scan, watchdog, or pr-diff (default: watchdog)

.PARAMETER Output
    Output directory for reports (default: artifacts/drift)

.PARAMETER FailOnWarn
    Exit with error on warnings (not just critical violations)

.EXAMPLE
    # Quick watchdog check
    .\scripts\governance-watchdog.ps1

.EXAMPLE
    # Full scan before commit
    .\scripts\governance-watchdog.ps1 -Mode full-scan

.EXAMPLE
    # PR diff check
    git diff origin/main...HEAD -- '*.md' > artifacts/drift/pr_diff.patch
    .\scripts\governance-watchdog.ps1 -Mode pr-diff

.NOTES
    Exit Codes:
      0 - PASS (no violations)
      1 - FAIL (critical violations)
      2 - WARN (non-critical violations)
      3 - ERROR (infrastructure failure)
      4 - SKIP (no files to scan)
#>

param(
    [Parameter()]
    [ValidateSet("full-scan", "watchdog", "pr-diff")]
    [string]$Mode = "watchdog",
    
    [Parameter()]
    [string]$Output = "artifacts/drift",
    
    [Parameter()]
    [switch]$FailOnWarn = $false
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Colors
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

# Header
Write-ColorOutput "`n╔════════════════════════════════════════════════════════════╗" Cyan
Write-ColorOutput "║  Documentation Governance Watchdog - First Light           ║" Cyan
Write-ColorOutput "║  The organism does not move unless the Cortex approves.    ║" Cyan
Write-ColorOutput "╚════════════════════════════════════════════════════════════╝`n" Cyan

# Ensure output directory exists
$outputPath = Join-Path $PSScriptRoot ".." $Output
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

# Run radar
Write-ColorOutput "Running documentation governance radar..." White
Write-ColorOutput "  Mode:   $Mode" Gray
Write-ColorOutput "  Output: $outputPath`n" Gray

$radarScript = Join-Path $PSScriptRoot "radars" "doc_governance_drift_radar.py"
$docsPath = Join-Path $PSScriptRoot ".." "docs"

try {
    $result = python $radarScript `
        --mode $Mode `
        --docs $docsPath `
        --output $outputPath
    
    $exitCode = $LASTEXITCODE
    
    # Print summary
    Write-Host ""
    Write-ColorOutput "════════════════════════════════════════════════════════════" Cyan
    
    switch ($exitCode) {
        0 {
            Write-ColorOutput "✅ PASS: No governance violations detected" Green
            Write-ColorOutput "   The organism maintains narrative integrity." Gray
            $success = $true
        }
        1 {
            Write-ColorOutput "❌ FAIL: Critical governance violations detected" Red
            Write-ColorOutput "   ⛔ ORGANISM NOT ALIVE" Red
            Write-ColorOutput "   No document may imply 'organism alive' until First Light completes." Gray
            $success = $false
        }
        2 {
            Write-ColorOutput "⚠️  WARN: Non-critical governance violations detected" Yellow
            if ($FailOnWarn) {
                Write-ColorOutput "   Treating warnings as failures (--FailOnWarn)" Yellow
                $success = $false
            } else {
                Write-ColorOutput "   Review recommended but not blocking." Gray
                $success = $true
            }
        }
        3 {
            Write-ColorOutput "💥 ERROR: Infrastructure failure" Red
            $success = $false
        }
        4 {
            Write-ColorOutput "⏭️  SKIP: No files to scan" Gray
            $success = $true
        }
        default {
            Write-ColorOutput "❓ Unknown exit code: $exitCode" Red
            $success = $false
        }
    }
    
    Write-ColorOutput "════════════════════════════════════════════════════════════`n" Cyan
    
    # Show report locations
    $reportPath = Join-Path $outputPath "doc_governance_drift_summary.md"
    if (Test-Path $reportPath) {
        Write-ColorOutput "📄 Detailed report:" White
        Write-ColorOutput "   $reportPath`n" Gray
        
        # Show first few violations if any
        if ($exitCode -eq 1 -or $exitCode -eq 2) {
            Write-ColorOutput "Preview (first 20 lines):" Yellow
            Get-Content $reportPath -TotalCount 20 | ForEach-Object {
                Write-Host "   $_" -ForegroundColor Gray
            }
            Write-Host ""
        }
    }
    
    # Exit with appropriate code
    if ($success) {
        exit 0
    } else {
        if ($FailOnWarn -and $exitCode -eq 2) {
            exit 1
        } else {
            exit $exitCode
        }
    }
}
catch {
    Write-ColorOutput "💥 ERROR: Failed to run governance radar" Red
    Write-ColorOutput "   $_" Red
    exit 3
}
