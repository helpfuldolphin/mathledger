<#
.SYNOPSIS
    MathLedger Hostile Audit Script - PowerShell Version

.DESCRIPTION
    Executes integrity checks against mathledger.ai.
    Designed for hostile auditors who trust nothing.
    Uses curl.exe for reliable HTTP checking.

    ARCHITECTURE NOTE:
    /demo/ is a SINGLE live demo instance for the CURRENT version only.
    Archived versions are immutable snapshots; they do NOT have hosted demos.

    - If auditing CURRENT version: demo version mismatch is CRITICAL
    - If auditing SUPERSEDED version: demo check is INFO-only (expected mismatch)
      and must verify the superseded disclaimer is present.

.PARAMETER Version
    The version to audit (e.g., v0.2.1, v0.2.2)

.EXAMPLE
    .\hostile_audit.ps1 -Version v0.2.2

.NOTES
    Exit codes:
    0 = All checks passed
    1 = One or more CRITICAL/HIGH checks failed
    2 = Only MEDIUM checks failed (warning)
#>

param(
    [Parameter(Position=0)]
    [string]$Version = "v0.2.2"
)

# Strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

# Configuration
$Base = "https://mathledger.ai"
$Pass = 0
$Fail = 0
$CriticalFail = 0
$HighFail = 0
$MediumFail = 0
$InfoSkip = 0

# Results tracking
$Results = @()

# Determine if auditing CURRENT or SUPERSEDED version
# Fetch releases.json from /versions/ to get current_version
$CurrentVersion = $null
$IsCurrentVersion = $false
try {
    # Fetch the manifest of any version to check releases info, or use the /versions/ page
    # Best approach: fetch a version's manifest.json and check status
    $manifestUrl = "$Base/$Version/manifest.json"
    $manifestContent = & curl.exe -s --max-time 30 $manifestUrl 2>$null
    if ($manifestContent) {
        $manifest = $manifestContent | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($manifest.status -eq "current") {
            $IsCurrentVersion = $true
            $CurrentVersion = $Version
        } else {
            $IsCurrentVersion = $false
            # Try to determine current version from status (e.g., "superseded-by-v0.2.2")
            if ($manifest.status -match "superseded-by-(.+)") {
                $CurrentVersion = $Matches[1]
            }
        }
    }
} catch {
    # If we can't determine, assume auditing current version (strictest mode)
    $IsCurrentVersion = $true
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "          MathLedger Hostile Audit Script                       " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Version:  $Version" -ForegroundColor White
Write-Host "Base URL: $Base" -ForegroundColor White
Write-Host "Started:  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White

if ($IsCurrentVersion) {
    Write-Host "Mode:     CURRENT VERSION (strict demo checks)" -ForegroundColor Green
} else {
    Write-Host "Mode:     SUPERSEDED VERSION (demo mismatch expected)" -ForegroundColor Yellow
    if ($CurrentVersion) {
        Write-Host "Current:  $CurrentVersion (from manifest.status)" -ForegroundColor DarkGray
    }
}
Write-Host ""

function Test-Check {
    param(
        [int]$Num,
        [string]$Url,
        [string]$Expect,
        [string]$Desc,
        [string]$Severity,
        [bool]$Negate = $false,
        [bool]$UseGet = $false,
        [bool]$Skip = $false,
        [string]$SkipReason = ""
    )

    $numStr = $Num.ToString().PadLeft(2)

    # Handle SKIP (for checks that don't apply to this version type)
    if ($Skip) {
        Write-Host "[$numStr] " -NoNewline -ForegroundColor White
        Write-Host "SKIP" -NoNewline -ForegroundColor Cyan
        Write-Host " [INFO] " -NoNewline -ForegroundColor Cyan
        Write-Host "$Desc" -ForegroundColor White
        if ($SkipReason) {
            Write-Host "      Reason: $SkipReason" -ForegroundColor DarkGray
        }
        $script:InfoSkip++
        $script:Results += @{ Num=$Num; Status="SKIP"; Desc=$Desc; Severity="INFO" }
        return
    }

    # Use curl.exe for reliable HTTP checking
    # Some endpoints (demo) reject HEAD, so use GET with -i for headers
    if ($UseGet) {
        $content = & curl.exe -si --max-time 30 $Url 2>$null
    } else {
        $content = & curl.exe -sI --max-time 30 $Url 2>$null
    }

    $matched = $content -match $Expect

    # For negate checks (check 14), we want NO match
    if ($Negate) {
        $passed = -not $matched
    } else {
        $passed = $matched
    }

    $severityColor = switch ($Severity) {
        "CRITICAL" { "Magenta" }
        "HIGH"     { "Red" }
        "MEDIUM"   { "Yellow" }
        "INFO"     { "Cyan" }
        default    { "Gray" }
    }

    if ($passed) {
        Write-Host "[$numStr] " -NoNewline -ForegroundColor White
        Write-Host "PASS" -NoNewline -ForegroundColor Green
        Write-Host " [$Severity] " -NoNewline -ForegroundColor $severityColor
        Write-Host "$Desc" -ForegroundColor White
        $script:Pass++
        $script:Results += @{ Num=$Num; Status="PASS"; Desc=$Desc; Severity=$Severity }
    } else {
        Write-Host "[$numStr] " -NoNewline -ForegroundColor White
        Write-Host "FAIL" -NoNewline -ForegroundColor Red
        Write-Host " [$Severity] " -NoNewline -ForegroundColor $severityColor
        Write-Host "$Desc" -ForegroundColor White
        Write-Host "      Expected: $Expect" -ForegroundColor DarkGray
        Write-Host "      URL: $Url" -ForegroundColor DarkGray
        $script:Fail++
        $script:Results += @{ Num=$Num; Status="FAIL"; Desc=$Desc; Severity=$Severity }

        switch ($Severity) {
            "CRITICAL" { $script:CriticalFail++ }
            "HIGH"     { $script:HighFail++ }
            "MEDIUM"   { $script:MediumFail++ }
            # INFO failures don't block (they're informational)
        }
    }
}

# ==============================================================================
# STRUCTURAL CHECKS (1-11)
# ==============================================================================

Write-Host "--- Structural Checks ------------------------------------------" -ForegroundColor Yellow
Write-Host ""

# Check 1: Root redirect (HIGH)
Test-Check -Num 1 -Url "$Base/" -Expect "30[12]" -Desc "Root redirects" -Severity "HIGH"

# Check 2: Landing page exists (CRITICAL)
Test-Check -Num 2 -Url "$Base/$Version/" -Expect "200" -Desc "Landing page exists" -Severity "CRITICAL"

# Check 3: Manifest exists (HIGH)
Test-Check -Num 3 -Url "$Base/$Version/manifest.json" -Expect "200" -Desc "Manifest exists" -Severity "HIGH"

# Check 4: Scope lock doc (MEDIUM)
Test-Check -Num 4 -Url "$Base/$Version/docs/scope-lock/" -Expect "200" -Desc "Scope lock doc exists" -Severity "MEDIUM"

# Check 5: Explanation doc (MEDIUM)
Test-Check -Num 5 -Url "$Base/$Version/docs/explanation/" -Expect "200" -Desc "Explanation doc exists" -Severity "MEDIUM"

# Check 6: Invariants doc (MEDIUM)
Test-Check -Num 6 -Url "$Base/$Version/docs/invariants/" -Expect "200" -Desc "Invariants doc exists" -Severity "MEDIUM"

# Check 7: Fixtures index (MEDIUM)
Test-Check -Num 7 -Url "$Base/$Version/fixtures/" -Expect "200" -Desc "Fixtures index exists" -Severity "MEDIUM"

# Check 8: Evidence pack landing (HIGH)
Test-Check -Num 8 -Url "$Base/$Version/evidence-pack/" -Expect "200" -Desc "Evidence pack landing exists" -Severity "HIGH"

# Check 9: Verifier page (HIGH)
Test-Check -Num 9 -Url "$Base/$Version/evidence-pack/verify/" -Expect "200" -Desc "Verifier page exists" -Severity "HIGH"

# Check 10: Demo backend (HIGH) - uses GET because demo rejects HEAD
Test-Check -Num 10 -Url "$Base/demo/healthz" -Expect "200" -Desc "Demo backend responds" -Severity "HIGH" -UseGet $true

# Check 11: Demo UI (HIGH) - uses GET because demo rejects HEAD
Test-Check -Num 11 -Url "$Base/demo/" -Expect "200" -Desc "Demo UI loads" -Severity "HIGH" -UseGet $true

Write-Host ""

# ==============================================================================
# HEADER CHECKS (12-15)
# ==============================================================================

Write-Host "--- Header Checks ----------------------------------------------" -ForegroundColor Yellow
Write-Host ""

# Check 12: Archive has immutable cache (MEDIUM)
Test-Check -Num 12 -Url "$Base/$Version/manifest.json" -Expect "immutable" -Desc "Archive has immutable cache" -Severity "MEDIUM"

# Check 13: Demo uses worker proxy (HIGH) - uses GET
Test-Check -Num 13 -Url "$Base/demo/healthz" -Expect "X-Proxied-By" -Desc "Demo uses worker proxy" -Severity "HIGH" -UseGet $true

# Check 14: Archive NOT proxied (MEDIUM) - NEGATE check
Test-Check -Num 14 -Url "$Base/$Version/manifest.json" -Expect "X-Proxied-By" -Desc "Archive not proxied" -Severity "MEDIUM" -Negate $true

# Check 15: Version match (CRITICAL for current, SKIP for superseded)
if ($IsCurrentVersion) {
    Test-Check -Num 15 -Url "$Base/demo/healthz" -Expect "x-mathledger-version: $Version" -Desc "Demo version matches audited version" -Severity "CRITICAL" -UseGet $true
} else {
    Test-Check -Num 15 -Url "$Base/demo/healthz" -Expect "x-mathledger-version: $Version" -Desc "Demo version match (N/A for superseded)" -Severity "INFO" -Skip $true -SkipReason "/demo/ runs CURRENT version; superseded versions have local-only demo"
}

# Check 16: Superseded disclaimer (HIGH for superseded, SKIP for current)
# Superseded versions must state that /demo/ runs the current version
if (-not $IsCurrentVersion) {
    $supersededDisclaimer = "hosted demo.*runs.*current version"
    Test-Check -Num 16 -Url "$Base/$Version/" -Expect $supersededDisclaimer -Desc "Superseded disclaimer present" -Severity "HIGH" -UseGet $true
} else {
    Test-Check -Num 16 -Url "$Base/$Version/" -Expect ".*" -Desc "Superseded disclaimer (N/A for current)" -Severity "INFO" -Skip $true -SkipReason "Current version does not need superseded disclaimer"
}

Write-Host ""

# ==============================================================================
# SUMMARY
# ==============================================================================

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "RESULTS TABLE" -ForegroundColor Cyan
Write-Host ""
Write-Host "  #    Status Severity   Description" -ForegroundColor DarkGray
Write-Host "  ---- ------ ---------- -----------" -ForegroundColor DarkGray

foreach ($r in $Results) {
    $statusColor = switch ($r.Status) {
        "PASS" { "Green" }
        "SKIP" { "Cyan" }
        default { "Red" }
    }
    $sevColor = switch ($r.Severity) {
        "CRITICAL" { "Magenta" }
        "HIGH"     { "Red" }
        "MEDIUM"   { "Yellow" }
        "INFO"     { "Cyan" }
        default    { "Gray" }
    }
    $numStr = $r.Num.ToString().PadLeft(2)
    Write-Host "  $numStr   " -NoNewline -ForegroundColor White
    Write-Host "$($r.Status.PadRight(6)) " -NoNewline -ForegroundColor $statusColor
    Write-Host "$($r.Severity.PadRight(10)) " -NoNewline -ForegroundColor $sevColor
    Write-Host $r.Desc -ForegroundColor White
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host ""

$TotalChecks = 16
$ActiveChecks = $TotalChecks - $InfoSkip
$passColor = if ($Pass -eq $ActiveChecks) { "Green" } else { "Yellow" }
$failColor = if ($Fail -eq 0) { "Green" } else { "Red" }
$skipColor = if ($InfoSkip -gt 0) { "Cyan" } else { "DarkGray" }

Write-Host "  PASS: $Pass / $ActiveChecks (active checks)" -ForegroundColor $passColor
Write-Host "  FAIL: $Fail / $ActiveChecks (active checks)" -ForegroundColor $failColor
Write-Host "  SKIP: $InfoSkip / $TotalChecks (N/A for this version type)" -ForegroundColor $skipColor
Write-Host ""
Write-Host "  Failures by severity:" -ForegroundColor DarkGray

$critColor = if ($CriticalFail -gt 0) { "Magenta" } else { "Green" }
$highColor = if ($HighFail -gt 0) { "Red" } else { "Green" }
$medColor = if ($MediumFail -gt 0) { "Yellow" } else { "Green" }

Write-Host "    CRITICAL: $CriticalFail" -ForegroundColor $critColor
Write-Host "    HIGH:     $HighFail" -ForegroundColor $highColor
Write-Host "    MEDIUM:   $MediumFail" -ForegroundColor $medColor
Write-Host ""

# Mode-specific guidance
if ($IsCurrentVersion) {
    Write-Host "  [CURRENT VERSION MODE]" -ForegroundColor Green
    Write-Host "  Demo version must match: Check 15 is CRITICAL" -ForegroundColor DarkGray
} else {
    Write-Host "  [SUPERSEDED VERSION MODE]" -ForegroundColor Yellow
    Write-Host "  Demo mismatch expected: Check 15 skipped" -ForegroundColor DarkGray
    Write-Host "  Superseded disclaimer: Check 16 is HIGH" -ForegroundColor DarkGray
}
Write-Host ""

# Determine exit code and status
if ($Fail -eq 0) {
    Write-Host "STATUS: ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host ""
    exit 0
} elseif ($CriticalFail -gt 0 -or $HighFail -gt 0) {
    Write-Host "STATUS: AUDIT FAILED (CRITICAL/HIGH failures)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Do NOT proceed with acquisition until these issues are resolved." -ForegroundColor Red
    Write-Host ""
    exit 1
} else {
    Write-Host "STATUS: AUDIT PASSED WITH WARNINGS (MEDIUM failures only)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Proceed with caution. MEDIUM issues should be tracked." -ForegroundColor Yellow
    Write-Host ""
    exit 2
}
