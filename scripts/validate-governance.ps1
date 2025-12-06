# Governance validation CI script (PowerShell)
# Usage: powershell -File .\scripts\validate-governance.ps1
# Exit codes: 0=LAWFUL, 1=UNLAWFUL

$ErrorActionPreference = "Stop"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "GOVERNANCE VALIDATION (CI)" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check for required files
$GovernanceChain = "artifacts\governance\governance_chain.json"
$DeclaredRoots = "artifacts\governance\declared_roots.json"

if (-not (Test-Path $GovernanceChain)) {
    Write-Host "❌ ERROR: Governance chain not found: $GovernanceChain" -ForegroundColor Red
    Write-Host "   Run: python backend\governance\export.py"
    exit 1
}

if (-not (Test-Path $DeclaredRoots)) {
    Write-Host "❌ ERROR: Declared roots not found: $DeclaredRoots" -ForegroundColor Red
    Write-Host "   Run: python backend\governance\export.py --db-url `$env:DATABASE_URL"
    exit 1
}

# Run validation
Write-Host "📋 Validating governance artifacts..." -ForegroundColor White
Write-Host "   - $GovernanceChain"
Write-Host "   - $DeclaredRoots"
Write-Host ""

try {
    uv run python backend\governance\validator.py `
        --governance $GovernanceChain `
        --roots $DeclaredRoots

    $ValidationResult = $LASTEXITCODE

    if ($ValidationResult -eq 0) {
        Write-Host ""
        Write-Host "✅ VERDICT: LAWFUL" -ForegroundColor Green
        Write-Host "   All provenance seals validated successfully."
        Write-Host ""

        # Generate verdict if requested
        if ($args[0] -eq "--generate-verdict") {
            Write-Host "📄 Generating governance_verdict.md..."
            if (Test-Path "governance_verdict.md") {
                Write-Host "   ✓ governance_verdict.md present" -ForegroundColor Green
            }
        }

        exit 0
    }
    else {
        Write-Host ""
        Write-Host "❌ VERDICT: UNLAWFUL" -ForegroundColor Red
        Write-Host "   Provenance seal violations detected."
        Write-Host "   See errors above for details."
        Write-Host ""
        exit 1
    }
}
catch {
    Write-Host ""
    Write-Host "❌ ERROR: Validation script failed" -ForegroundColor Red
    Write-Host "   $($_.Exception.Message)"
    exit 1
}
