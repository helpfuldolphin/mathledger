# PowerShell API Contract Tests
#
# Validates that FastAPI endpoints return data in formats expected by PowerShell scripts.
# Tests type coercion, field naming, and JSON parsing compatibility.
#
# Usage: powershell -File tests/interop/Test-APIContracts.ps1

param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$ApiKey = "devkey"
)

$ErrorActionPreference = "Stop"

# Test framework
$script:PassCount = 0
$script:FailCount = 0

function Write-TestHeader {
    param([string]$Title)
    Write-Host "`n$('='*60)" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "$('='*60)" -ForegroundColor Cyan
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if ($Condition) {
        $script:PassCount++
        Write-Host "✅ [PASS] $Message" -ForegroundColor Green
    } else {
        $script:FailCount++
        Write-Host "❌ [FAIL] $Message" -ForegroundColor Red
        throw "Assertion failed: $Message"
    }
}

function Assert-Equal {
    param(
        $Actual,
        $Expected,
        [string]$Message
    )
    if ($Actual -eq $Expected) {
        $script:PassCount++
        Write-Host "✅ [PASS] $Message" -ForegroundColor Green
    } else {
        $script:FailCount++
        Write-Host "❌ [FAIL] $Message (Expected: $Expected, Got: $Actual)" -ForegroundColor Red
        throw "Assertion failed: $Message"
    }
}

function Assert-NotNull {
    param(
        $Value,
        [string]$Message
    )
    if ($null -ne $Value) {
        $script:PassCount++
        Write-Host "✅ [PASS] $Message" -ForegroundColor Green
    } else {
        $script:FailCount++
        Write-Host "❌ [FAIL] $Message (Value is null)" -ForegroundColor Red
        throw "Assertion failed: $Message"
    }
}

function Assert-Type {
    param(
        $Value,
        [string]$TypeName,
        [string]$Message
    )
    $actualType = $Value.GetType().Name
    if ($actualType -eq $TypeName -or $Value -is [type]$TypeName) {
        $script:PassCount++
        Write-Host "✅ [PASS] $Message (Type: $actualType)" -ForegroundColor Green
    } else {
        $script:FailCount++
        Write-Host "❌ [FAIL] $Message (Expected: $TypeName, Got: $actualType)" -ForegroundColor Red
        throw "Assertion failed: $Message"
    }
}

function Test-MetricsEndpoint {
    Write-Host "`n--- Testing /metrics endpoint ---" -ForegroundColor Yellow

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/metrics" -Method GET -TimeoutSec 10

        # Test 1: Required fields exist
        Assert-NotNull $response.proofs "Metrics has proofs field"
        Assert-NotNull $response.block_count "Metrics has block_count field"
        Assert-NotNull $response.max_depth "Metrics has max_depth field"

        # Test 2: Proofs structure
        Assert-NotNull $response.proofs.success "Metrics has proofs.success field"
        Assert-NotNull $response.proofs.failure "Metrics has proofs.failure field"

        # Test 3: Type validation (PowerShell auto-converts JSON numbers)
        Assert-Type $response.proofs.success "Int32" "proofs.success is integer"
        Assert-Type $response.block_count "Int32" "block_count is integer"
        Assert-Type $response.max_depth "Int32" "max_depth is integer"

        # Test 4: Additional fields (used by sanity.ps1 line 148)
        if ($response.statements) {
            Assert-NotNull $response.statements.total "statements.total exists"
        } elseif ($response.statement_counts) {
            Assert-NotNull $response.statement_counts "statement_counts exists"
        } else {
            Write-Host "⚠️ [WARN] No statement count field found" -ForegroundColor Yellow
        }

        # Test 5: Success rate (if present)
        if ($response.success_rate) {
            Assert-True ($response.success_rate -ge 0 -and $response.success_rate -le 100) `
                "success_rate is percentage (0-100)"
        }

        Write-Host "[PASS] Metrics endpoint contract verified" -ForegroundColor Green

    } catch {
        Write-Host "[FAIL] Metrics endpoint test failed: $_" -ForegroundColor Red
        throw
    }
}

function Test-HeartbeatEndpoint {
    Write-Host "`n--- Testing /heartbeat.json endpoint ---" -ForegroundColor Yellow

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/heartbeat.json" -Method GET -TimeoutSec 10

        # Test 1: Required fields (healthcheck.ps1 lines 142-146)
        Assert-NotNull $response.ok "Heartbeat has ok field"
        Assert-NotNull $response.ts "Heartbeat has ts field"
        Assert-NotNull $response.proofs "Heartbeat has proofs field"
        Assert-NotNull $response.blocks "Heartbeat has blocks field"

        # Test 2: Boolean type (JSON true → PowerShell $true)
        Assert-Type $response.ok "Boolean" "ok is boolean"
        Assert-True ($response.ok -eq $true -or $response.ok -eq $false) "ok is true or false"

        # Test 3: Nested structure
        Assert-NotNull $response.proofs.success "proofs.success exists"
        Assert-NotNull $response.blocks.height "blocks.height exists"
        Assert-NotNull $response.blocks.latest "blocks.latest exists"

        # Test 4: Merkle can be null or string
        if ($null -ne $response.blocks.latest.merkle) {
            Assert-Type $response.blocks.latest.merkle "String" "merkle is string when not null"
        } else {
            Write-Host "✅ [PASS] merkle is null (valid)" -ForegroundColor Green
            $script:PassCount++
        }

        # Test 5: Timestamp format (ISO 8601)
        Assert-Type $response.ts "String" "ts is string"
        $timestamp = [DateTime]::Parse($response.ts)
        Assert-NotNull $timestamp "ts parses as DateTime"

        # Test 6: Redis info (optional)
        if ($response.redis) {
            Assert-NotNull $response.redis.ml_jobs_len "redis.ml_jobs_len exists"
            Assert-Type $response.redis.ml_jobs_len "Int32" "redis.ml_jobs_len is integer"
        }

        Write-Host "[PASS] Heartbeat endpoint contract verified" -ForegroundColor Green

    } catch {
        Write-Host "[FAIL] Heartbeat endpoint test failed: $_" -ForegroundColor Red
        throw
    }
}

function Test-BlocksEndpoint {
    Write-Host "`n--- Testing /blocks/latest endpoint ---" -ForegroundColor Yellow

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/blocks/latest" -Method GET -TimeoutSec 10 -ErrorAction SilentlyContinue

        if ($null -eq $response) {
            Write-Host "⚠️ [WARN] No blocks available (404 expected)" -ForegroundColor Yellow
            return
        }

        # Test 1: Required fields
        Assert-NotNull $response.block_number "blocks/latest has block_number"
        Assert-NotNull $response.merkle_root "blocks/latest has merkle_root"
        Assert-NotNull $response.created_at "blocks/latest has created_at"
        Assert-NotNull $response.header "blocks/latest has header"

        # Test 2: Type validation
        Assert-Type $response.block_number "Int32" "block_number is integer"
        Assert-Type $response.merkle_root "String" "merkle_root is string"
        Assert-Type $response.created_at "String" "created_at is string"

        # Test 3: Timestamp parsing
        $timestamp = [DateTime]::Parse($response.created_at)
        Assert-NotNull $timestamp "created_at parses as DateTime"

        Write-Host "[PASS] Blocks/latest endpoint contract verified" -ForegroundColor Green

    } catch {
        if ($_.Exception.Response.StatusCode -eq 404) {
            Write-Host "⚠️ [WARN] 404 response (no blocks available)" -ForegroundColor Yellow
        } else {
            Write-Host "[FAIL] Blocks endpoint test failed: $_" -ForegroundColor Red
            throw
        }
    }
}

function Test-HealthEndpoint {
    Write-Host "`n--- Testing /health endpoint ---" -ForegroundColor Yellow

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -TimeoutSec 10

        # Test 1: Required fields
        Assert-NotNull $response.status "health has status field"
        Assert-NotNull $response.timestamp "health has timestamp field"

        # Test 2: Type validation
        Assert-Type $response.status "String" "status is string"
        Assert-Type $response.timestamp "String" "timestamp is string"

        # Test 3: Status value
        Assert-Equal $response.status "healthy" "status is 'healthy'"

        # Test 4: Timestamp parsing
        $timestamp = [DateTime]::Parse($response.timestamp)
        Assert-NotNull $timestamp "timestamp parses as DateTime"

        Write-Host "[PASS] Health endpoint contract verified" -ForegroundColor Green

    } catch {
        Write-Host "[FAIL] Health endpoint test failed: $_" -ForegroundColor Red
        throw
    }
}

function Test-StatementsEndpointAuth {
    Write-Host "`n--- Testing /statements endpoint authentication ---" -ForegroundColor Yellow

    try {
        # Test 1: Without API key (should fail)
        try {
            $response = Invoke-RestMethod -Uri "$BaseUrl/statements?hash=abc123" -Method GET -TimeoutSec 10 -ErrorAction Stop
            Assert-True $false "Should have returned 401 without API key"
        } catch {
            if ($_.Exception.Response.StatusCode -eq 401) {
                Write-Host "✅ [PASS] Returns 401 without API key" -ForegroundColor Green
                $script:PassCount++
            } else {
                throw
            }
        }

        # Test 2: With invalid hash format (should fail with 400)
        try {
            $headers = @{ "X-API-Key" = $ApiKey }
            $response = Invoke-RestMethod -Uri "$BaseUrl/statements?hash=invalid" -Method GET -Headers $headers -TimeoutSec 10 -ErrorAction Stop
            Assert-True $false "Should have returned 400 for invalid hash"
        } catch {
            if ($_.Exception.Response.StatusCode -eq 400) {
                Write-Host "✅ [PASS] Returns 400 for invalid hash format" -ForegroundColor Green
                $script:PassCount++
            } else {
                throw
            }
        }

        Write-Host "[PASS] Statements authentication contract verified" -ForegroundColor Green

    } catch {
        Write-Host "[FAIL] Statements auth test failed: $_" -ForegroundColor Red
        throw
    }
}

function Test-TypeCoercion {
    Write-Host "`n--- Testing PowerShell type coercion ---" -ForegroundColor Yellow

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/heartbeat.json" -Method GET -TimeoutSec 10

        # Test 1: Boolean conversion (JSON true/false → PS $true/$false)
        $okValue = $response.ok
        Assert-True ($okValue -is [bool]) "Boolean coerces correctly"

        # Test 2: Integer preservation (JSON number → PS Int32)
        $heightValue = $response.blocks.height
        Assert-True ($heightValue -is [int]) "Integer coerces correctly"

        # Test 3: Null handling (JSON null → PS $null)
        if ($null -eq $response.blocks.latest.merkle) {
            Write-Host "✅ [PASS] Null coerces correctly" -ForegroundColor Green
            $script:PassCount++
        } else {
            # String is also valid
            Assert-True ($response.blocks.latest.merkle -is [string]) "Merkle is string or null"
        }

        # Test 4: String preservation
        $tsValue = $response.ts
        Assert-True ($tsValue -is [string]) "String coerces correctly"

        # Test 5: Object/hashtable conversion
        Assert-True ($response.proofs -is [PSCustomObject] -or $response.proofs -is [hashtable]) `
            "Nested object coerces correctly"

        Write-Host "[PASS] Type coercion verified" -ForegroundColor Green

    } catch {
        Write-Host "[FAIL] Type coercion test failed: $_" -ForegroundColor Red
        throw
    }
}

function Test-FieldNameConsistency {
    Write-Host "`n--- Testing field name consistency ---" -ForegroundColor Yellow

    try {
        $metrics = Invoke-RestMethod -Uri "$BaseUrl/metrics" -Method GET -TimeoutSec 10
        $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -TimeoutSec 10
        $heartbeat = Invoke-RestMethod -Uri "$BaseUrl/heartbeat.json" -Method GET -TimeoutSec 10

        # Test 1: Snake_case convention
        $metricsFields = $metrics.PSObject.Properties.Name
        foreach ($field in $metricsFields) {
            if ($field -match "_") {
                # Has underscore, verify lowercase
                Assert-True ($field -ceq $field.ToLower()) "Field $field uses snake_case"
            }
        }

        # Test 2: Case sensitivity (PowerShell is case-insensitive, but verify naming)
        Assert-NotNull $metrics.block_count "block_count (not blockCount)"
        Assert-NotNull $metrics.max_depth "max_depth (not maxDepth)"

        # Test 3: Timestamp field naming difference (documented)
        Assert-NotNull $health.timestamp "health uses 'timestamp'"
        Assert-NotNull $heartbeat.ts "heartbeat uses 'ts'"

        Write-Host "[PASS] Field naming consistency verified" -ForegroundColor Green

    } catch {
        Write-Host "[FAIL] Field naming test failed: $_" -ForegroundColor Red
        throw
    }
}

function Test-JSONRoundTrip {
    Write-Host "`n--- Testing JSON round-trip fidelity ---" -ForegroundColor Yellow

    try {
        # Get response and convert to/from JSON
        $response = Invoke-RestMethod -Uri "$BaseUrl/metrics" -Method GET -TimeoutSec 10

        # Serialize to JSON
        $json = $response | ConvertTo-Json -Depth 10

        # Deserialize back
        $roundTrip = $json | ConvertFrom-Json

        # Test 1: Integer values preserved
        Assert-Equal $roundTrip.proofs.success $response.proofs.success `
            "Integer survives round-trip"

        # Test 2: String values preserved
        if ($response.policy) {
            Assert-Equal $roundTrip.policy.hash $response.policy.hash `
                "String survives round-trip"
        }

        # Test 3: Nested structure preserved
        Assert-NotNull $roundTrip.proofs "Nested object survives round-trip"
        Assert-NotNull $roundTrip.proofs.success "Nested field survives round-trip"

        Write-Host "[PASS] JSON round-trip fidelity verified" -ForegroundColor Green

    } catch {
        Write-Host "[FAIL] JSON round-trip test failed: $_" -ForegroundColor Red
        throw
    }
}

# Main test execution
try {
    Write-TestHeader "POWERSHELL API CONTRACT TEST SUITE"
    Write-Host "Testing PowerShell ↔ Python (FastAPI)" -ForegroundColor Cyan
    Write-Host "Base URL: $BaseUrl" -ForegroundColor Gray
    Write-Host ""

    # Run all tests
    Test-MetricsEndpoint
    Test-HeartbeatEndpoint
    Test-BlocksEndpoint
    Test-HealthEndpoint
    Test-StatementsEndpointAuth
    Test-TypeCoercion
    Test-FieldNameConsistency
    Test-JSONRoundTrip

    # Summary
    Write-Host "`n$('='*60)" -ForegroundColor Cyan
    Write-Host "Test Summary: $script:PassCount passed, $script:FailCount failed" -ForegroundColor Cyan
    Write-Host "$('='*60)" -ForegroundColor Cyan

    if ($script:FailCount -eq 0) {
        Write-Host "`n[PASS] Interop Verified langs=2 (PS↔Python) drift≤ε" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "`n[FAIL] Interop drift detected" -ForegroundColor Red
        exit 1
    }

} catch {
    Write-Host "`nFatal test error: $_" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
