# MathLedger Snapshot Exporter Script
# This script exports statements and proofs tables to JSONL format

param(
    [string]$ConfigPath = "config/nightly.env",
    [string]$OutputDir = "",
    [switch]$IncludeDependencies = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to load configuration from .env file
function Load-Config {
    param([string]$ConfigPath)

    $config = @{}
    if (Test-Path $ConfigPath) {
        Get-Content $ConfigPath | ForEach-Object {
            if ($_ -match '^([^#][^=]+)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim()
                $config[$key] = $value
            }
        }
    } else {
        Write-Error "Configuration file not found: $ConfigPath"
        exit 1
    }
    return $config
}

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

# Function to export table to JSONL
function Export-TableToJsonl {
    param(
        [string]$TableName,
        [string]$Query,
        [string]$OutputFile,
        [hashtable]$Config
    )

    Write-Log "Exporting $TableName to $OutputFile"

    try {
        # Execute query and get results
        $result = docker exec infra-postgres-1 psql -U ml -d mathledger -t -c $Query --csv

        if ($LASTEXITCODE -ne 0) {
            throw "PostgreSQL query failed"
        }

        # Convert CSV to JSONL
        $jsonlContent = @()
        $lines = $result -split "`n" | Where-Object { $_.Trim() -ne "" }

        if ($lines.Count -gt 0) {
            # Get column headers from first line
            $headers = $lines[0] -split ","

            # Process data lines
            for ($i = 1; $i -lt $lines.Count; $i++) {
                $values = $lines[$i] -split ","
                $row = @{}

                for ($j = 0; $j -lt $headers.Count; $j++) {
                    $header = $headers[$j].Trim()
                    $value = $values[$j].Trim()

                    # Convert numeric values
                    if ($value -match '^\d+$') {
                        $row[$header] = [int]$value
                    } elseif ($value -match '^\d+\.\d+$') {
                        $row[$header] = [double]$value
                    } elseif ($value -eq 'true' -or $value -eq 'false') {
                        $row[$header] = [bool]$value
                    } else {
                        $row[$header] = $value
                    }
                }

                $jsonlContent += ($row | ConvertTo-Json -Compress)
            }
        }

        # Write to file
        $jsonlContent | Out-File -FilePath $OutputFile -Encoding UTF8
        Write-Log "Exported $($jsonlContent.Count) records to $OutputFile"

        return @{ success = $true; recordCount = $jsonlContent.Count }
    } catch {
        Write-Log "Failed to export $TableName : $_" "ERROR"
        return @{ success = $false; error = $_ }
    }
}

# Function to export statements table
function Export-Statements {
    param(
        [string]$OutputFile,
        [hashtable]$Config
    )

    $query = @"
SELECT
    id,
    theory_id,
    hash,
    content_norm,
    is_axiom,
    derivation_rule,
    derivation_depth,
    status,
    created_at
FROM statements
ORDER BY id;
"@

    return Export-TableToJsonl "statements" $query $OutputFile $Config
}

# Function to export proofs table
function Export-Proofs {
    param(
        [string]$OutputFile,
        [hashtable]$Config
    )

    $query = @"
SELECT
    id,
    statement_id,
    prover,
    success,
    time_ms,
    steps,
    created_at
FROM proofs
ORDER BY id;
"@

    return Export-TableToJsonl "proofs" $query $OutputFile $Config
}

# Function to export dependencies table
function Export-Dependencies {
    param(
        [string]$OutputFile,
        [hashtable]$Config
    )

    $query = @"
SELECT
    id,
    proof_id,
    used_statement_id,
    dependency_type,
    created_at
FROM dependencies
ORDER BY id;
"@

    return Export-TableToJsonl "dependencies" $query $OutputFile $Config
}

# Function to export blocks table
function Export-Blocks {
    param(
        [string]$OutputFile,
        [hashtable]$Config
    )

    $query = @"
SELECT
    id,
    run_id,
    root_hash,
    counts,
    created_at
FROM blocks
ORDER BY id;
"@

    return Export-TableToJsonl "blocks" $query $OutputFile $Config
}

# Function to export lemma cache
function Export-LemmaCache {
    param(
        [string]$OutputFile,
        [hashtable]$Config
    )

    $query = @"
SELECT
    lc.id,
    lc.statement_id,
    lc.usage_count,
    lc.created_at,
    s.content_norm
FROM lemma_cache lc
JOIN statements s ON lc.statement_id = s.id
ORDER BY lc.id;
"@

    return Export-TableToJsonl "lemma_cache" $query $OutputFile $Config
}

# Function to create export summary
function New-ExportSummary {
    param(
        [string]$OutputDir,
        [hashtable]$Results
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $summaryFile = Join-Path $OutputDir "export-summary-$((Get-Date).ToString('yyyyMMdd-HHmmss')).json"

    $summary = @{
        timestamp = $timestamp
        export_dir = $OutputDir
        tables = @{}
    }

    foreach ($table in $Results.Keys) {
        $summary.tables[$table] = @{
            success = $Results[$table].success
            record_count = $Results[$table].recordCount
            file = $Results[$table].file
        }
    }

    $summary | ConvertTo-Json -Depth 3 | Out-File -FilePath $summaryFile -Encoding UTF8
    Write-Log "Export summary written to: $summaryFile"

    return $summaryFile
}

# Main execution
try {
    # Load configuration
    Write-Log "Loading configuration from: $ConfigPath"
    $config = Load-Config $ConfigPath

    # Determine output directory
    if ([string]::IsNullOrEmpty($OutputDir)) {
        $OutputDir = $config.EXPORTS_DIR
    }

    # Create output directory if it doesn't exist
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
        Write-Log "Created output directory: $OutputDir"
    }

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Write-Log "Starting snapshot export to: $OutputDir"

    $results = @{}

    # Export statements
    $statementsFile = Join-Path $OutputDir "statements-$timestamp.jsonl"
    $results["statements"] = Export-Statements $statementsFile $config
    $results["statements"].file = $statementsFile

    # Export proofs
    $proofsFile = Join-Path $OutputDir "proofs-$timestamp.jsonl"
    $results["proofs"] = Export-Proofs $proofsFile $config
    $results["proofs"].file = $proofsFile

    # Export blocks
    $blocksFile = Join-Path $OutputDir "blocks-$timestamp.jsonl"
    $results["blocks"] = Export-Blocks $blocksFile $config
    $results["blocks"].file = $blocksFile

    # Export lemma cache
    $lemmasFile = Join-Path $OutputDir "lemmas-$timestamp.jsonl"
    $results["lemmas"] = Export-LemmaCache $lemmasFile $config
    $results["lemmas"].file = $lemmasFile

    # Export dependencies if requested
    if ($IncludeDependencies) {
        $depsFile = Join-Path $OutputDir "dependencies-$timestamp.jsonl"
        $results["dependencies"] = Export-Dependencies $depsFile $config
        $results["dependencies"].file = $depsFile
    }

    # Create export summary
    $summaryFile = New-ExportSummary $OutputDir $results

    # Report results
    Write-Log "=== Export Summary ==="
    foreach ($table in $results.Keys) {
        $result = $results[$table]
        if ($result.success) {
            Write-Log "$table`: $($result.recordCount) records exported to $($result.file)"
        } else {
            Write-Log "$table`: FAILED - $($result.error)" "ERROR"
        }
    }

    Write-Log "Snapshot export completed"
    Write-Log "Summary file: $summaryFile"

} catch {
    Write-Log "Fatal error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}
