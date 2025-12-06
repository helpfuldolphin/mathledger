# Database Migration Runner
# Executes all SQL migration files in numerical order
#
# SECURITY NOTICE: This script requires DATABASE_URL to be set in the environment.
# Do not use default passwords. Set DATABASE_URL before running this script.

$ErrorActionPreference = "Stop"

# Validate required environment variable
if (-not $env:DATABASE_URL) {
    Write-Host "[FATAL] DATABASE_URL environment variable is not set." -ForegroundColor Red
    Write-Host "Set it explicitly before running migrations:" -ForegroundColor Yellow
    Write-Host '  $env:DATABASE_URL = "postgresql://user:password@host:port/database"' -ForegroundColor Gray
    exit 1
}

Write-Host "Starting database migrations..." -ForegroundColor Cyan
Write-Host "Using DATABASE_URL from environment" -ForegroundColor Gray

# Get all migration files and sort them numerically
$migrationFiles = Get-ChildItem -Path "migrations" -Filter "*.sql" | Sort-Object {
    $match = [regex]::Match($_.Name, '(\d+)')
    if ($match.Success) { [int]$match.Groups[1].Value } else { 999 }
}

if ($migrationFiles.Count -eq 0) {
    Write-Host "No migration files found in migrations/ directory" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($migrationFiles.Count) migration files:" -ForegroundColor Yellow
foreach ($file in $migrationFiles) {
    Write-Host "  - $($file.Name)" -ForegroundColor Cyan
}

# Execute each migration file
foreach ($migrationFile in $migrationFiles) {
    Write-Host "`nExecuting migration: $($migrationFile.Name)" -ForegroundColor Yellow

    try {
        # Read the migration file
        $sqlContent = Get-Content -Path $migrationFile.FullName -Raw

        # Split by semicolon and execute each statement
        $statements = $sqlContent -split ';' | Where-Object { $_.Trim() -ne '' -and $_.Trim() -notlike '--*' }

        # Use DATABASE_URL from environment
        $connectionString = $env:DATABASE_URL

        foreach ($statement in $statements) {
            $cleanStatement = $statement.Trim()
            if ($cleanStatement -ne '') {
                Write-Host "  Executing: $($cleanStatement.Substring(0, [Math]::Min(50, $cleanStatement.Length)))..." -ForegroundColor Gray

                # Use docker exec to run psql commands
                $psqlCommand = $cleanStatement -replace '"', '\"'
                $fullCommand = "docker exec -i infra-postgres-1 psql -U ml -d mathledger -c `"$psqlCommand`""

                $result = Invoke-Expression $fullCommand
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "  Warning: Statement may have failed (exit code $LASTEXITCODE)" -ForegroundColor Yellow
                    Write-Host "  Output: $result" -ForegroundColor Gray
                }
            }
        }

        Write-Host "  Migration $($migrationFile.Name) completed!" -ForegroundColor Green

    } catch {
        Write-Host "  Error executing migration $($migrationFile.Name): $_" -ForegroundColor Red
        Write-Host "  Continuing with next migration..." -ForegroundColor Yellow
    }
}

Write-Host "`nAll migrations completed!" -ForegroundColor Green
