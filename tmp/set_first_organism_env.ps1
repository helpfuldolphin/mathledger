Set-Location 'C:\dev\mathledger'
Get-Content 'config/first_organism.env' | Where-Object {  -match '^\w+=' } | ForEach-Object {
     =  -split '=', 2
    [Environment]::SetEnvironmentVariable([0], [1], 'Process')
}
uv run python scripts/run-migrations.py
uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop -v -s
