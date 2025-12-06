# Database Seeding Script
# Inserts initial "Propositional" theory and K and S axioms

$ErrorActionPreference = "Stop"

Write-Host "Seeding database with initial data..." -ForegroundColor Cyan

# Insert the Propositional theory into theories table
Write-Host "Inserting Propositional theory..." -ForegroundColor Yellow
docker exec -i infra-postgres-1 psql -U ml -d mathledger -c "INSERT INTO theories (name, slug, version, logic) VALUES ('Propositional', 'pl', 'v0', 'classical') ON CONFLICT (name) DO NOTHING;"

# Get the theory ID for Propositional logic
Write-Host "Getting theory ID..." -ForegroundColor Yellow
$theoryId = docker exec -i infra-postgres-1 psql -U ml -d mathledger -t -c "SELECT id FROM theories WHERE name = 'Propositional' LIMIT 1;"
$theoryId = $theoryId.Trim()

Write-Host "Theory ID: $theoryId" -ForegroundColor Cyan

# Insert K axiom: P → (Q → P)
Write-Host "Inserting K axiom..." -ForegroundColor Yellow
docker exec -i infra-postgres-1 psql -U ml -d mathledger -c @"
INSERT INTO statements (content_norm, text, is_axiom, derivation_depth, system_id)
VALUES ('P → (Q → P)', 'P → (Q → P)', true, 0, 'theory123');
"@

# Insert S axiom: (P → (Q → R)) → ((P → Q) → (P → R))
Write-Host "Inserting S axiom..." -ForegroundColor Yellow
docker exec -i infra-postgres-1 psql -U ml -d mathledger -c @"
INSERT INTO statements (content_norm, text, is_axiom, derivation_depth, system_id)
VALUES ('(P → (Q → R)) → ((P → Q) → (P → R))', '(P → (Q → R)) → ((P → Q) → (P → R))', true, 0, 'theory123');
"@

Write-Host "Database seeding completed!" -ForegroundColor Green
