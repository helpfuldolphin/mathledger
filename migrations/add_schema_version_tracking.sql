-- Migration: Add schema version tracking to statements table
-- Date: 2025-12-06
-- Author: Manus-A
-- Purpose: Enable versioned canonicalization and hash algorithm tracking

-- Add schema version columns to statements table
ALTER TABLE statements 
ADD COLUMN IF NOT EXISTS canon_schema_version VARCHAR(16) DEFAULT 'v1.0.0',
ADD COLUMN IF NOT EXISTS hash_algorithm VARCHAR(32) DEFAULT 'sha256-domain-sep-v1',
ADD COLUMN IF NOT EXISTS json_canon_version VARCHAR(16) DEFAULT 'rfc8785-v1';

-- Add indexes for version-based queries
CREATE INDEX IF NOT EXISTS idx_statements_canon_schema_version 
ON statements(canon_schema_version);

CREATE INDEX IF NOT EXISTS idx_statements_hash_algorithm 
ON statements(hash_algorithm);

-- Add version tracking to blocks table
ALTER TABLE blocks
ADD COLUMN IF NOT EXISTS attestation_schema_version VARCHAR(16) DEFAULT 'v2.0.0',
ADD COLUMN IF NOT EXISTS merkle_schema_version VARCHAR(16) DEFAULT 'v1.0.0';

-- Create schema_versions metadata table
CREATE TABLE IF NOT EXISTS schema_versions (
    id SERIAL PRIMARY KEY,
    component VARCHAR(64) NOT NULL,
    version VARCHAR(16) NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    UNIQUE(component, version)
);

-- Insert initial schema versions
INSERT INTO schema_versions (component, version, description) VALUES
('canon', 'v1.0.0', 'String-based canonicalization (normalization/canon.py)'),
('ast_canon', 'v1.0.0', 'AST-based canonicalization (normalization/ast_canon.py)'),
('hash', 'sha256-domain-sep-v1', 'SHA-256 with domain separation (backend/crypto/hashing.py)'),
('json_canon', 'rfc8785-v1', 'RFC 8785 JSON canonicalization (backend/basis/canon.py)'),
('merkle', 'v1.0.0', 'Merkle tree construction with sorted leaves'),
('attestation', 'v2.0.0', 'Dual-root attestation (H_t = SHA256(R_t || U_t))')
ON CONFLICT (component, version) DO NOTHING;

-- Add comments for documentation
COMMENT ON COLUMN statements.canon_schema_version IS 'Version of canonicalization algorithm used';
COMMENT ON COLUMN statements.hash_algorithm IS 'Version of hash algorithm used';
COMMENT ON COLUMN statements.json_canon_version IS 'Version of JSON canonicalization used';
COMMENT ON COLUMN blocks.attestation_schema_version IS 'Version of attestation algorithm used';
COMMENT ON COLUMN blocks.merkle_schema_version IS 'Version of Merkle tree construction used';
COMMENT ON TABLE schema_versions IS 'Tracks all schema version changes for audit trail';

-- Verification query: Check that all statements have version metadata
-- SELECT 
--     canon_schema_version,
--     hash_algorithm,
--     json_canon_version,
--     COUNT(*) as statement_count
-- FROM statements
-- GROUP BY canon_schema_version, hash_algorithm, json_canon_version;
