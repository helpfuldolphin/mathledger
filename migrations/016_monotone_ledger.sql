-- migrations/016_monotone_ledger.sql
-- Cursor D - Ledger Architect
-- Rebuild ledger schema for monotone, dual-attested block sealing.
-- Ensures deterministic UPSERT semantics, cryptographic integrity, and
-- block metadata necessary for dual attestation and sorted Merkle commits.

-- This migration is intentionally idempotent. Re-running it is safe.

BEGIN;

-- ============================================================================
-- 0. EXTENSIONS & UTILITIES
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Helper to detect column data types
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'statements'
          AND column_name = 'hash'
          AND data_type = 'bytea'
    ) THEN
        ALTER TABLE statements
        ALTER COLUMN hash TYPE TEXT
        USING encode(hash, 'hex');
    END IF;
END $$;

ALTER TABLE statements
    ALTER COLUMN hash SET NOT NULL;

-- Normalize statement hashes to hex SHA256 if missing or malformed
UPDATE statements
SET hash = encode(
        digest(
            convert_to(COALESCE(content_norm, ''), 'utf8'),
            'sha256'
        ),
        'hex'
    )
WHERE hash IS NULL
   OR length(hash) <> 64
   OR hash !~ '^[0-9a-f]{64}$';

-- Ensure statement/system uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS statements_hash_unique
    ON statements (hash);

CREATE UNIQUE INDEX IF NOT EXISTS statements_system_hash_unique
    ON statements (system_id, hash);

CREATE INDEX IF NOT EXISTS statements_system_created_idx
    ON statements (system_id, created_at DESC);

ALTER TABLE statements
    ADD COLUMN IF NOT EXISTS normalized_text TEXT;

ALTER TABLE statements
    ADD COLUMN IF NOT EXISTS truth_domain TEXT;

ALTER TABLE statements
    ADD COLUMN IF NOT EXISTS is_axiom BOOLEAN DEFAULT FALSE;

-- ============================================================================
-- 1. PROOFS TABLE HARDENING
-- ============================================================================

ALTER TABLE proofs
    ADD COLUMN IF NOT EXISTS method TEXT,
    ADD COLUMN IF NOT EXISTS proof_text TEXT,
    ADD COLUMN IF NOT EXISTS module_name TEXT,
    ADD COLUMN IF NOT EXISTS stdout TEXT,
    ADD COLUMN IF NOT EXISTS stderr TEXT,
    ADD COLUMN IF NOT EXISTS attestation_root TEXT,
    ADD COLUMN IF NOT EXISTS duration_ms INT,
    ADD COLUMN IF NOT EXISTS proof_hash TEXT;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'proofs'
          AND column_name = 'proof_hash'
          AND data_type = 'bytea'
    ) THEN
        ALTER TABLE proofs
        ALTER COLUMN proof_hash TYPE TEXT
        USING encode(proof_hash, 'hex');
    END IF;
END $$;

-- Backfill proof hashes deterministically from payloads
UPDATE proofs
SET proof_hash = encode(
        digest(
            convert_to(
                COALESCE(proof_term, '') ||
                COALESCE(proof_text, '') ||
                COALESCE(prover, '') ||
                COALESCE(status, ''),
                'utf8'
            ),
            'sha256'
        ),
        'hex'
    )
WHERE proof_hash IS NULL
   OR length(proof_hash) <> 64
   OR proof_hash !~ '^[0-9a-f]{64}$';

ALTER TABLE proofs
    ALTER COLUMN proof_hash SET NOT NULL;

-- Harden status domain
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'proofs_status_check'
    ) THEN
        ALTER TABLE proofs DROP CONSTRAINT proofs_status_check;
    END IF;
END $$;

ALTER TABLE proofs
    ADD CONSTRAINT proofs_status_check
    CHECK (status IN ('success', 'failure', 'timeout', 'unknown'));

ALTER TABLE proofs
    ADD CONSTRAINT proofs_duration_positive
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

CREATE UNIQUE INDEX IF NOT EXISTS proofs_statement_prover_hash_unique
    ON proofs (statement_id, prover, proof_hash);

CREATE INDEX IF NOT EXISTS proofs_system_created_idx
    ON proofs (system_id, created_at DESC);

-- ============================================================================
-- 2. LEDGER SEQUENCING STATE
-- ============================================================================

CREATE TABLE IF NOT EXISTS ledger_sequences (
    system_id          UUID PRIMARY KEY REFERENCES theories(id) ON DELETE CASCADE,
    run_id             BIGINT REFERENCES runs(id) ON DELETE SET NULL,
    height             BIGINT NOT NULL DEFAULT 0,
    prev_block_id      BIGINT REFERENCES blocks(id) ON DELETE SET NULL,
    prev_block_hash    TEXT,
    prev_composite_root TEXT,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ledger_sequences_prev_block_idx
    ON ledger_sequences (prev_block_id);

-- Backfill sequence state from existing blocks
INSERT INTO ledger_sequences (system_id, run_id, height, prev_block_id, prev_block_hash, prev_composite_root)
SELECT DISTINCT ON (b.system_id)
       b.system_id,
       b.run_id,
       COALESCE(b.block_number, 0),
       b.id,
       b.root_hash,
       COALESCE(b.composite_attestation_root, b.root_hash)
FROM blocks b
ORDER BY b.system_id, b.block_number DESC
ON CONFLICT (system_id) DO UPDATE
SET height = EXCLUDED.height,
    prev_block_id = EXCLUDED.prev_block_id,
    prev_block_hash = EXCLUDED.prev_block_hash,
    prev_composite_root = EXCLUDED.prev_composite_root,
    updated_at = NOW(),
    run_id = COALESCE(ledger_sequences.run_id, EXCLUDED.run_id);

-- ============================================================================
-- 3. BLOCK TABLE AUGMENTATION
-- ============================================================================

ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS sealed_at TIMESTAMPTZ DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS sealed_by TEXT DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS statement_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS proof_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS reasoning_merkle_root TEXT,
    ADD COLUMN IF NOT EXISTS ui_merkle_root TEXT,
    ADD COLUMN IF NOT EXISTS composite_attestation_root TEXT,
    ADD COLUMN IF NOT EXISTS attestation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS payload_hash TEXT,
    ADD COLUMN IF NOT EXISTS block_hash TEXT,
    ADD COLUMN IF NOT EXISTS prev_block_id BIGINT REFERENCES blocks(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS canonical_statements JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS canonical_proofs JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Backfill prev_block_id and hashes using window functions
WITH ordered AS (
    SELECT
        id,
        system_id,
        block_number,
        LAG(id) OVER (PARTITION BY system_id ORDER BY block_number) AS prev_id,
        LAG(root_hash) OVER (PARTITION BY system_id ORDER BY block_number) AS prev_root
    FROM blocks
)
UPDATE blocks b
SET prev_block_id = o.prev_id,
    prev_hash = COALESCE(b.prev_hash, o.prev_root)
FROM ordered o
WHERE o.id = b.id
  AND (b.prev_block_id IS DISTINCT FROM o.prev_id OR b.prev_hash IS NULL);

-- Fill attestation roots and counts
UPDATE blocks
SET statement_count = COALESCE(jsonb_array_length(statements), 0)
WHERE statement_count IS DISTINCT FROM COALESCE(jsonb_array_length(statements), 0);

UPDATE blocks
SET proof_count = COALESCE(proof_count, 0)
WHERE proof_count IS NULL OR proof_count < 0;

UPDATE blocks
SET reasoning_merkle_root = root_hash
WHERE reasoning_merkle_root IS NULL;

UPDATE blocks
SET ui_merkle_root = CASE
        WHEN ui_merkle_root IS NULL THEN root_hash
        ELSE ui_merkle_root
    END;

UPDATE blocks
SET composite_attestation_root = CASE
        WHEN composite_attestation_root IS NULL THEN root_hash
        ELSE composite_attestation_root
    END;

UPDATE blocks
SET payload_hash = COALESCE(payload_hash, root_hash),
    block_hash = COALESCE(block_hash, root_hash),
    canonical_statements = CASE
        WHEN canonical_statements = '[]'::jsonb AND statements IS NOT NULL THEN statements
        ELSE canonical_statements
    END
WHERE payload_hash IS NULL OR block_hash IS NULL OR canonical_statements = '[]'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS blocks_block_hash_unique
    ON blocks (block_hash);

CREATE UNIQUE INDEX IF NOT EXISTS blocks_composite_root_unique
    ON blocks (composite_attestation_root);

CREATE INDEX IF NOT EXISTS blocks_prev_block_idx
    ON blocks (prev_block_id);

-- ============================================================================
-- 4. NORMALIZED BLOCK PAYLOAD TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS block_statements (
    block_id      BIGINT NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    position      INT NOT NULL,
    statement_id  UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    statement_hash TEXT NOT NULL,
    PRIMARY KEY (block_id, position)
);

CREATE UNIQUE INDEX IF NOT EXISTS block_statements_unique
    ON block_statements (block_id, statement_id);

CREATE TABLE IF NOT EXISTS block_proofs (
    block_id     BIGINT NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    position     INT NOT NULL,
    proof_id     UUID NOT NULL REFERENCES proofs(id) ON DELETE CASCADE,
    proof_hash   TEXT NOT NULL,
    statement_id UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    PRIMARY KEY (block_id, position)
);

CREATE UNIQUE INDEX IF NOT EXISTS block_proofs_unique
    ON block_proofs (block_id, proof_id);

-- ============================================================================
-- 5. MAINTENANCE TRIGGERS (UPDATED TIMESTAMPS)
-- ============================================================================

CREATE OR REPLACE FUNCTION touch_ledger_sequences() RETURNS TRIGGER AS $$
BEGIN
    UPDATE ledger_sequences
    SET updated_at = NOW()
    WHERE system_id = NEW.system_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS blocks_touch_sequence ON blocks;

CREATE TRIGGER blocks_touch_sequence
AFTER INSERT OR UPDATE ON blocks
FOR EACH ROW
EXECUTE FUNCTION touch_ledger_sequences();

COMMIT;

