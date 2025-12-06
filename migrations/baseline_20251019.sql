-- migrations/baseline_20251019.sql
-- MathLedger Baseline Schema Migration
-- Consolidates migrations 001-014 into single authoritative schema
-- Generated: 2025-10-19 by Manus G - Systems Mechanic
-- 
-- This migration supersedes all previous migrations and provides a clean,
-- idempotent foundation for the MathLedger database schema.
--
-- Design Principles:
-- 1. Idempotent: Safe to run multiple times
-- 2. Postgres 15 Compatible: Uses DO blocks for conditional constraints
-- 3. Type Consistent: hash as TEXT (hex-encoded), system_id throughout
-- 4. Backend Aligned: Matches actual backend code expectations
-- 5. Defensive: Checks for existence before creating/altering

-- ============================================================================
-- 0. MIGRATION TRACKING
-- ============================================================================

-- Create migration tracking table to record applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    description TEXT,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Record this baseline migration
INSERT INTO schema_migrations (version, description)
VALUES ('baseline_20251019', 'Consolidated baseline schema from migrations 001-014')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- 1. EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For UUID generation
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- For fuzzy text search

-- ============================================================================
-- 2. THEORIES (SYSTEMS)
-- ============================================================================

CREATE TABLE IF NOT EXISTS theories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    slug TEXT UNIQUE,
    version TEXT DEFAULT 'v0',
    logic TEXT DEFAULT 'unspecified',
    parent_id UUID REFERENCES theories(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure slug column exists (for compatibility)
ALTER TABLE theories ADD COLUMN IF NOT EXISTS slug TEXT;
ALTER TABLE theories ADD COLUMN IF NOT EXISTS parent_id UUID REFERENCES theories(id) ON DELETE SET NULL;

-- Create indexes
CREATE INDEX IF NOT EXISTS theories_slug_idx ON theories(slug);
CREATE INDEX IF NOT EXISTS theories_name_idx ON theories(name);
CREATE INDEX IF NOT EXISTS theories_parent_id_idx ON theories(parent_id);

-- Add unique constraint on slug using DO block (Postgres 15 compatible)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'theories_slug_unique') THEN
        ALTER TABLE theories ADD CONSTRAINT theories_slug_unique UNIQUE (slug);
    END IF;
END $$;

-- ============================================================================
-- 3. SYMBOLS
-- ============================================================================

CREATE TABLE IF NOT EXISTS symbols (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    theory_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    arity INT NOT NULL CHECK (arity >= 0),
    sort TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add unique constraint
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'symbols_theory_name_arity_unique') THEN
        ALTER TABLE symbols ADD CONSTRAINT symbols_theory_name_arity_unique UNIQUE (theory_id, name, arity);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS symbols_theory_id_idx ON symbols(theory_id);

-- ============================================================================
-- 4. STATEMENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS statements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE,
    hash TEXT NOT NULL UNIQUE,  -- SHA-256 hex-encoded (not bytea)
    sort TEXT,
    content_norm TEXT NOT NULL,  -- Normalized s-expr / canonical form
    content TEXT,                -- Alternative content representation
    content_lean TEXT,           -- Lean source (goal)
    content_latex TEXT,          -- LaTeX representation
    normalized_text TEXT,        -- ML-normalized text (ASCII operators)
    status TEXT NOT NULL CHECK (status IN ('proven','disproven','open','unknown')) DEFAULT 'unknown',
    truth_domain TEXT,           -- e.g., "classical", "finite-model(n=5)"
    derivation_rule TEXT,        -- Rule used to derive this statement
    derivation_depth INT,        -- Depth in derivation tree
    is_axiom BOOLEAN DEFAULT FALSE,
    fol_type TEXT,               -- FOL type (axiom, theorem, lemma, definition)
    variables TEXT[],            -- Array of variables in the statement
    predicates TEXT[],           -- Array of predicates used
    functions TEXT[],            -- Array of functions used
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure all columns exist (for compatibility with existing code)
ALTER TABLE statements ADD COLUMN IF NOT EXISTS system_id UUID REFERENCES theories(id) ON DELETE CASCADE;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS content TEXT;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS normalized_text TEXT;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS derivation_rule TEXT;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS derivation_depth INT;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS is_axiom BOOLEAN DEFAULT FALSE;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS fol_type TEXT;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS variables TEXT[];
ALTER TABLE statements ADD COLUMN IF NOT EXISTS predicates TEXT[];
ALTER TABLE statements ADD COLUMN IF NOT EXISTS functions TEXT[];

-- Create indexes
CREATE INDEX IF NOT EXISTS statements_system_id_idx ON statements(system_id);
CREATE INDEX IF NOT EXISTS statements_status_idx ON statements(status);
CREATE INDEX IF NOT EXISTS statements_derivation_depth_idx ON statements(derivation_depth);
CREATE INDEX IF NOT EXISTS statements_created_at_idx ON statements(created_at);
CREATE INDEX IF NOT EXISTS statements_hash_idx ON statements(hash);
CREATE INDEX IF NOT EXISTS statements_content_norm_idx ON statements(content_norm);
CREATE INDEX IF NOT EXISTS statements_is_axiom_idx ON statements(is_axiom);

-- Add check constraint for derivation_depth
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'statements_derivation_depth_positive') THEN
        ALTER TABLE statements ADD CONSTRAINT statements_derivation_depth_positive
            CHECK (derivation_depth IS NULL OR derivation_depth >= 0);
    END IF;
END $$;

-- ============================================================================
-- 5. PROOFS
-- ============================================================================

CREATE TABLE IF NOT EXISTS proofs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    statement_id UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    system_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE,
    prover TEXT NOT NULL,           -- e.g., 'lean4', 'z3', 'vampire'
    method TEXT,                    -- Proof method/strategy
    status TEXT NOT NULL CHECK (status IN ('success', 'failure', 'timeout', 'unknown')) DEFAULT 'unknown',
    proof_term TEXT,                -- Optional serialized proof term / script
    time_ms INT CHECK (time_ms >= 0),
    duration_ms INT CHECK (duration_ms >= 0),  -- Alias for time_ms
    steps INT CHECK (steps >= 0),
    success BOOLEAN NOT NULL DEFAULT FALSE,
    proof_hash TEXT,                -- Hash of proof term (TEXT, not bytea)
    kernel_version TEXT,            -- Lean/mathlib commit for reproducibility
    derivation_rule TEXT,           -- Rule used in this proof
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure all columns exist
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS system_id UUID REFERENCES theories(id) ON DELETE CASCADE;
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS method TEXT;
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'unknown';
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS duration_ms INT;
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS derivation_rule TEXT;

-- Create indexes
CREATE INDEX IF NOT EXISTS proofs_statement_id_idx ON proofs(statement_id);
CREATE INDEX IF NOT EXISTS proofs_system_id_idx ON proofs(system_id);
CREATE INDEX IF NOT EXISTS proofs_prover_idx ON proofs(prover);
CREATE INDEX IF NOT EXISTS proofs_success_idx ON proofs(success);
CREATE INDEX IF NOT EXISTS proofs_status_idx ON proofs(status);
CREATE INDEX IF NOT EXISTS proofs_created_at_idx ON proofs(created_at);

-- Add check constraints
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'proofs_time_ms_positive') THEN
        ALTER TABLE proofs ADD CONSTRAINT proofs_time_ms_positive
            CHECK (time_ms IS NULL OR time_ms >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'proofs_steps_positive') THEN
        ALTER TABLE proofs ADD CONSTRAINT proofs_steps_positive
            CHECK (steps IS NULL OR steps >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'proofs_status_check') THEN
        ALTER TABLE proofs ADD CONSTRAINT proofs_status_check
            CHECK (status IN ('success', 'failure', 'timeout', 'unknown'));
    END IF;
END $$;

-- ============================================================================
-- 6. DEPENDENCIES
-- ============================================================================

CREATE TABLE IF NOT EXISTS dependencies (
    proof_id UUID NOT NULL REFERENCES proofs(id) ON DELETE CASCADE,
    used_statement_id UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    PRIMARY KEY (proof_id, used_statement_id)
);

CREATE INDEX IF NOT EXISTS dependencies_proof_id_idx ON dependencies(proof_id);
CREATE INDEX IF NOT EXISTS dependencies_used_statement_id_idx ON dependencies(used_statement_id);

-- ============================================================================
-- 7. RUNS
-- ============================================================================

CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    system_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')) DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure columns exist
ALTER TABLE runs ADD COLUMN IF NOT EXISTS system_id UUID REFERENCES theories(id) ON DELETE CASCADE;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- Create indexes
CREATE INDEX IF NOT EXISTS runs_system_id_idx ON runs(system_id);
CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status);
CREATE INDEX IF NOT EXISTS runs_created_at_idx ON runs(created_at);

-- ============================================================================
-- 8. BLOCKS
-- ============================================================================

CREATE TABLE IF NOT EXISTS blocks (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    system_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE,
    block_number BIGINT NOT NULL,       -- Sequential block number within system
    prev_hash TEXT,                     -- Hash of previous block (for chaining)
    root_hash TEXT NOT NULL,            -- Merkle root of statements in this block
    merkle_root TEXT,                   -- Alias for root_hash (compatibility)
    header JSONB NOT NULL DEFAULT '{}', -- Block metadata (counts, timestamps, etc.)
    statements JSONB NOT NULL DEFAULT '[]', -- Array of statement hashes in this block
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure columns exist
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS merkle_root TEXT;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS header JSONB DEFAULT '{}';
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS statements JSONB DEFAULT '[]';

-- Create indexes
CREATE INDEX IF NOT EXISTS blocks_run_id_idx ON blocks(run_id);
CREATE INDEX IF NOT EXISTS blocks_system_id_idx ON blocks(system_id);
CREATE INDEX IF NOT EXISTS blocks_block_number_idx ON blocks(block_number);
CREATE INDEX IF NOT EXISTS blocks_created_at_idx ON blocks(created_at);

-- Add unique constraint on (system_id, block_number)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'blocks_system_block_number_unique') THEN
        ALTER TABLE blocks ADD CONSTRAINT blocks_system_block_number_unique UNIQUE (system_id, block_number);
    END IF;
END $$;

-- ============================================================================
-- 9. LEMMA CACHE
-- ============================================================================

CREATE TABLE IF NOT EXISTS lemma_cache (
    id BIGSERIAL PRIMARY KEY,
    statement_id UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    usage_count INT NOT NULL DEFAULT 0,
    last_used TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add unique constraint
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'lemma_cache_statement_id_unique') THEN
        ALTER TABLE lemma_cache ADD CONSTRAINT lemma_cache_statement_id_unique UNIQUE (statement_id);
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS lemma_cache_usage_count_idx ON lemma_cache(usage_count DESC);
CREATE INDEX IF NOT EXISTS lemma_cache_statement_id_idx ON lemma_cache(statement_id);
CREATE INDEX IF NOT EXISTS lemma_cache_last_used_idx ON lemma_cache(last_used);

-- Add check constraint
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'lemma_cache_usage_count_positive') THEN
        ALTER TABLE lemma_cache ADD CONSTRAINT lemma_cache_usage_count_positive
            CHECK (usage_count >= 0);
    END IF;
END $$;

-- ============================================================================
-- 10. AXIOMS
-- ============================================================================

CREATE TABLE IF NOT EXISTS axioms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    theory_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE,
    statement_id UUID REFERENCES statements(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS axioms_theory_id_idx ON axioms(theory_id);
CREATE INDEX IF NOT EXISTS axioms_statement_id_idx ON axioms(statement_id);

-- ============================================================================
-- 11. POLICY SETTINGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS policy_settings (
    id SERIAL PRIMARY KEY,
    key TEXT UNIQUE,
    value TEXT,
    policy_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS policy_settings_key_idx ON policy_settings(key);

-- ============================================================================
-- 12. PROOF PARENTS (for derivation tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS proof_parents (
    id SERIAL PRIMARY KEY,
    child_hash TEXT NOT NULL,
    parent_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS proof_parents_child_hash_idx ON proof_parents(child_hash);
CREATE INDEX IF NOT EXISTS proof_parents_parent_hash_idx ON proof_parents(parent_hash);

-- ============================================================================
-- 13. DERIVED STATEMENTS (for tracking derivations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS derived_statements (
    id SERIAL PRIMARY KEY,
    statement_id UUID REFERENCES statements(id) ON DELETE CASCADE,
    statement_text TEXT NOT NULL,
    derivation_rule TEXT,
    derivation_depth INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS derived_statements_statement_id_idx ON derived_statements(statement_id);

-- ============================================================================
-- 14. SEED DATA
-- ============================================================================

-- Insert default theories if missing
INSERT INTO theories (name, slug, version, logic) VALUES
    ('Propositional', 'pl', 'v0', 'classical'),
    ('First Order Logic', 'fol', 'v0', 'classical')
ON CONFLICT (name) DO NOTHING;

-- Insert default run for existing data (if no runs exist)
DO $$
DECLARE
    pl_theory_id UUID;
BEGIN
    -- Get the Propositional theory ID
    SELECT id INTO pl_theory_id FROM theories WHERE slug = 'pl' LIMIT 1;
    
    IF pl_theory_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM runs LIMIT 1) THEN
        INSERT INTO runs (name, system_id, status, completed_at)
        VALUES ('initial_run', pl_theory_id, 'completed', NOW());
    END IF;
END $$;

-- ============================================================================
-- 15. MIGRATION COMPLETE
-- ============================================================================

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Baseline migration baseline_20251019 completed successfully';
    RAISE NOTICE 'Schema version: 2025-10-19';
    RAISE NOTICE 'Tables created: theories, symbols, statements, proofs, dependencies, runs, blocks, lemma_cache, axioms, policy_settings, proof_parents, derived_statements';
END $$;

