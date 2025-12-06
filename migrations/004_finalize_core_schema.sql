-- migrations/004_finalize_core_schema.sql
-- Finalize canonical database schema for MathLedger
-- This migration consolidates all table structures into a single, authoritative format

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- 1. THEORIES AND SYMBOLS (canonical)
-- ============================================================================

-- Theories table (already exists, ensure it has all required fields)
CREATE TABLE IF NOT EXISTS theories (
  id          uuid primary key default gen_random_uuid(),
  name        text not null unique,
  slug        text unique,  -- Added in 003, ensure it exists
  version     text default 'v0',
  logic       text default 'unspecified',
  created_at  timestamptz not null default now()
);

-- Add slug if missing
ALTER TABLE theories ADD COLUMN IF NOT EXISTS slug text;
CREATE UNIQUE INDEX IF NOT EXISTS theories_slug_idx ON theories(slug);

-- Symbols table (already exists)
CREATE TABLE IF NOT EXISTS symbols (
  id          uuid primary key default gen_random_uuid(),
  theory_id   uuid not null references theories(id) on delete cascade,
  name        text not null,
  arity       int  not null check (arity >= 0),
  sort        text,
  unique(theory_id, name, arity)
);

-- ============================================================================
-- 2. STATEMENTS (canonical with system_id)
-- ============================================================================

-- Statements table with both theory_id and system_id for compatibility
CREATE TABLE IF NOT EXISTS statements (
  id              uuid primary key default gen_random_uuid(),
  theory_id       uuid not null references theories(id) on delete cascade,
  system_id       uuid not null references theories(id) on delete cascade,
  hash            bytea not null unique,  -- SHA-256 of normalized form
  sort            text,
  content_norm    text not null,          -- normalized s-expr / canonical form
  content_lean    text,                   -- Lean source (goal)
  content_latex   text,
  status          text not null check (status in ('proven','disproven','open','unknown')) default 'unknown',
  truth_domain    text,                   -- e.g., "classical", "finite-model(n=5)", etc.
  derivation_rule text,                   -- Rule used to derive this statement
  derivation_depth int,                   -- Depth in derivation tree
  created_at      timestamptz not null default now()
);

-- Ensure system_id exists and is populated
ALTER TABLE statements ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
ALTER TABLE statements ADD COLUMN IF NOT EXISTS derivation_rule text;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS derivation_depth int;

-- Backfill system_id from theory_id if missing
UPDATE statements SET system_id = theory_id WHERE system_id IS NULL AND theory_id IS NOT NULL;

-- Make system_id NOT NULL after backfilling (only if all rows have values)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM statements WHERE system_id IS NULL) THEN
        BEGIN
            ALTER TABLE statements ALTER COLUMN system_id SET NOT NULL;
        EXCEPTION WHEN OTHERS THEN
            -- Ignore if already NOT NULL or other constraint issues
            NULL;
        END;
    END IF;
END $$;

-- ============================================================================
-- 3. PROOFS (canonical with system_id)
-- ============================================================================

CREATE TABLE IF NOT EXISTS proofs (
  id              uuid primary key default gen_random_uuid(),
  statement_id    uuid not null references statements(id) on delete cascade,
  system_id       uuid not null references theories(id) on delete cascade,
  prover          text not null,          -- e.g., 'lean4'
  method          text,                   -- proof method/strategy
  proof_term      text,                   -- optional serialized proof term / script
  time_ms         int  check (time_ms >= 0),
  steps           int  check (steps >= 0),
  success         boolean not null default false,
  proof_hash      bytea,                  -- optional hash of proof term
  kernel_version  text,                   -- Lean/mathlib commit for reproducibility
  created_at      timestamptz not null default now()
);

-- Ensure system_id exists and is populated
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS method text;

-- Backfill system_id from statement's system_id
UPDATE proofs SET system_id = s.system_id
FROM statements s
WHERE proofs.statement_id = s.id AND proofs.system_id IS NULL;

-- Make system_id NOT NULL after backfilling (only if all rows have values)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM proofs WHERE system_id IS NULL) THEN
        BEGIN
            ALTER TABLE proofs ALTER COLUMN system_id SET NOT NULL;
        EXCEPTION WHEN OTHERS THEN
            -- Ignore if already NOT NULL or other constraint issues
            NULL;
        END;
    END IF;
END $$;

-- ============================================================================
-- 4. DEPENDENCIES (canonical)
-- ============================================================================

CREATE TABLE IF NOT EXISTS dependencies (
  proof_id            uuid not null references proofs(id) on delete cascade,
  used_statement_id   uuid not null references statements(id) on delete cascade,
  primary key (proof_id, used_statement_id)
);

-- ============================================================================
-- 5. RUNS (canonical with system_id)
-- ============================================================================

CREATE TABLE IF NOT EXISTS runs (
  id          bigserial primary key,
  name        text,
  system_id   uuid not null references theories(id),
  status      text not null check (status in ('running', 'completed', 'failed')) default 'running',
  started_at  timestamptz not null default now(),
  completed_at timestamptz,
  created_at  timestamptz not null default now()
);

-- Ensure system_id exists
ALTER TABLE runs ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);

-- Backfill system_id for existing runs (default to first theory)
UPDATE runs SET system_id = (SELECT id FROM theories LIMIT 1) WHERE system_id IS NULL;

-- Make system_id NOT NULL after backfilling (only if all rows have values)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM runs WHERE system_id IS NULL) THEN
        BEGIN
            ALTER TABLE runs ALTER COLUMN system_id SET NOT NULL;
        EXCEPTION WHEN OTHERS THEN
            -- Ignore if already NOT NULL or other constraint issues
            NULL;
        END;
    END IF;
END $$;

-- ============================================================================
-- 6. BLOCKS (canonical - using runs-based approach)
-- ============================================================================

-- Create canonical blocks table (idempotent - only create if doesn't exist)
-- Note: We don't drop existing blocks table to preserve data
-- Instead, we ensure all required columns exist
CREATE TABLE IF NOT EXISTS blocks (
  id          bigserial primary key,
  run_id      bigint not null references runs(id) on delete cascade,
  system_id   uuid not null references theories(id),
  block_number bigint not null,  -- Sequential block number within system
  prev_hash   text,              -- Hash of previous block (for chaining)
  root_hash   text not null,     -- Merkle root of statements in this block
  header      jsonb not null,    -- Block metadata (counts, timestamps, etc.)
  statements  jsonb not null,    -- Array of statement hashes in this block
  created_at  timestamptz not null default now()
);

-- Ensure all required columns exist in blocks (additive only)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'blocks') THEN
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS run_id bigint;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS block_number bigint;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS prev_hash text;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS root_hash text;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS header jsonb DEFAULT '{}'::jsonb;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS statements jsonb DEFAULT '[]'::jsonb;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
    END IF;
END $$;

-- ============================================================================
-- 7. LEMMA CACHE (canonical)
-- ============================================================================

-- Create lemma_cache table (idempotent - don't drop existing)
CREATE TABLE IF NOT EXISTS lemma_cache (
  id            bigserial primary key,
  statement_id  uuid not null references statements(id) on delete cascade,
  usage_count   int not null default 0,
  last_used     timestamptz not null default now(),
  created_at    timestamptz not null default now(),
  unique(statement_id)
);

-- Ensure all required columns exist in lemma_cache (additive only)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'lemma_cache') THEN
        ALTER TABLE lemma_cache ADD COLUMN IF NOT EXISTS statement_id uuid REFERENCES statements(id) ON DELETE CASCADE;
        ALTER TABLE lemma_cache ADD COLUMN IF NOT EXISTS usage_count int DEFAULT 0;
        ALTER TABLE lemma_cache ADD COLUMN IF NOT EXISTS last_used timestamptz DEFAULT now();
        ALTER TABLE lemma_cache ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
    END IF;
END $$;

-- ============================================================================
-- 8. INDEXES (comprehensive)
-- ============================================================================

-- Theories indexes
CREATE INDEX IF NOT EXISTS theories_slug_idx ON theories(slug);
CREATE INDEX IF NOT EXISTS theories_name_idx ON theories(name);

-- Statements indexes
CREATE INDEX IF NOT EXISTS statements_theory_idx ON statements(theory_id);
CREATE INDEX IF NOT EXISTS statements_system_id_idx ON statements(system_id);
CREATE INDEX IF NOT EXISTS statements_status_idx ON statements(status);
CREATE INDEX IF NOT EXISTS statements_derivation_depth_idx ON statements(derivation_depth);
CREATE INDEX IF NOT EXISTS statements_created_at_idx ON statements(created_at);
CREATE INDEX IF NOT EXISTS statements_content_norm_idx ON statements(content_norm);

-- Proofs indexes
CREATE INDEX IF NOT EXISTS proofs_stmt_idx ON proofs(statement_id);
CREATE INDEX IF NOT EXISTS proofs_system_id_idx ON proofs(system_id);
CREATE INDEX IF NOT EXISTS proofs_prover_idx ON proofs(prover);
CREATE INDEX IF NOT EXISTS proofs_success_idx ON proofs(success);
CREATE INDEX IF NOT EXISTS proofs_created_at_idx ON proofs(created_at);

-- Dependencies indexes
CREATE INDEX IF NOT EXISTS dependencies_proof_idx ON dependencies(proof_id);
CREATE INDEX IF NOT EXISTS dependencies_statement_idx ON dependencies(used_statement_id);

-- Runs indexes
CREATE INDEX IF NOT EXISTS runs_system_id_idx ON runs(system_id);
CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status);
CREATE INDEX IF NOT EXISTS runs_created_at_idx ON runs(created_at);

-- Blocks indexes
CREATE INDEX IF NOT EXISTS blocks_run_id_idx ON blocks(run_id);
CREATE INDEX IF NOT EXISTS blocks_system_id_idx ON blocks(system_id);
CREATE INDEX IF NOT EXISTS blocks_block_number_idx ON blocks(block_number);
CREATE INDEX IF NOT EXISTS blocks_created_at_idx ON blocks(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS blocks_system_block_number_uq ON blocks(system_id, block_number);

-- Lemma cache indexes
CREATE INDEX IF NOT EXISTS lemma_cache_usage_count_idx ON lemma_cache(usage_count desc);
CREATE INDEX IF NOT EXISTS lemma_cache_statement_id_idx ON lemma_cache(statement_id);
CREATE INDEX IF NOT EXISTS lemma_cache_last_used_idx ON lemma_cache(last_used);

-- ============================================================================
-- 9. SEED DATA
-- ============================================================================

-- Insert default theories if missing
INSERT INTO theories (name, slug, version, logic) VALUES
  ('Propositional', 'pl', 'v0', 'classical'),
  ('First Order', 'fol', 'v0', 'classical')
ON CONFLICT (name) DO NOTHING;

-- Insert default run for existing data
INSERT INTO runs (name, system_id, status, completed_at)
SELECT 'initial_run', t.id, 'completed', now()
FROM theories t
WHERE t.slug = 'pl'
  AND NOT EXISTS (SELECT 1 FROM runs LIMIT 1);

-- ============================================================================
-- 10. CONSTRAINTS AND VALIDATIONS
-- ============================================================================

-- Add check constraints for data integrity
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'statements_derivation_depth_positive'
          AND conrelid = 'public.statements'::regclass
    ) THEN
        ALTER TABLE statements
        ADD CONSTRAINT statements_derivation_depth_positive
          CHECK (derivation_depth IS NULL OR derivation_depth >= 0);
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'proofs_time_ms_positive'
          AND conrelid = 'public.proofs'::regclass
    ) THEN
        ALTER TABLE proofs
        ADD CONSTRAINT proofs_time_ms_positive
          CHECK (time_ms IS NULL OR time_ms >= 0);
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'proofs_steps_positive'
          AND conrelid = 'public.proofs'::regclass
    ) THEN
        ALTER TABLE proofs
        ADD CONSTRAINT proofs_steps_positive
          CHECK (steps IS NULL OR steps >= 0);
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'lemma_cache_usage_count_positive'
          AND conrelid = 'public.lemma_cache'::regclass
    ) THEN
        ALTER TABLE lemma_cache
        ADD CONSTRAINT lemma_cache_usage_count_positive
          CHECK (usage_count >= 0);
    END IF;
END
$$;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Log completion
INSERT INTO theories (name, slug, version, logic)
VALUES ('Migration', 'migration', '004', 'schema')
ON CONFLICT (name) DO NOTHING;
