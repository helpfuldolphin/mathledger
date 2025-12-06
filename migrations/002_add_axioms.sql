-- migrations/002_add_axioms.sql
-- Add support for axiomatic systems

-- Add is_axiom field to statements table
-- Use DO block to handle NOT NULL constraint safely
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'statements'
          AND column_name = 'is_axiom'
    ) THEN
        ALTER TABLE statements ADD COLUMN is_axiom boolean DEFAULT false;
        UPDATE statements SET is_axiom = false WHERE is_axiom IS NULL;
        ALTER TABLE statements ALTER COLUMN is_axiom SET NOT NULL;
    END IF;
END $$;

-- Add derivation tracking fields
ALTER TABLE statements ADD COLUMN IF NOT EXISTS derivation_rule text;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS derivation_depth integer DEFAULT 0;

-- Update dependencies table to track derivation relationships
-- (keeping existing proof dependencies, adding derivation dependencies)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'dependencies'
          AND column_name = 'dependency_type'
    ) THEN
        ALTER TABLE dependencies ADD COLUMN dependency_type text DEFAULT 'proof';
        UPDATE dependencies SET dependency_type = 'proof' WHERE dependency_type IS NULL;
    END IF;
    
    -- Add check constraint if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'dependencies_dependency_type_check'
    ) THEN
        ALTER TABLE dependencies ADD CONSTRAINT dependencies_dependency_type_check
        CHECK (dependency_type IN ('proof', 'derivation'));
    END IF;
END $$;

-- Add index for axiom queries
CREATE INDEX IF NOT EXISTS statements_is_axiom_idx ON statements(is_axiom);
CREATE INDEX IF NOT EXISTS statements_derivation_rule_idx ON statements(derivation_rule);
CREATE INDEX IF NOT EXISTS statements_derivation_depth_idx ON statements(derivation_depth);

-- Insert the axioms: K and S
-- Handle both bytea and text hash columns by encoding digest result
DO $$
DECLARE
    hash_type TEXT;
    k_hash BYTEA;
    s_hash BYTEA;
    k_hash_text TEXT;
    s_hash_text TEXT;
BEGIN
    -- Detect hash column type
    SELECT data_type INTO hash_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'statements'
      AND column_name = 'hash';
    
    -- Compute hashes once
    k_hash := digest('p -> (q -> p)', 'sha256');
    s_hash := digest('(p -> (q -> r)) -> ((p -> q) -> (p -> r))', 'sha256');
    k_hash_text := encode(k_hash, 'hex');
    s_hash_text := encode(s_hash, 'hex');
    
    -- Insert K axiom
    IF hash_type = 'bytea' THEN
        INSERT INTO statements (theory_id, hash, content_norm, is_axiom, derivation_rule, derivation_depth, status)
        SELECT
            t.id,
            k_hash,
            'p -> (q -> p)',
            true,
            'axiom',
            0,
            'proven'
        FROM theories t
        WHERE t.name = 'Propositional'
        AND NOT EXISTS (
            SELECT 1 FROM statements s
            WHERE s.content_norm = 'p -> (q -> p)'
            AND s.is_axiom = true
        );
    ELSE
        INSERT INTO statements (theory_id, hash, content_norm, is_axiom, derivation_rule, derivation_depth, status)
        SELECT
            t.id,
            k_hash_text,
            'p -> (q -> p)',
            true,
            'axiom',
            0,
            'proven'
        FROM theories t
        WHERE t.name = 'Propositional'
        AND NOT EXISTS (
            SELECT 1 FROM statements s
            WHERE s.content_norm = 'p -> (q -> p)'
            AND s.is_axiom = true
        );
    END IF;
    
    -- Insert S axiom
    IF hash_type = 'bytea' THEN
        INSERT INTO statements (theory_id, hash, content_norm, is_axiom, derivation_rule, derivation_depth, status)
        SELECT
            t.id,
            s_hash,
            '(p -> (q -> r)) -> ((p -> q) -> (p -> r))',
            true,
            'axiom',
            0,
            'proven'
        FROM theories t
        WHERE t.name = 'Propositional'
        AND NOT EXISTS (
            SELECT 1 FROM statements s
            WHERE s.content_norm = '(p -> (q -> r)) -> ((p -> q) -> (p -> r))'
            AND s.is_axiom = true
        );
    ELSE
        INSERT INTO statements (theory_id, hash, content_norm, is_axiom, derivation_rule, derivation_depth, status)
        SELECT
            t.id,
            s_hash_text,
            '(p -> (q -> r)) -> ((p -> q) -> (p -> r))',
            true,
            'axiom',
            0,
            'proven'
        FROM theories t
        WHERE t.name = 'Propositional'
        AND NOT EXISTS (
            SELECT 1 FROM statements s
            WHERE s.content_norm = '(p -> (q -> r)) -> ((p -> q) -> (p -> r))'
            AND s.is_axiom = true
        );
    END IF;
END $$;
