-- migrations/003_add_system_id.sql
-- Add system_id tagging support for multi-theory systems

-- Add slug field to theories table for easier lookup
ALTER TABLE theories ADD COLUMN IF NOT EXISTS slug text;
CREATE UNIQUE INDEX IF NOT EXISTS theories_slug_idx ON theories(slug);

-- Update existing Propositional theory to have slug 'pl'
UPDATE theories SET slug = 'pl' WHERE name = 'Propositional' AND slug IS NULL;

-- Add system_id fields to all relevant tables (only if tables exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'runs') THEN
        ALTER TABLE runs ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'statements') THEN
        ALTER TABLE statements ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'proofs') THEN
        ALTER TABLE proofs ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'blocks') THEN
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
    END IF;
END $$;

-- Add indexes for system_id queries
CREATE INDEX IF NOT EXISTS runs_system_id_idx ON runs(system_id);
CREATE INDEX IF NOT EXISTS statements_system_id_idx ON statements(system_id);
CREATE INDEX IF NOT EXISTS proofs_system_id_idx ON proofs(system_id);
CREATE INDEX IF NOT EXISTS blocks_system_id_idx ON blocks(system_id);

-- Backfill system_id for existing data (default to Propositional theory) - only if tables exist
DO $$
DECLARE
    prop_id UUID;
BEGIN
    -- Get Propositional theory ID once
    SELECT id INTO prop_id FROM theories WHERE name = 'Propositional' LIMIT 1;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'runs') THEN
        UPDATE runs SET system_id = prop_id WHERE system_id IS NULL AND prop_id IS NOT NULL;
        -- Only set NOT NULL if all rows have system_id
        IF NOT EXISTS (SELECT 1 FROM runs WHERE system_id IS NULL) THEN
            BEGIN
                ALTER TABLE runs ALTER COLUMN system_id SET NOT NULL;
            EXCEPTION WHEN OTHERS THEN
                -- Ignore if constraint already exists or other issues
                NULL;
            END;
        END IF;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'statements') THEN
        UPDATE statements SET system_id = theory_id WHERE system_id IS NULL AND theory_id IS NOT NULL;
        -- Fallback to Propositional if theory_id is NULL
        UPDATE statements SET system_id = prop_id WHERE system_id IS NULL AND prop_id IS NOT NULL;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'proofs') THEN
        UPDATE proofs SET system_id = (SELECT s.system_id FROM statements s WHERE s.id = proofs.statement_id) 
        WHERE system_id IS NULL;
        -- Fallback to Propositional if statement lookup fails
        UPDATE proofs SET system_id = prop_id WHERE system_id IS NULL AND prop_id IS NOT NULL;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'blocks') THEN
        UPDATE blocks SET system_id = (SELECT r.system_id FROM runs r WHERE r.id = blocks.run_id) 
        WHERE system_id IS NULL;
        -- Fallback to Propositional if run lookup fails
        UPDATE blocks SET system_id = prop_id WHERE system_id IS NULL AND prop_id IS NOT NULL;
    END IF;
END $$;
