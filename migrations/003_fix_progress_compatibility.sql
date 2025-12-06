-- migration_003_fix_progress_compatibility.sql
-- Ensures schema compatibility with progress.py requirements

-- Ensure blocks table has required columns
DO $$
BEGIN
    -- Add block_number if missing (some schemas use 'id' instead)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'block_number') THEN
        ALTER TABLE blocks ADD COLUMN block_number BIGINT;
        -- Populate with existing id values if id column exists
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'id') THEN
            UPDATE blocks SET block_number = id WHERE block_number IS NULL;
        END IF;
    END IF;

    -- Add merkle_root if missing (some schemas use 'root_hash')
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'merkle_root') THEN
        ALTER TABLE blocks ADD COLUMN merkle_root TEXT;
        -- Copy from root_hash if it exists
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'root_hash') THEN
            UPDATE blocks SET merkle_root = root_hash WHERE merkle_root IS NULL;
        END IF;
    END IF;

    -- Add header column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'header') THEN
        ALTER TABLE blocks ADD COLUMN header JSONB DEFAULT '{}';
    END IF;

    -- Add created_at if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'created_at') THEN
        ALTER TABLE blocks ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();
    END IF;
END $$;

-- Ensure statements table has required columns
DO $$
BEGIN
    -- Add text column if missing (some schemas use 'content_norm')
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'statements' AND column_name = 'text') THEN
        ALTER TABLE statements ADD COLUMN text TEXT;
        -- Copy from content_norm if it exists
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'statements' AND column_name = 'content_norm') THEN
            UPDATE statements SET text = content_norm WHERE text IS NULL;
        END IF;
    END IF;

    -- Add system_id if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'statements' AND column_name = 'system_id') THEN
        ALTER TABLE statements ADD COLUMN system_id UUID;
        -- Copy from theory_id if it exists
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'statements' AND column_name = 'theory_id') THEN
            UPDATE statements SET system_id = theory_id WHERE system_id IS NULL;
        END IF;
    END IF;

    -- Add is_axiom if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'statements' AND column_name = 'is_axiom') THEN
        ALTER TABLE statements ADD COLUMN is_axiom BOOLEAN DEFAULT FALSE;
    END IF;

    -- Add derivation_depth if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'statements' AND column_name = 'derivation_depth') THEN
        ALTER TABLE statements ADD COLUMN derivation_depth INTEGER DEFAULT 0;
    END IF;
END $$;

-- Ensure proofs table has status column (not success)
DO $$
BEGIN
    -- Add status column if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'proofs' AND column_name = 'status') THEN
        ALTER TABLE proofs ADD COLUMN status TEXT DEFAULT 'unknown';
        -- Copy from success column if it exists
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'proofs' AND column_name = 'success') THEN
            UPDATE proofs SET status = CASE WHEN success = true THEN 'success' ELSE 'failed' END;
        END IF;
    END IF;

    -- Add statement_id if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'proofs' AND column_name = 'statement_id') THEN
        ALTER TABLE proofs ADD COLUMN statement_id UUID;
    END IF;
END $$;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS blocks_block_number_idx ON blocks(block_number);
CREATE INDEX IF NOT EXISTS statements_system_id_idx ON statements(system_id);
CREATE INDEX IF NOT EXISTS proofs_status_idx ON proofs(status);
CREATE INDEX IF NOT EXISTS proofs_statement_id_idx ON proofs(statement_id);
