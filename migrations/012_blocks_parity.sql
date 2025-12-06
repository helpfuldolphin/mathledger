-- Only apply if blocks table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'blocks') THEN
        -- Add columns if missing
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS block_number BIGINT;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS prev_hash TEXT;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS merkle_root TEXT;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS header JSONB;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS proof_count INTEGER;
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;

        -- Alter column types and defaults (only if columns exist)
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'header') THEN
            ALTER TABLE blocks ALTER COLUMN header TYPE JSONB USING header::jsonb;
        END IF;

        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'proof_count') THEN
            ALTER TABLE blocks ALTER COLUMN proof_count SET DEFAULT 0;
        END IF;

        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'blocks' AND column_name = 'created_at') THEN
            ALTER TABLE blocks ALTER COLUMN created_at SET DEFAULT now();
        END IF;

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_blocks_number ON blocks(block_number DESC);
        CREATE INDEX IF NOT EXISTS idx_blocks_created_at ON blocks(created_at DESC);
    END IF;
END $$;
