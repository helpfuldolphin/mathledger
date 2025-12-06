BEGIN;
-- Create runs table for logging policy runs and performance metrics
CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    system TEXT NOT NULL DEFAULT 'pl',
    slice TEXT NOT NULL,
    params_json JSONB,
    policy_hash TEXT,
    proofs_success INTEGER DEFAULT 0,
    proofs_per_sec REAL DEFAULT 0.0,
    depth_max_reached INTEGER DEFAULT 0,
    abstain_pct REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_policy_hash ON runs(policy_hash);
CREATE INDEX IF NOT EXISTS idx_runs_system_slice ON runs(system, slice);
CREATE INDEX IF NOT EXISTS idx_runs_proofs_success ON runs(proofs_success DESC);

-- Add constraint for valid percentages (using DO block for IF NOT EXISTS)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'runs_abstain_pct_check') THEN
        ALTER TABLE runs ADD CONSTRAINT runs_abstain_pct_check
            CHECK (abstain_pct >= 0.0 AND abstain_pct <= 100.0);
    END IF;
END $$;

-- Add constraint for positive performance metrics
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'runs_proofs_per_sec_check') THEN
        ALTER TABLE runs ADD CONSTRAINT runs_proofs_per_sec_check
            CHECK (proofs_per_sec >= 0.0);
    END IF;
END $$;

COMMIT;
