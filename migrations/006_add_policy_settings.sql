-- Add policy settings table for tracking active policy hash
CREATE TABLE IF NOT EXISTS policy_settings (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default policy hash (null)
INSERT INTO policy_settings (key, value) VALUES ('active_policy_hash', NULL)
ON CONFLICT (key) DO NOTHING;
