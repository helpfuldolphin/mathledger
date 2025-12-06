-- migrations/006_add_pg_trgm_extension.sql
-- Enable pg_trgm extension for text search functionality

-- Enable the pg_trgm extension for trigram-based text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create GIN index on content_norm for fast text search if it doesn't exist
CREATE INDEX IF NOT EXISTS statements_content_norm_trgm_idx
ON statements USING gin (content_norm gin_trgm_ops);

-- Create GIN index on content_lean for fast text search if it doesn't exist
CREATE INDEX IF NOT EXISTS statements_content_lean_trgm_idx
ON statements USING gin (content_lean gin_trgm_ops);

-- Create GIN index on content_latex for fast text search if it doesn't exist
CREATE INDEX IF NOT EXISTS statements_content_latex_trgm_idx
ON statements USING gin (content_latex gin_trgm_ops);

-- Log completion
INSERT INTO theories (name, slug, version, logic)
VALUES ('Migration', 'migration', '006', 'pg_trgm_extension')
ON CONFLICT (name) DO NOTHING;
