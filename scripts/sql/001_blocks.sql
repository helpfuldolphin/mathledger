-- scripts/sql/001_blocks.sql
-- Block storage schema for MathLedger blockchain

CREATE TABLE IF NOT EXISTS blocks (
  id BIGSERIAL PRIMARY KEY,
  block_number BIGINT NOT NULL,
  prev_hash TEXT NOT NULL,
  merkle_root TEXT NOT NULL,
  header JSONB NOT NULL,
  statements JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS blocks_block_number_uq ON blocks(block_number);
