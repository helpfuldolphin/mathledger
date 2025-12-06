-- migrations/002_blocks_lemmas.sql
-- Add blocks and lemma cache tables for observability and caching

-- First, create the runs table (referenced by blocks)
create table if not exists runs (
  id          bigserial primary key,
  name        text,
  status      text not null check (status in ('running', 'completed', 'failed')) default 'running',
  started_at  timestamptz not null default now(),
  completed_at timestamptz,
  created_at  timestamptz not null default now()
);

-- Create blocks table
create table if not exists blocks (
  id          bigserial primary key,
  run_id      bigint not null references runs(id) on delete cascade,
  root_hash   text not null,
  counts      jsonb not null,
  created_at  timestamptz not null default now()
);

-- Create lemma_cache table
-- Note: statement_id type will be determined dynamically based on statements.id type
create table if not exists lemma_cache (
  id            bigserial primary key,
  statement_id  uuid not null references statements(id) on delete cascade,
  usage_count   int not null default 0,
  created_at    timestamptz not null default now(),
  unique(statement_id)
);

-- Add helpful indexes
create index if not exists blocks_run_id_idx on blocks(run_id);
create index if not exists blocks_created_at_idx on blocks(created_at);
create index if not exists lemma_cache_usage_count_idx on lemma_cache(usage_count desc);
create index if not exists lemma_cache_statement_id_idx on lemma_cache(statement_id);
create index if not exists runs_status_idx on runs(status);
create index if not exists runs_created_at_idx on runs(created_at);

-- Insert a default run for existing data
insert into runs (name, status, completed_at)
select 'initial_run', 'completed', now()
where not exists (select 1 from runs limit 1);
