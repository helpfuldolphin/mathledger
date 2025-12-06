-- migrations/001_init.sql
-- MathLedger: core schema for theories, statements, proofs, deps.

-- 0) Extensions (for UUIDs)
create extension if not exists pgcrypto;

-- 1) Theories and symbols
create table if not exists theories (
  id          uuid primary key default gen_random_uuid(),
  name        text not null unique,
  version     text default 'v0',
  logic       text default 'unspecified',
  created_at  timestamptz not null default now()
);

create table if not exists symbols (
  id          uuid primary key default gen_random_uuid(),
  theory_id   uuid not null references theories(id) on delete cascade,
  name        text not null,
  arity       int  not null check (arity >= 0),
  sort        text,
  unique(theory_id, name, arity)
);

-- 2) Statements
create table if not exists statements (
  id              uuid primary key default gen_random_uuid(),
  theory_id       uuid not null references theories(id) on delete cascade,
  hash            bytea not null unique,  -- SHA-256 of normalized form
  sort            text,
  content_norm    text not null,          -- normalized s-expr / canonical form
  content_lean    text,                   -- Lean source (goal)
  content_latex   text,
  status          text not null check (status in ('proven','disproven','open','unknown')) default 'unknown',
  truth_domain    text,                   -- e.g., "classical", "finite-model(n=5)", etc.
  created_at      timestamptz not null default now()
);

-- 3) Proofs
create table if not exists proofs (
  id              uuid primary key default gen_random_uuid(),
  statement_id    uuid not null references statements(id) on delete cascade,
  prover          text not null,          -- e.g., 'lean4'
  proof_term      text,                   -- optional serialized proof term / script
  time_ms         int  check (time_ms >= 0),
  steps           int  check (steps >= 0),
  success         boolean not null default false,
  proof_hash      bytea,                  -- optional hash of proof term
  kernel_version  text,                   -- Lean/mathlib commit for reproducibility
  created_at      timestamptz not null default now()
);

-- 4) Dependencies (which statements a proof used)
create table if not exists dependencies (
  proof_id            uuid not null references proofs(id) on delete cascade,
  used_statement_id   uuid not null references statements(id) on delete cascade,
  primary key (proof_id, used_statement_id)
);

-- 5) Helpful indexes
create index if not exists statements_theory_idx on statements(theory_id);
create index if not exists proofs_stmt_idx      on proofs(statement_id);
create index if not exists statements_status_idx on statements(status);

-- 6) Seed a base theory (Propositional) if missing
insert into theories (name, version, logic)
select 'Propositional', 'v0', 'classical'
where not exists (select 1 from theories where name='Propositional');
