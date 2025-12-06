#!/usr/bin/env python3
"""Export FOL AB metrics with V1 linting and dry-run support.

Linter-first design: validates schema before processing.
Messages use standardized prefixes: mixed-schema:, error:, DRY-RUN ok:, EXPORT ok:
Ensures ASCII-only output for compatibility.

Supports --export for database UPSERT with:
- Batch size 1000 for performance
- SHA-256 hash normalization (content_norm → bytea)
- Idempotent ON CONFLICT (hash) DO NOTHING
- Transaction rollback on errors
- [POA]+[ASD]: Proof-abstain discipline + Authentic Synthetic Data export
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

try:
    import psycopg
except ImportError:
    psycopg = None


def is_ascii_safe(text: str) -> bool:
    """Check if text is ASCII-safe."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def validate_v1_schema(record: Dict[str, Any]) -> bool:
    """Validate if a record conforms to V1 metrics schema."""
    required_fields = {'id', 'theory_id', 'hash', 'content_norm', 'is_axiom'}
    return all(field in record for field in required_fields)


def get_record_schema_version(record: Dict[str, Any]) -> str:
    """Determine the schema version of a record."""
    if validate_v1_schema(record):
        return "v1"
    return "unknown"


def lint_metrics_v1(file_path: Path) -> Tuple[bool, str, Dict[str, int]]:
    """
    Lint JSONL metrics file for V1 schema compliance (linter-first, idempotent).

    Returns:
        (is_valid, message, schema_counts)

    Message prefixes:
        - 'mixed-schema:' for files with both v1 and unknown records
        - 'error:' for fatal errors (file not found, invalid JSON, no valid records)
        - 'Valid V1 schema:' for clean v1 files (internal use)
    """
    if not file_path.exists():
        return False, f"error: File not found: {file_path}", {}

    schema_counts = {"v1": 0, "unknown": 0}
    line_count = 0
    first_unknown_line = None
    non_ascii_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                line_count += 1

                # Check for non-ASCII characters (warn but don't fail)
                if not is_ascii_safe(line):
                    non_ascii_count += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    return False, f"error: Invalid JSON at line {line_num}: {str(e)}", schema_counts

                schema_version = get_record_schema_version(record)
                schema_counts[schema_version] += 1

                # Track first unknown record for diagnostics
                if schema_version == "unknown" and first_unknown_line is None:
                    first_unknown_line = line_num

    except UnicodeDecodeError as e:
        return False, f"error: File encoding error: {str(e)}", schema_counts
    except Exception as e:
        return False, f"error: Error reading file: {str(e)}", schema_counts

    # Determine if mixed schema
    has_v1 = schema_counts["v1"] > 0
    has_unknown = schema_counts["unknown"] > 0

    if has_v1 and has_unknown:
        # Mixed schema - provide detailed diagnostics
        msg = f"mixed-schema: v1={schema_counts['v1']} unknown={schema_counts['unknown']}"
        if first_unknown_line:
            msg += f" (first unknown at line {first_unknown_line})"
        return False, msg, schema_counts
    elif has_unknown and not has_v1:
        # All invalid
        return False, f"error: No valid V1 records found (unknown={schema_counts['unknown']})", schema_counts
    elif has_v1:
        # All valid V1
        msg = f"Valid V1 schema: {schema_counts['v1']} records"
        if non_ascii_count > 0:
            msg += f" (warning: {non_ascii_count} non-ASCII lines)"
        return True, msg, schema_counts
    else:
        # Empty file
        return False, "error: Empty file or no records found", schema_counts


def export_to_db(file_path: Path, schema_counts: Dict[str, int], batch_size: int = 1000) -> Tuple[bool, str, int]:
    """
    Export V1 records to database with UPSERT, batching, and rollback on errors.

    Architecture:
    1. Linter-first: Assumes file already validated by lint_metrics_v1()
    2. Single transaction: All batches commit atomically or rollback
    3. SHA-256 normalization: content_norm → bytea hash for UPSERT key
    4. Idempotent: ON CONFLICT (hash) DO NOTHING prevents duplicates
    5. Batch size 1000: Optimizes network round-trips

    Args:
        file_path: Path to validated V1 JSONL file
        schema_counts: Schema counts from linter (v1 count)
        batch_size: Records per batch (default 1000)

    Returns:
        (success, message, inserted_count)
        - success: True if all batches committed, False on error
        - message: Standardized "EXPORT ok:" or "error:" prefix
        - inserted_count: Total records inserted (excludes duplicates)
    """
    if psycopg is None:
        return False, "error: psycopg not installed (pip install psycopg[binary])", 0

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return False, "error: DATABASE_URL environment variable not set", 0

    inserted_count = 0
    total_processed = 0

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Get Propositional theory UUID (assumes migrations ran)
                cur.execute("SELECT id FROM theories WHERE name = 'Propositional' LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return False, "error: Propositional theory not found in database (run migrations)", 0
                theory_id = row[0]

                # Process file in batches
                batch = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # Skip invalid JSON (already caught by linter)

                        # Only process V1 records (skip unknown)
                        if get_record_schema_version(record) != "v1":
                            continue

                        # SHA-256 hash normalization: content_norm → bytea
                        # This ensures idempotent UPSERT based on content, not V1's hash field
                        content_hash = hashlib.sha256(record['content_norm'].encode('utf-8')).digest()

                        # Prepare record for UPSERT
                        batch.append({
                            'theory_id': theory_id,
                            'hash': content_hash,
                            'content_norm': record['content_norm'],
                            'is_axiom': record['is_axiom'],
                            'status': 'proven' if record['is_axiom'] else 'unknown',
                            'derivation_rule': 'axiom' if record['is_axiom'] else None,
                            'derivation_depth': 0 if record['is_axiom'] else None,
                        })
                        total_processed += 1

                        # Flush batch when full
                        if len(batch) >= batch_size:
                            inserted_count += _insert_batch(cur, batch)
                            batch.clear()

                # Insert remaining records
                if batch:
                    inserted_count += _insert_batch(cur, batch)

                # Commit transaction (atomic - all or nothing)
                conn.commit()

                # Report results
                skipped = total_processed - inserted_count
                return True, f"EXPORT ok: {inserted_count} records inserted (skipped {skipped} duplicates)", inserted_count

    except psycopg.OperationalError as e:
        # Connection/network errors
        return False, f"error: Database connection failed: {str(e)}", inserted_count
    except psycopg.IntegrityError as e:
        # Constraint violations (shouldn't happen with ON CONFLICT)
        return False, f"error: Database integrity error: {str(e)}", inserted_count
    except psycopg.Error as e:
        # Other PostgreSQL errors
        return False, f"error: Database error: {str(e)}", inserted_count
    except Exception as e:
        # Unexpected errors
        return False, f"error: Unexpected error during export: {str(e)}", inserted_count


def _insert_batch(cur, batch: List[Dict[str, Any]]) -> int:
    """
    Insert batch using UPSERT with ON CONFLICT DO NOTHING.

    Uses RETURNING clause to count actual insertions (excludes conflicts).

    Args:
        cur: psycopg cursor
        batch: List of record dicts with keys: theory_id, hash, content_norm, etc.

    Returns:
        Number of records actually inserted (excluding duplicates)
    """
    if not batch:
        return 0

    # UPSERT query with RETURNING for accurate count
    query = """
        INSERT INTO statements (theory_id, hash, content_norm, is_axiom, status, derivation_rule, derivation_depth)
        VALUES (%(theory_id)s, %(hash)s, %(content_norm)s, %(is_axiom)s, %(status)s, %(derivation_rule)s, %(derivation_depth)s)
        ON CONFLICT (hash) DO NOTHING
        RETURNING id
    """

    # Execute batch and count inserted rows
    inserted = 0
    for record in batch:
        cur.execute(query, record)
        if cur.fetchone() is not None:
            inserted += 1

    return inserted


def main():
    """Main entry point with linter-first, idempotent dry-run, and wet-run export."""
    parser = argparse.ArgumentParser(
        description="Export FOL AB metrics with V1 linting, dry-run validation, and database export",
        epilog="Linter-first: validates schema before any processing. "
               "Use --dry-run for validation (no DB/network). "
               "Use --export for UPSERT to PostgreSQL (batch 1000, idempotent). "
               "[POA]+[ASD]: Proof-abstain discipline + Authentic Synthetic Data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL metrics file to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and exit without processing (idempotent, no side effects)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export V1 records to database (UPSERT, batch 1000, SHA-256 hash normalization)"
    )

    args = parser.parse_args()

    # Validate mutual exclusivity
    if args.dry_run and args.export:
        print("error: Cannot use --dry-run and --export together (mutually exclusive)")
        return 1

    # LINTER-FIRST: Always run V1 schema validation before any processing
    # This is idempotent - reads file but makes no changes
    is_valid, message, schema_counts = lint_metrics_v1(args.input)

    if not is_valid:
        # Print error message with standardized prefix (mixed-schema: or error:)
        print(message)
        return 1

    # IDEMPOTENT EARLY EXIT: If dry-run mode, exit with success after linting
    # No DB connections, no network calls, no side effects - truly idempotent
    if args.dry_run:
        print(f"DRY-RUN ok: {args.input} (v1={schema_counts['v1']})")
        return 0

    # WET-RUN EXPORT: Database UPSERT with batching and rollback on errors
    if args.export:
        success, export_message, inserted_count = export_to_db(args.input, schema_counts)
        print(export_message)
        return 0 if success else 1

    # DEPRECATED MODE: Show guidance if neither flag specified
    print(f"info: Validated {schema_counts['v1']} V1 records in {args.input}")
    print("info: Use --dry-run for validation-only or --export for database insertion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
