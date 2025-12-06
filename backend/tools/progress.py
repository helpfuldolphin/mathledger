"""
Progress tracking utilities for MathLedger.

Provides functions to append run summaries to progress.md.
"""

import os
import argparse
import psycopg
from datetime import datetime
from typing import Optional, Dict, Any


def append_to_progress(md_path: str, summary: dict) -> None:
    """
    Append a derivation run summary to progress.md.

    Args:
        md_path: Path to the progress.md file
        summary: Dictionary containing run statistics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Format the summary entry
    entry = f"""## [{timestamp}] Derivation run
- steps: {summary.get('steps', 0)}
- new statements: {summary.get('n_new', 0)}
- max depth: {summary.get('max_depth', 0)}
- queue enqueued: {summary.get('n_jobs', 0)}
- success % (last hour): {summary.get('pct_success', 0.0):.1f}

"""

    # Append to file
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write(entry)


def get_progress_path() -> str:
    """Get the default progress.md path."""
    # Check if docs directory exists, otherwise use root
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    docs_path = os.path.join(root_dir, "docs", "progress.md")
    if os.path.exists(os.path.dirname(docs_path)):
        return docs_path
    else:
        # Fallback to root directory
        return os.path.join(root_dir, "progress.md")


def get_latest_run_data(db_url: str, offline: bool = False) -> Dict[str, Any]:
    """
    Get the latest run data from the database.

    Args:
        db_url: Database connection URL
        offline: If True, use mock data instead of database

    Returns:
        Dictionary containing latest run statistics
    """
    if offline:
        # Check ratchet_last.txt for schema violations
        try:
            with open('metrics/ratchet_last.txt', 'r') as f:
                ratchet = json.load(f)
                if 'advance_flag' in ratchet or 'next_slice' in ratchet:
                    with open('metrics/integration_errors.log', 'a') as log:
                        log.write(f"{datetime.now().isoformat()} progress=blocked ratchet_schema_violation\n")
                    print("Progress blocked: ratchet schema violation detected")
                    return
        except: pass
        # Use mock data for offline mode
        mock_block = {
            'block_number': 999,
            'merkle_root': '0x' + 'a' * 64,
            'created_at': datetime.now(),
            'header': '{"statements": 100}',
            'block_height': 999
        }
        return {
            'latest_block': mock_block,
            'statements': 100,
            'proofs_total': 85,
            'proofs_success': 80
        }

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Get latest block
            cur.execute("""
                SELECT block_number, merkle_root, created_at, header
                FROM blocks
                ORDER BY block_number DESC
                LIMIT 1
            """)
            latest_block = cur.fetchone()

            # Get max block height
            cur.execute("SELECT COALESCE(MAX(block_number), 0) FROM blocks")
            max_height = cur.fetchone()[0]

            # Get statement counts
            cur.execute("SELECT COUNT(*) FROM statements")
            statements = cur.fetchone()[0]

            # Get proof statistics (gracefully handle missing proofs table)
            proofs_total = 0
            proofs_success = 0

            try:
                cur.execute("SELECT COUNT(*) FROM proofs")
                proofs_total = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM proofs WHERE status='success'")
                proofs_success = cur.fetchone()[0]
            except Exception:
                # Proofs table doesn't exist or has issues - treat as 0 gracefully
                proofs_total = 0
                proofs_success = 0

            return {
                'latest_block': {
                    'block_number': latest_block[0] if latest_block else None,
                    'merkle_root': latest_block[1] if latest_block else None,
                    'created_at': latest_block[2] if latest_block else None,
                    'header': latest_block[3] if latest_block else None,
                    'block_height': max_height
                },
                'statements': statements,
                'proofs_total': proofs_total,
                'proofs_success': proofs_success
            }


def append_latest_to_progress(md_path: str, db_url: str, offline: bool = False) -> None:
    """
    Append latest run data to progress.md.

    Args:
        md_path: Path to the progress.md file
        db_url: Database connection URL
        offline: If True, use mock data instead of database
    """
    try:
        data = get_latest_run_data(db_url, offline)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Check if this block was already logged (idempotent)
        if _is_block_already_logged(md_path, data['latest_block']['block_number']):
            print(f"Block {data['latest_block']['block_number']} already logged, skipping")
            return

        # Format the entry according to requirements
        entry = f"""## [{timestamp}] Block {data['latest_block']['block_number'] or 'None'}
- merkle_root: {data['latest_block']['merkle_root'] or 'None'}
- block_height: {data['latest_block']['block_height']}
- statements: {data['statements']}
- proofs_total: {data['proofs_total']}
- proofs_success: {data['proofs_success']}

"""

        # Append to file
        with open(md_path, 'a', encoding='utf-8') as f:
            f.write(entry)

        print(f"Latest run summary appended to {md_path}")

    except Exception as e:
        print(f"Error appending latest run data: {e}")
        raise


def _is_block_already_logged(md_path: str, block_number: Optional[int]) -> bool:
    """
    Check if a block was already logged to avoid duplicate entries.

    Args:
        md_path: Path to the progress.md file
        block_number: Block number to check for

    Returns:
        True if block was already logged, False otherwise
    """
    if not block_number or not os.path.exists(md_path):
        return False

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for existing entry with this block number in the tail N lines
            lines = content.split('\n')
            tail_lines = lines[-50:] if len(lines) > 50 else lines  # Check last 50 lines
            tail_content = '\n'.join(tail_lines)
            return f"Block {block_number}" in tail_content
    except Exception:
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="MathLedger progress tracking utilities")
    parser.add_argument('--append-latest', action='store_true',
                       help='Append latest run data to progress.md')
    parser.add_argument('--offline', action='store_true',
                       help='Use mock data instead of database connection')
    parser.add_argument('--progress-path', default=None,
                       help='Path to progress.md file (default: docs/progress.md)')
    parser.add_argument('--db-url', default=None,
                       help='Database URL (default: from DATABASE_URL env var)')

    args = parser.parse_args()

    # Set defaults
    if not args.progress_path:
        args.progress_path = get_progress_path()

    if not args.db_url:
        from backend.security.runtime_env import get_required_env

        args.db_url = get_required_env("DATABASE_URL")

    if args.append_latest:
        append_latest_to_progress(args.progress_path, args.db_url, args.offline)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
