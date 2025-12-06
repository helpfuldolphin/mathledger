#!/usr/bin/env python3
"""
MathLedger monitor: prints proofs.success and recent blocks on an interval.
- Uses DATABASE_URL if present, else falls back to local canonical DSN.
- Handles transient DB errors with simple reconnect logic.
"""

import os
import sys
import time
from datetime import datetime

import psycopg

# --- Config ---
from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

try:
    DSN = get_database_url()
except MissingEnvironmentVariable as exc:
    raise RuntimeError(str(exc)) from exc
INTERVAL_SEC = int(os.getenv("MONITOR_INTERVAL", "60"))  # default 60 seconds

# Table/column names (discovered)
# blocks: id, block_number, merkle_root, created_at
# proofs: status (expects 'success' for proved entries)

def tick(conn) -> None:
    with conn.cursor() as cur:
        # 1) How many proved proofs do we have?
        cur.execute("SELECT count(*) FROM proofs WHERE status = 'success';")
        n_success = cur.fetchone()[0]

        # 2) Latest block number
        cur.execute("SELECT COALESCE(MAX(block_number), 0) FROM blocks;")
        latest_block = cur.fetchone()[0]

        # 3) Last 5 blocks (newest first)
        cur.execute("""
            SELECT block_number, merkle_root, created_at
            FROM blocks
            ORDER BY block_number DESC
            LIMIT 5;
        """)
        last5 = cur.fetchall()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] proofs.success={n_success}  latest.block={latest_block}")
    if last5:
        for bno, merkle, created in last5:
            merkle_short = (merkle or "")[:16]
            print(f"  block={bno:<6}  merkle={merkle_short:16}  created={created}")
    else:
        print("  (no blocks yet)")
    sys.stdout.flush()

def main():
    print(f"Connecting to {DSN} ...")
    conn = None
    while True:
        try:
            if conn is None or conn.closed:
                conn = psycopg.connect(DSN, autocommit=True)
                print("DB connected.")

            tick(conn)
            time.sleep(INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            break

        except Exception as e:
            # Print error, sleep a bit, and try to reconnect.
            print(f"[WARN] Monitor error: {e!r}")
            try:
                if conn and not conn.closed:
                    conn.close()
            except Exception:
                pass
            conn = None
            time.sleep(5)

if __name__ == "__main__":
    main()
