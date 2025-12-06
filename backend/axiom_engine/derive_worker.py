import os, time, json, logging, signal, hashlib
from typing import List, Dict, Any, Optional
import redis, psycopg
from backend.axiom_engine.derive import DerivationEngine
from backend.ledger.blockchain import seal_block, merkle_root
from normalization.canon import normalize
from backend.repro.determinism import deterministic_timestamp_from_content, deterministic_hash
from backend.security.runtime_env import (
    MissingEnvironmentVariable,
    get_database_url,
    get_redis_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    DB_URL = get_database_url()
except MissingEnvironmentVariable as exc:
    raise RuntimeError(str(exc)) from exc

try:
    REDIS_URL = get_redis_url()
except MissingEnvironmentVariable as exc:
    raise RuntimeError(str(exc)) from exc
QUEUE     = "ml:jobs"

# Force demo sealing: 1 statement per block, or by time
BATCH_N = int(os.getenv("SEAL_BATCH_N", "50"))
BATCH_T = float(os.getenv("SEAL_BATCH_T", "60"))

shutdown = False
def _handle_signal(signum, frame):
    global shutdown
    shutdown = True
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

def _get_last_block(conn) -> Optional[Dict[str, Any]]:
    with conn, conn.cursor() as cur:
        cur.execute("SELECT block_number, merkle_root FROM blocks ORDER BY block_number DESC LIMIT 1")
        row = cur.fetchone()
        if not row: return None
        return {"block_number": row[0], "merkle_root": row[1]}

def _insert_block(conn, blk: Dict[str, Any]):
    h = blk["header"]
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO blocks(block_number, prev_hash, merkle_root, header, statements) VALUES (%s,%s,%s,%s::jsonb,%s::jsonb)",
            (h["block_number"], h["prev_hash"], h["merkle_root"], json.dumps(h), json.dumps(blk["statements"]))
        )

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def main():
    eng = DerivationEngine(DB_URL, REDIS_URL)
    r = redis.from_url(REDIS_URL)
    logging.info("Worker started, DB=%s, Redis=%s", DB_URL, REDIS_URL)

    pending_ids: List[str] = []
    last_seal_ts = time.time()
    with psycopg.connect(DB_URL) as conn:
        last_blk = _get_last_block(conn)
        if last_blk:
            prev_hash    = last_blk["merkle_root"]
            block_number = int(last_blk["block_number"]) + 1
        else:
            prev_hash    = "GENESIS"
            block_number = 1

    while not shutdown:
        try:
            item = r.lpop(QUEUE)
            if item is None:
                # Seal by time if buffer exists
                now_op = time.time()
                if pending_ids and (now_op - last_seal_ts) >= BATCH_T:
                    # Content-derived timestamp for the block
                    mroot = merkle_root(pending_ids)
                    ts_dt = deterministic_timestamp_from_content(mroot)
                    ts_block = ts_dt.timestamp()
                    
                    with psycopg.connect(DB_URL) as conn:
                        blk = seal_block(pending_ids, prev_hash, block_number, ts_block, version="v1")
                        _insert_block(conn, blk)
                        logging.info("sealed block number=%d statements=%d merkle=%s ts=%s", 
                                     block_number, len(pending_ids), blk["header"]["merkle_root"], ts_dt.isoformat())
                        prev_hash    = blk["header"]["merkle_root"]
                        block_number += 1
                        pending_ids.clear()
                        last_seal_ts = now_op
                time.sleep(0.8)
                continue

            try:
                job = json.loads(item)
            except Exception:
                job = {"raw": item.decode() if isinstance(item, (bytes, bytearray)) else str(item)}

            t0 = time.perf_counter()
            summary: Dict[str, Any] = eng.derive_statements(steps=1)
            t1 = time.perf_counter()

            # Always buffer at least one surrogate ID per job so sealing progresses
            txt  = normalize(str(job.get("text", "tick"))) or "tick"
            sid0 = _sha(txt)
            cnt  = max(1, int(summary.get("n_new", 0) or 0))
            
            # Deterministic ID generation based on content + block context
            # Replaces time.time_ns() which is nondeterministic
            for k in range(cnt):
                # Mix in block_number to ensure uniqueness across blocks if same content repeats
                suffix_seed = f"{sid0}:{block_number}:{k}"
                suffix = deterministic_hash(suffix_seed, algorithm="sha256")[:8]
                pending_ids.append(f"{sid0}:{suffix}")

            # Seal by count (N=1 forces seal every job)
            if len(pending_ids) >= BATCH_N:
                now_op = time.time() # Operational time for latency/logs
                
                # Content-derived timestamp for the block itself
                mroot = merkle_root(pending_ids)
                ts_dt = deterministic_timestamp_from_content(mroot)
                ts_block = ts_dt.timestamp()
                
                with psycopg.connect(DB_URL) as conn:
                    blk = seal_block(pending_ids, prev_hash, block_number, ts_block, version="v1")
                    _insert_block(conn, blk)
                    logging.info("sealed block number=%d statements=%d merkle=%s ts=%s", 
                                 block_number, len(pending_ids), blk["header"]["merkle_root"], ts_dt.isoformat())
                    prev_hash    = blk["header"]["merkle_root"]
                    block_number += 1
                    pending_ids.clear()
                    last_seal_ts = now_op

            logging.info("tick n_new=%s max_depth=%s n_jobs=%s pct_success=%s latency_ms=%.1f queue_len=%s",
                         summary.get("n_new"), summary.get("max_depth"), summary.get("n_jobs"),
                         summary.get("pct_success"), (t1 - t0)*1000.0, r.llen(QUEUE))

        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception("Worker error: %s", e)
            time.sleep(1.5)

    logging.info("Worker shutting down")

if __name__ == "__main__":
    main()
def _persist_job_echo(db_url: str, job_text: str, depth: int = 0):
    import psycopg
    try:
        with psycopg.connect(db_url) as c, c.cursor() as cur:
            cur.execute("""
                INSERT INTO derived_statements (statement_id, statement_text, derivation_rule, derivation_depth)
                VALUES (%s,%s,%s,%s)
            """, (None, job_text, "job_enqueued", depth))
    except Exception:
        pass
