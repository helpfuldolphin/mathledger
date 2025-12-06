import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from backend.axiom_engine.derive_core import DerivationEngine
from backend.security.runtime_env import get_database_url, get_redis_url

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-breadth", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=1000)
    parser.add_argument("--depth-max", type=int, default=4)
    args = parser.parse_args()

    db_url = get_database_url()
    redis_url = get_redis_url()

    print(f"Initializing DerivationEngine for System {args.system_id}...")
    engine = DerivationEngine(
        db_url=db_url,
        redis_url=redis_url,
        max_depth=args.depth_max,
        max_breadth=args.max_breadth,
        max_total=args.max_total
    )

    print(f"Deriving {args.steps} steps...")
    stats = engine.derive_statements(steps=args.steps)
    
    print(f"Derivation Complete: {stats}")

if __name__ == "__main__":
    main()
