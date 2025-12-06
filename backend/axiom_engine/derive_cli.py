import os
import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.axiom_engine.derive import derive

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system-id", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--depth-max", type=int, default=3,
                   help="Maximum derivation depth (currently unused by derive function)")
    ap.add_argument("--max-breadth", type=int, default=100)
    ap.add_argument("--max-total", type=int, default=1000)
    args = ap.parse_args()

    from backend.security.runtime_env import get_required_env

    url = get_required_env("DATABASE_URL")
    engine = create_engine(url, future=True)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as db:
        derive(db_session=db, system_id=args.system_id,
               steps=args.steps, breadth_cap=args.max_breadth, total_cap=args.max_total)

if __name__ == "__main__":
    main()
