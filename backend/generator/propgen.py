#!/usr/bin/env python3
"""
Propositional formula generator for MathLedger.

Generates propositional formulas with given atoms and maximum depth,
then enqueues them as JSON jobs to Redis.
"""

import argparse
import json
import os
import redis
from typing import List

from backend.lean_interface import sanitize_statement


def normalize_connective(s: str) -> str:
    """Normalize Unicode/LaTeX connectives to canonical ASCII."""
    stmt = sanitize_statement(s)
    return stmt.ascii_pretty


def generate_atomic_formulas(atoms: List[str]) -> List[str]:
    """Generate atomic formulas (just the atoms themselves)."""
    return atoms.copy()


def generate_negations(formulas: List[str]) -> List[str]:
    """Generate negations of existing formulas."""
    return [f"~{f}" for f in formulas]


def generate_implications(formulas: List[str]) -> List[str]:
    """Generate implications between all pairs of formulas."""
    result = []
    for i, left in enumerate(formulas):
        for j, right in enumerate(formulas):
            if i != j:  # Avoid p -> p for now (handled separately)
                result.append(f"({left} -> {right})")
    return result


def generate_conjunctions(formulas: List[str]) -> List[str]:
    """Generate conjunctions between all pairs of formulas."""
    result = []
    for i, left in enumerate(formulas):
        for j, right in enumerate(formulas[i+1:], i+1):  # Avoid duplicates
            result.append(f"({left} /\\ {right})")
    return result


def generate_disjunctions(formulas: List[str]) -> List[str]:
    """Generate disjunctions between all pairs of formulas."""
    result = []
    for i, left in enumerate(formulas):
        for j, right in enumerate(formulas[i+1:], i+1):  # Avoid duplicates
            result.append(f"({left} \\/ {right})")
    return result


def generate_formulas_at_depth(atoms: List[str], depth: int) -> List[str]:
    """Generate all propositional formulas up to given depth."""
    if depth < 1:
        return []

    # Depth 1: atomic formulas
    formulas = generate_atomic_formulas(atoms)
    seen = set(formulas)
    all_formulas = formulas.copy()

    # Depth 2+: compound formulas
    for d in range(2, depth + 1):
        new_formulas = []

        # Generate negations
        new_formulas.extend(generate_negations(formulas))

        # Generate implications
        new_formulas.extend(generate_implications(formulas))

        # Generate conjunctions
        new_formulas.extend(generate_conjunctions(formulas))

        # Generate disjunctions
        new_formulas.extend(generate_disjunctions(formulas))

        # Normalize and deduplicate
        new_formulas = sorted(
            {
                normalized
                for f in new_formulas
                if (normalized := normalize_connective(f))
            }
        )

        # Add to our collection deterministically (preserve sorted order)
        for formula in new_formulas:
            if formula not in seen:
                seen.add(formula)
                all_formulas.append(formula)
        formulas = new_formulas  # For next iteration

    return all_formulas


def enqueue_jobs(formulas: List[str], redis_client, queue_key: str, theory: str = "Propositional") -> int:
    """Enqueue formulas as JSON jobs to Redis."""
    enqueued = 0
    seen = set()
    for formula in formulas:
        stmt = sanitize_statement(formula)
        if stmt.is_empty():
            continue
        if stmt.canonical in seen:
            continue
        seen.add(stmt.canonical)
        job = {
            "job_version": 1,
            "theory": theory,
            "goal_type": stmt.ascii_pretty,
            "statement": stmt.ascii_pretty,
            "canonical": stmt.canonical,
        }
        redis_client.rpush(queue_key, json.dumps(job, ensure_ascii=False))
        enqueued += 1
        print(f"Enqueued: {stmt.ascii_pretty}")

    return enqueued


def main():
    parser = argparse.ArgumentParser(description="Generate propositional formulas and enqueue them")
    parser.add_argument("--depth", type=int, default=3, help="Maximum depth of formulas (default: 3)")
    parser.add_argument("--atoms", nargs="+", default=["p", "q"], help="Atomic propositions (default: p q)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually enqueue, just print formulas")
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                       help="Redis URL")
    parser.add_argument("--queue-key", default=os.environ.get("QUEUE_KEY", "ml:jobs"),
                       help="Redis queue key")

    args = parser.parse_args()

    # Normalize atoms
    atoms = [normalize_connective(atom) for atom in args.atoms if sanitize_statement(atom).canonical]

    print(f"Generating propositional formulas with atoms: {atoms}, max depth: {args.depth}")

    # Generate formulas
    formulas = generate_formulas_at_depth(atoms, args.depth)
    print(f"Generated {len(formulas)} formulas")

    if args.dry_run:
        print("\nGenerated formulas:")
        for i, formula in enumerate(formulas, 1):
            stmt = sanitize_statement(formula)
            if stmt.is_empty():
                continue
            print(f"{i:3d}: {stmt.ascii_pretty}")
        return

    # Connect to Redis and enqueue
    try:
        redis_client = redis.from_url(args.redis_url, decode_responses=True)

        # Test connection
        redis_client.ping()
        print(f"Connected to Redis at {args.redis_url}")

        # Enqueue jobs
        enqueued = enqueue_jobs(formulas, redis_client, args.queue_key)

        # Show queue length
        queue_len = redis_client.llen(args.queue_key)
        print(f"\nEnqueued {enqueued} jobs. Queue length: {queue_len}")

    except redis.ConnectionError as e:
        print(f"Failed to connect to Redis: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
