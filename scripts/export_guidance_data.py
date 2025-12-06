#!/usr/bin/env python3
"""Export guidance data from sealed blocks for policy training."""

import os
import csv
import json
import psycopg
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

def _require_database_url() -> str:
    """Return DATABASE_URL or raise if not set (zero-trust policy)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "[FATAL] DATABASE_URL environment variable is not set. "
            "Set it explicitly before running this script."
        )
    return url


DATABASE_URL = _require_database_url()

def get_sealed_blocks(conn) -> List[Dict[str, Any]]:
    """Get sealed blocks with their statements and proofs."""
    cur = conn.cursor()

    # Get blocks with their statements and proof information
    cur.execute("""
        SELECT
            b.id,
            b.block_number,
            b.merkle_root,
            b.header,
            b.proof_count,
            b.created_at,
            s.text,
            s.normalized_text,
            s.hash as statement_hash,
            s.derivation_depth,
            p.method,
            p.status,
            p.created_at as proof_created_at
        FROM blocks b
        LEFT JOIN statements s ON s.id IN (
            SELECT jsonb_array_elements_text(b.statements)::bigint
        )
        LEFT JOIN proofs p ON p.statement_id = s.id
        WHERE b.proof_count > 0
        ORDER BY b.block_number DESC, s.derivation_depth ASC, p.created_at ASC
    """)

    blocks_data = {}
    for row in cur.fetchall():
        block_id = row[0]
        if block_id not in blocks_data:
            blocks_data[block_id] = {
                'block_number': row[1],
                'merkle_root': row[2],
                'header': row[3],
                'proof_count': row[4],
                'created_at': row[5],
                'statements': []
            }

        if row[6]:  # statement text exists
            statement = {
                'text': row[6],
                'normalized_text': row[7],
                'hash': row[8],
                'derivation_depth': row[9],
                'method': row[10],
                'proof_status': row[11],
                'proof_created_at': row[12]
            }
            blocks_data[block_id]['statements'].append(statement)

    return list(blocks_data.values())

def generate_positives(blocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate positive examples from observed successful actions."""
    positives = []

    for block in blocks_data:
        for stmt in block['statements']:
            if stmt['proof_status'] == 'success':
                # Feature: statement context (depth, method, etc.)
                features = {
                    'derivation_depth': stmt['derivation_depth'] or 0,
                    'method': stmt['method'] or 'unknown',
                    'block_proof_count': block['proof_count'],
                    'statement_length': len(stmt['normalized_text']) if stmt['normalized_text'] else 0,
                    'has_implication': '->' in (stmt['normalized_text'] or ''),
                    'has_conjunction': '/\\' in (stmt['normalized_text'] or ''),
                    'has_disjunction': '\\/' in (stmt['normalized_text'] or ''),
                    'complexity_score': calculate_complexity(stmt['normalized_text'] or '')
                }

                # Action: the successful method/approach used
                action = {
                    'method': stmt['method'] or 'direct',
                    'success': True,
                    'target': stmt['hash']
                }

                positives.append({
                    'features': features,
                    'action': action,
                    'block_number': block['block_number'],
                    'statement_hash': stmt['hash']
                })

    return positives

def generate_negatives(blocks_data: List[Dict[str, Any]], num_negatives: int = None) -> List[Dict[str, Any]]:
    """Generate negative examples by sampling unsuccessful or alternative actions."""
    if num_negatives is None:
        num_negatives = len(blocks_data) * 2  # 2x negatives per positive

    negatives = []

    # Get all unique methods used
    methods = set()
    for block in blocks_data:
        for stmt in block['statements']:
            if stmt['method']:
                methods.add(stmt['method'])

    if not methods:
        methods = {'direct', 'modus_ponens', 'contrapositive', 'axiom'}

    # Generate negative examples
    for _ in range(num_negatives):
        # Sample a random block and statement
        block = random.choice(blocks_data) if blocks_data else None
        if not block or not block['statements']:
            continue

        stmt = random.choice(block['statements'])

        # Create features similar to positives
        features = {
            'derivation_depth': stmt['derivation_depth'] or 0,
            'method': stmt['method'] or 'unknown',
            'block_proof_count': block['proof_count'],
            'statement_length': len(stmt['normalized_text']) if stmt['normalized_text'] else 0,
            'has_implication': '->' in (stmt['normalized_text'] or ''),
            'has_conjunction': '/\\' in (stmt['normalized_text'] or ''),
            'has_disjunction': '\\/' in (stmt['normalized_text'] or ''),
            'complexity_score': calculate_complexity(stmt['normalized_text'] or '')
        }

        # Action: a different method (negative example)
        available_methods = list(methods - {stmt['method'] or 'direct'})
        if not available_methods:
            available_methods = ['modus_ponens', 'contrapositive']  # fallback
        alternative_method = random.choice(available_methods)
        action = {
            'method': alternative_method,
            'success': False,
            'target': stmt['hash']
        }

        negatives.append({
            'features': features,
            'action': action,
            'block_number': block['block_number'],
            'statement_hash': stmt['hash']
        })

    return negatives

def calculate_complexity(text: str) -> int:
    """Calculate complexity score based on statement structure."""
    if not text:
        return 0

    score = 0
    score += text.count('->') * 2  # Implications
    score += text.count('/\\') * 1  # Conjunctions
    score += text.count('\\/') * 1  # Disjunctions
    score += text.count('(') * 1  # Nesting
    score += len(text) // 10  # Length factor

    return min(score, 20)  # Cap at 20

def write_csv_file(filename: str, data: List[Dict[str, Any]], split: str):
    """Write data to CSV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = [
            'split', 'features_json', 'action_json', 'block_number',
            'statement_hash', 'derivation_depth', 'method', 'success'
        ]
        writer.writerow(header)

        # Write data rows
        for item in data:
            row = [
                split,
                json.dumps(item['features']),
                json.dumps(item['action']),
                item['block_number'],
                item['statement_hash'],
                item['features']['derivation_depth'],
                item['action']['method'],
                item['action']['success']
            ]
            writer.writerow(row)

def split_data(positives: List[Dict[str, Any]], negatives: List[Dict[str, Any]],
               train_ratio: float = 0.8) -> Tuple[List, List, List, List]:
    """Split data into train/validation sets."""
    # Combine and shuffle
    all_data = positives + negatives
    random.shuffle(all_data)

    # Split
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Separate by label
    train_pos = [d for d in train_data if d['action']['success']]
    train_neg = [d for d in train_data if not d['action']['success']]
    val_pos = [d for d in val_data if d['action']['success']]
    val_neg = [d for d in val_data if not d['action']['success']]

    return train_pos, train_neg, val_pos, val_neg

def main():
    """Main function to export guidance data."""
    print("Exporting guidance data from sealed blocks...")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            # Get sealed blocks data
            print("Fetching sealed blocks data...")
            blocks_data = get_sealed_blocks(conn)
            print(f"Found {len(blocks_data)} sealed blocks")

            if not blocks_data:
                print("No sealed blocks found. Creating synthetic data...")
                # Create minimal synthetic data for testing
                blocks_data = [{
                    'block_number': 1,
                    'merkle_root': 'test123',
                    'header': {},
                    'proof_count': 1,
                    'created_at': datetime.now(),
                    'statements': [{
                        'text': 'P -> (Q -> P)',
                        'normalized_text': 'P->(Q->P)',
                        'hash': 'test_hash',
                        'derivation_depth': 0,
                        'method': 'axiom',
                        'proof_status': 'success',
                        'proof_created_at': datetime.now()
                    }]
                }]

            # Generate positive and negative examples
            print("Generating positive examples...")
            positives = generate_positives(blocks_data)
            print(f"Generated {len(positives)} positive examples")

            print("Generating negative examples...")
            negatives = generate_negatives(blocks_data)
            print(f"Generated {len(negatives)} negative examples")

            # Split data
            print("Splitting data into train/validation sets...")
            train_pos, train_neg, val_pos, val_neg = split_data(positives, negatives)

            # Write CSV files
            print("Writing training data...")
            write_csv_file("artifacts/guidance/train.csv", train_pos + train_neg, "train")
            write_csv_file("artifacts/guidance/val.csv", val_pos + val_neg, "val")

            # Print summary
            print(f"\nData export summary:")
            print(f"  Total positives: {len(positives)}")
            print(f"  Total negatives: {len(negatives)}")
            print(f"  Train set: {len(train_pos + train_neg)} examples")
            print(f"  Validation set: {len(val_pos + val_neg)} examples")
            print(f"  Files written:")
            print(f"    - artifacts/guidance/train.csv")
            print(f"    - artifacts/guidance/val.csv")

    except Exception as e:
        print(f"Error exporting guidance data: {e}")
        raise

if __name__ == "__main__":
    main()
