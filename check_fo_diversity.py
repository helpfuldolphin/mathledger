#!/usr/bin/env python3
"""
Check candidate_hash diversity in FO cycle logs.
Run this on results/fo_baseline.jsonl or results/fo_rfl.jsonl to verify diversity.
"""
import json
import sys
from pathlib import Path
from collections import Counter

def analyze_diversity(file_path: Path, window_size: int = 50):
    """Analyze candidate_hash diversity in a FO cycle log."""
    if not file_path.exists():
        print(f"ERROR: {file_path} not found")
        sys.exit(1)
    
    cycles = []
    candidate_hashes = []
    candidate_texts = []
    abstention_flags = []
    statuses = []
    
    with open(file_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cycles.append(data.get('cycle', -1))
                deriv = data.get('derivation', {})
                candidate_hashes.append(deriv.get('candidate_hash', 'none'))
                candidate_texts.append(deriv.get('candidate_text', 'none'))
                abstention_flags.append(data.get('abstention', False))
                statuses.append(data.get('status', 'unknown'))
    
    if not cycles:
        print(f"ERROR: {file_path} is empty")
        sys.exit(1)
    
    total_cycles = len(cycles)
    unique_hashes = len(set(candidate_hashes))
    unique_texts = len(set(candidate_texts))
    abstention_rate = sum(abstention_flags) / total_cycles if total_cycles > 0 else 0.0
    verified_count = sum(1 for s in statuses if s == 'verified')
    abstain_count = sum(1 for s in statuses if s == 'abstain')
    
    print("=" * 60)
    print(f"DIVERSITY ANALYSIS: {file_path.name}")
    print("=" * 60)
    print(f"Total cycles: {total_cycles}")
    print(f"Unique candidate_hashes: {unique_hashes}")
    print(f"Unique candidate_texts: {unique_texts}")
    print(f"Abstention rate: {abstention_rate:.3f} ({abstention_rate*100:.1f}%)")
    print(f"Status breakdown:")
    print(f"  Verified: {verified_count} ({verified_count/total_cycles*100:.1f}%)")
    print(f"  Abstain:  {abstain_count} ({abstain_count/total_cycles*100:.1f}%)")
    print(f"  Error:    {total_cycles - verified_count - abstain_count}")
    print()
    
    # Check for pathology
    if unique_hashes == 1:
        print("❌ PATHOLOGY DETECTED: All cycles have the same candidate_hash")
        print("   This indicates the derivation is frozen - no diversity.")
        print("   Most common hash:", Counter(candidate_hashes).most_common(1)[0][0])
    elif unique_hashes < 10:
        print(f"⚠️  LOW DIVERSITY: Only {unique_hashes} unique candidate_hashes")
        print("   This may indicate limited derivation search space.")
        print("   Most common hashes:")
        for hash_val, count in Counter(candidate_hashes).most_common(5):
            print(f"     {hash_val[:16]}...: {count} times ({count/total_cycles*100:.1f}%)")
    else:
        print(f"✅ GOOD DIVERSITY: {unique_hashes} unique candidate_hashes")
        print("   Derivation is producing diverse statements.")
    
    print()
    
    # Window analysis
    print(f"Window Analysis (first {window_size} cycles):")
    first_window_hashes = candidate_hashes[:window_size]
    first_window_unique = len(set(first_window_hashes))
    first_window_abstention = sum(abstention_flags[:window_size]) / window_size
    
    print(f"  Unique hashes: {first_window_unique}/{window_size}")
    print(f"  Abstention rate: {first_window_abstention:.3f} ({first_window_abstention*100:.1f}%)")
    
    if total_cycles > window_size:
        print(f"\nWindow Analysis (last {window_size} cycles):")
        last_window_hashes = candidate_hashes[-window_size:]
        last_window_unique = len(set(last_window_hashes))
        last_window_abstention = sum(abstention_flags[-window_size:]) / window_size
        
        print(f"  Unique hashes: {last_window_unique}/{window_size}")
        print(f"  Abstention rate: {last_window_abstention:.3f} ({last_window_abstention*100:.1f}%)")
        
        if last_window_abstention < first_window_abstention - 0.05:
            print(f"  ✅ Abstention decreased: {first_window_abstention:.3f} → {last_window_abstention:.3f}")
        elif last_window_abstention > first_window_abstention + 0.05:
            print(f"  ⚠️  Abstention increased: {first_window_abstention:.3f} → {last_window_abstention:.3f}")
    
    print()
    print("=" * 60)
    
    # Return summary
    return {
        'total_cycles': total_cycles,
        'unique_hashes': unique_hashes,
        'unique_texts': unique_texts,
        'abstention_rate': abstention_rate,
        'verified_count': verified_count,
        'abstain_count': abstain_count,
        'pathology': unique_hashes == 1,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_fo_diversity.py <path_to_jsonl> [window_size]")
        print("Example: python check_fo_diversity.py results/fo_rfl.jsonl 50")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    analyze_diversity(file_path, window_size)

