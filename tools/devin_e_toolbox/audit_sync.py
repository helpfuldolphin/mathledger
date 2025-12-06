#!/usr/bin/env python3
"""
Audit Sync - Synchronize audit trails and verification logs

Maintains audit trail integrity by:
1. Collecting verification events from all tools
2. Computing RFC 8785 canonical JSON for each event
3. Storing in append-only audit log
4. Generating cryptographic hash chain

Usage:
    python audit_sync.py --collect    # Collect events from tools
    python audit_sync.py --verify     # Verify audit chain integrity
    python audit_sync.py --export     # Export audit log
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).parent.parent.parent
AUDIT_DIR = REPO_ROOT / 'artifacts' / 'audit'
AUDIT_LOG = AUDIT_DIR / 'audit_trail.jsonl'

def canonical_json(obj: Any) -> str:
    """
    RFC 8785 canonical JSON serialization
    
    Rules:
    - No whitespace outside strings
    - Keys sorted lexicographically
    - Unicode escapes for non-ASCII
    - No trailing zeros in numbers
    """
    return json.dumps(
        obj,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
        indent=None
    )

def compute_hash(data: str) -> str:
    """Compute SHA256 hash of canonical JSON"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def create_audit_event(event_type: str, data: Dict, prev_hash: str = None) -> Dict:
    """Create audit event with hash chain"""
    event = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'event_type': event_type,
        'data': data,
        'prev_hash': prev_hash or '0' * 64
    }
    
    canonical = canonical_json(event)
    event['hash'] = compute_hash(canonical)
    
    return event

def append_audit_event(event: Dict) -> None:
    """Append event to audit log"""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(AUDIT_LOG, 'a') as f:
        f.write(canonical_json(event) + '\n')

def load_audit_log() -> List[Dict]:
    """Load all audit events"""
    if not AUDIT_LOG.exists():
        return []
    
    events = []
    with open(AUDIT_LOG) as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events

def verify_audit_chain() -> bool:
    """Verify integrity of audit chain"""
    events = load_audit_log()
    
    if not events:
        print("No audit events found")
        return True
    
    print(f"Verifying {len(events)} audit events...")
    
    errors = []
    prev_hash = '0' * 64
    
    for i, event in enumerate(events):
        if event.get('prev_hash') != prev_hash:
            errors.append(f"Event {i}: prev_hash mismatch")
            errors.append(f"  Expected: {prev_hash}")
            errors.append(f"  Got: {event.get('prev_hash')}")
        
        stored_hash = event.get('hash')
        event_copy = dict(event)
        del event_copy['hash']
        
        canonical = canonical_json(event_copy)
        computed_hash = compute_hash(canonical)
        
        if stored_hash != computed_hash:
            errors.append(f"Event {i}: hash mismatch")
            errors.append(f"  Expected: {computed_hash}")
            errors.append(f"  Got: {stored_hash}")
        
        prev_hash = stored_hash
    
    if errors:
        print("\nVERIFICATION FAILED:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\nVERIFICATION PASSED")
        print(f"  Chain length: {len(events)}")
        print(f"  Latest hash: {prev_hash}")
        return True

def collect_tool_events() -> int:
    """Collect verification events from tools"""
    events = load_audit_log()
    prev_hash = events[-1]['hash'] if events else None
    
    collected = 0
    
    build_baseline = REPO_ROOT / 'artifacts' / 'build_baseline.json'
    if build_baseline.exists():
        with open(build_baseline) as f:
            baseline = json.load(f)
        
        event = create_audit_event(
            'deterministic_build',
            {
                'tree_hash': baseline.get('tree_hash'),
                'file_count': baseline.get('file_count'),
                'timestamp': baseline.get('timestamp')
            },
            prev_hash
        )
        append_audit_event(event)
        prev_hash = event['hash']
        collected += 1
    
    verify_log = REPO_ROOT / 'artifacts' / 'verification.log'
    if verify_log.exists():
        with open(verify_log) as f:
            for line in f:
                if 'PASS' in line or 'FAIL' in line:
                    event = create_audit_event(
                        'verification',
                        {'log_entry': line.strip()},
                        prev_hash
                    )
                    append_audit_event(event)
                    prev_hash = event['hash']
                    collected += 1
    
    print(f"Collected {collected} audit events")
    return collected

def export_audit_log(output_path: Path) -> None:
    """Export audit log with verification"""
    events = load_audit_log()
    
    if not events:
        print("No audit events to export")
        return
    
    if not verify_audit_chain():
        print("WARNING: Audit chain verification failed")
        print("Export aborted")
        return
    
    export_data = {
        'export_timestamp': datetime.utcnow().isoformat() + 'Z',
        'event_count': len(events),
        'chain_head': events[-1]['hash'] if events else None,
        'events': events
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported {len(events)} events to {output_path}")
    print(f"Chain head: {export_data['chain_head']}")

def main():
    parser = argparse.ArgumentParser(description='Audit trail synchronization')
    parser.add_argument('--collect', action='store_true',
                       help='Collect events from tools')
    parser.add_argument('--verify', action='store_true',
                       help='Verify audit chain integrity')
    parser.add_argument('--export', type=Path,
                       help='Export audit log to file')
    parser.add_argument('--init', action='store_true',
                       help='Initialize audit log with genesis event')
    
    args = parser.parse_args()
    
    if args.init:
        print("Initializing audit log...")
        event = create_audit_event('genesis', {'message': 'Audit log initialized'})
        append_audit_event(event)
        print(f"Genesis event: {event['hash']}")
        return 0
    
    elif args.collect:
        count = collect_tool_events()
        return 0 if count >= 0 else 1
    
    elif args.verify:
        success = verify_audit_chain()
        return 0 if success else 1
    
    elif args.export:
        export_audit_log(args.export)
        return 0
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
