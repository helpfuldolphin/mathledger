# scripts/observatory/history.py
"""
Implements functions for historical analysis and governance summaries
of slice integrity data.
"""
from typing import Sequence, Dict, Any, List
from collections import OrderedDict

# Assuming data models are in a place that can be imported
# For this self-contained script, we'll redefine them conceptually
from scripts.observatory.data_models import SliceIdentityCard, ReconciliationResult

def build_slice_identity_history(cards: Sequence[SliceIdentityCard]) -> Dict[str, Any]:
    """
    Processes a sequence of SliceIdentityCards to build a historical summary.
    """
    history: Dict[str, Dict[str, Any]] = {}
    all_slice_names: set[str] = set()

    # Sort cards by timestamp to process them chronologically
    sorted_cards = sorted(cards, key=lambda c: c['timestamp_utc'])

    for card in sorted_cards:
        name = card['slice_name']
        all_slice_names.add(name)
        if name not in history:
            history[name] = {
                "first_seen_at": card['timestamp_utc'],
                "last_seen_at": card['timestamp_utc'],
                "number_of_drift_events": 0,
                "last_integrity_status": "PASS",
                "is_missing_prereg": True,
                "is_missing_manifest": True,
            }

        # Update historical data
        history[name]['last_seen_at'] = card['timestamp_utc']
        history[name]['last_integrity_status'] = card['integrity_status']
        if card['integrity_status'] in ["WARN", "FAIL"]:
            history[name]['number_of_drift_events'] += 1
        
        if card['has_prereg']:
            history[name]['is_missing_prereg'] = False
        if card['has_manifest_entry']:
            history[name]['is_missing_manifest'] = False

    # Calculate global summary
    total_slices = len(all_slice_names)
    slices_with_drift = sum(1 for s in history.values() if s['number_of_drift_events'] > 0)
    slices_missing_prereg = sum(1 for s in history.values() if s['is_missing_prereg'])
    slices_missing_manifest = sum(1 for s in history.values() if s['is_missing_manifest'])

    return {
        "slice_history": history,
        "global_summary": {
            "total_slices": total_slices,
            "slices_with_drift": slices_with_drift,
            "slices_missing_prereg": slices_missing_prereg,
            "slices_missing_manifest": slices_missing_manifest,
        }
    }

def summarize_slice_hash_for_governance(result: ReconciliationResult) -> Dict[str, Any]:
    """
    Creates a high-level summary suitable for a governance pipeline.
    """
    any_fail = any(s['integrity_status'] == 'FAIL' for s in result['slices'])
    any_warn = any(s['integrity_status'] == 'WARN' for s in result['slices'])
    
    status = "OK"
    if any_warn:
        status = "WARN"
    if any_fail:
        status = "FAIL"
        
    drift_slices = [
        s['slice_name'] for s in result['slices'] if s['integrity_status'] != 'PASS'
    ]
    
    # A slice is "accounted for" if it has both a prereg and manifest entry
    all_accounted = all(s['has_prereg'] and s['has_manifest_entry'] for s in result['slices'])

    return OrderedDict([
        ("status", status),
        ("all_slices_accounted_for", all_accounted),
        ("any_fail_detected", any_fail),
        ("drift_slices", sorted(list(set(drift_slices)))),
    ])
