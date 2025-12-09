# PHASE IV — EVIDENCE PACK INTEGRATION
"""
Handles the integration of governance artifacts into evidence packs.
"""
import json
from typing import Dict, Any

def attach_replay_governance_to_evidence(
    evidence_pack: Dict[str, Any],
    replay_snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Attaches a replay governance snapshot to an evidence pack in a
    non-mutating, deterministic way.
    """
    # Create a deep copy to ensure non-mutation
    new_pack = json.loads(json.dumps(evidence_pack))
    
    # Add the governance tile
    new_pack["replay_governance"] = replay_snapshot
    
    # Return a deterministically serialized string for consistency
    return json.dumps(new_pack, sort_keys=True, indent=2)
