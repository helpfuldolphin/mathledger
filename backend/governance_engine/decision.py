# backend/governance_engine/decision.py
from dataclasses import dataclass
from typing import List

@dataclass
class Decision:
    """Represents the output of the Governance Engine for a given signal."""
    action: str
    triggering_rule: str
