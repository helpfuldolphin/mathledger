"""
Lean Tactic Parser

Parses Lean verification output to extract tactic usage information.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

import re
from typing import Dict, Any, List


# Tactic patterns (regex)
TACTIC_PATTERNS = [
    r'\bapply\b',
    r'\brw\b',
    r'\bsimp\b',
    r'\bring\b',
    r'\bexact\b',
    r'\bintro\b',
    r'\bcases\b',
    r'\binduction\b',
    r'\brefl\b',
    r'\bconv\b',
    r'\bnorm_num\b',
    r'\bomega\b',
    r'\btauto\b',
    r'\bdecide\b',
]


def parse_tactics_from_output(output: str) -> Dict[str, Any]:
    """Parse Lean output to extract tactic information.
    
    Args:
        output: Lean stdout/stderr output
    
    Returns:
        Dict with tactic_count, tactic_depth, and tactics list
    """
    
    tactics = []
    tactic_counts = {}
    
    # Extract tactics using regex patterns
    for pattern in TACTIC_PATTERNS:
        matches = re.findall(pattern, output)
        if matches:
            tactic_name = pattern.strip(r'\b')
            tactics.extend([tactic_name] * len(matches))
            tactic_counts[tactic_name] = len(matches)
    
    # Estimate tactic depth (heuristic: count nested proof blocks)
    depth = output.count("begin") + output.count("by")
    
    return {
        "tactic_count": len(tactics),
        "tactic_depth": depth,
        "tactics": tactics,
        "tactic_counts": tactic_counts,
    }
