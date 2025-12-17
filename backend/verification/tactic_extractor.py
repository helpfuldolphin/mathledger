# REAL-READY
"""
Tactic Extractor for Lean Verification

Extracts tactic information from Lean output.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: REAL-READY
"""

from __future__ import annotations

import re
from typing import Dict, List, Any


def extract_tactics_from_output(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Extract tactic information from Lean output.
    
    Args:
        stdout: Standard output from Lean
        stderr: Standard error from Lean
    
    Returns:
        Dict with tactic information:
        - tactics: List of tactic names
        - tactic_counts: Dict mapping tactic name to count
        - tactic_depth: Estimated tactic depth
    """
    
    tactics = []
    tactic_counts = {}
    
    # Combine stdout and stderr
    combined_output = stdout + "\n" + stderr
    
    # Pattern 1: Tactic trace format: [tactic.NAME]
    trace_pattern = r'\[tactic\.(\w+)\]'
    trace_matches = re.findall(trace_pattern, combined_output)
    
    for tactic in trace_matches:
        tactics.append(tactic)
        tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
    
    # Pattern 2: Tactic application format: "applying tactic NAME"
    apply_pattern = r'applying tactic (\w+)'
    apply_matches = re.findall(apply_pattern, combined_output, re.IGNORECASE)
    
    for tactic in apply_matches:
        tactics.append(tactic)
        tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
    
    # Pattern 3: Common tactic keywords in proof text
    # (This is a heuristic and may have false positives)
    common_tactics = [
        "intro", "intros", "apply", "exact", "rw", "rewrite",
        "simp", "ring", "linarith", "omega", "decide",
        "split", "left", "right", "constructor",
        "cases", "induction", "by_cases",
        "have", "suffices", "calc",
    ]
    
    for tactic in common_tactics:
        # Match whole word only
        pattern = r'\b' + tactic + r'\b'
        matches = re.findall(pattern, combined_output, re.IGNORECASE)
        
        if matches:
            count = len(matches)
            tactics.extend([tactic] * count)
            tactic_counts[tactic] = tactic_counts.get(tactic, 0) + count
    
    # Estimate tactic depth (heuristic: count nested tactic blocks)
    tactic_depth = estimate_tactic_depth(combined_output)
    
    return {
        "tactics": tactics,
        "tactic_counts": tactic_counts,
        "tactic_depth": tactic_depth,
    }


def estimate_tactic_depth(output: str) -> int:
    """
    Estimate tactic depth from output.
    
    This is a heuristic based on indentation or nesting indicators.
    
    Args:
        output: Combined stdout and stderr
    
    Returns:
        Estimated tactic depth
    """
    
    # Heuristic 1: Count maximum indentation level
    max_indent = 0
    for line in output.split("\n"):
        # Count leading spaces
        indent = len(line) - len(line.lstrip())
        if indent > max_indent:
            max_indent = indent
    
    # Assume 2 spaces per indentation level
    depth_from_indent = max_indent // 2
    
    # Heuristic 2: Count nested "by" keywords
    by_count = output.count(" by ")
    
    # Return maximum of heuristics
    return max(depth_from_indent, by_count, 1)
