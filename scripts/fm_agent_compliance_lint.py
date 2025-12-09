#!/usr/bin/env python3
"""
PHASE IV — NOT RUN IN PHASE I

Agent Compliance Linter for Field Manual Label Contract

This tool checks agent configuration files against the Field Manual label
contract index to ensure agents comply with the canonical label definitions.

Usage:
    python scripts/fm_agent_compliance_lint.py --label-index <path> --agents <path1> [<path2> ...]

ABSOLUTE SAFEGUARDS:
    - This tool is DESCRIPTIVE, not NORMATIVE
    - No modifications to agent configs
    - No inference or claims regarding uplift
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).parent.parent


def extract_labels_from_text(content: str) -> Set[str]:
    """
    Extract label references from text content.
    
    Looks for patterns like:
    - `def:slice`, `inv:monotonicity` (backtick-wrapped)
    - \ref{def:slice} (LaTeX refs)
    - "def:slice" (quoted strings)
    """
    labels: Set[str] = set()
    
    # Pattern for label-like identifiers (def:*, inv:*, sec:*, etc.)
    label_pattern = re.compile(r'\b(def|inv|sec|tab|eq|fig):([a-z0-9_-]+)\b', re.IGNORECASE)
    
    # Find all label references
    for match in label_pattern.finditer(content):
        label = f"{match.group(1).lower()}:{match.group(2).lower()}"
        labels.add(label)
    
    # Also check for LaTeX refs
    latex_ref_pattern = re.compile(r'\\(?:ref|eqref|pageref)\{([^}]+)\}')
    for match in latex_ref_pattern.finditer(content):
        label = match.group(1).lower()
        if ':' in label:  # Only add if it looks like a label
            labels.add(label)
    
    return labels


def extract_labels_from_agent_config(config_path: Path) -> Set[str]:
    """
    Extract label references from an agent config file.
    
    Supports:
    - JSON files (searches all string values)
    - Text files (markdown, prompts, etc.)
    - YAML files (basic parsing)
    """
    if not config_path.exists():
        return set()
    
    content = config_path.read_text(encoding="utf-8")
    
    # Try JSON first
    if config_path.suffix == ".json":
        try:
            data = json.loads(content)
            # Recursively extract labels from JSON structure
            labels: Set[str] = set()
            
            def extract_from_json(obj: Any) -> None:
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(key, str):
                            labels.update(extract_labels_from_text(key))
                        extract_from_json(value)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_from_json(item)
                elif isinstance(obj, str):
                    labels.update(extract_labels_from_text(obj))
            
            extract_from_json(data)
            return labels
        except json.JSONDecodeError:
            pass
    
    # Fall back to text extraction
    return extract_labels_from_text(content)


def get_fm_labels_from_contract_index(label_index: Dict[str, Any]) -> Set[str]:
    """Extract all FM labels from label contract index."""
    fm_labels: Set[str] = set()
    
    label_index_data = label_index.get("label_index", {})
    for label, data in label_index_data.items():
        if data.get("in_fm", False):
            fm_labels.add(label)
    
    return fm_labels


def lint_agent_compliance(
    label_contract_index: Dict[str, Any],
    agent_config_paths: List[Path],
) -> Dict[str, Any]:
    """
    Lint agent configs for compliance with Field Manual label contract.
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Args:
        label_contract_index: Label contract index from build_label_contract_index()
        agent_config_paths: List of paths to agent config files
    
    Returns:
        {
            "schema_version": "1.0.0",
            "agents_checked": [
                {"agent": "<path>", "labels_found": [...], "missing_labels": [...], "superfluous_labels": [...]},
                ...
            ],
            "missing_labels": List[str],  # Labels used by agents but not in FM
            "superfluous_labels": List[str],  # FM labels not used by any agent
            "status": "OK" | "ATTENTION" | "BLOCK",
            "neutral_notes": List[str]
        }
    """
    # Get FM labels from contract index
    fm_labels = get_fm_labels_from_contract_index(label_contract_index)
    
    # Check each agent config
    agents_checked: List[Dict[str, Any]] = []
    all_agent_labels: Set[str] = set()
    
    for config_path in agent_config_paths:
        agent_labels = extract_labels_from_agent_config(config_path)
        all_agent_labels.update(agent_labels)
        
        # Find missing and superfluous labels for this agent
        missing = sorted(agent_labels - fm_labels)
        superfluous = sorted(fm_labels - agent_labels)
        
        agents_checked.append({
            "agent": str(config_path),
            "labels_found": sorted(agent_labels),
            "missing_labels": missing,
            "superfluous_labels": superfluous,
        })
    
    # Aggregate missing and superfluous labels
    global_missing = sorted(all_agent_labels - fm_labels)
    global_superfluous = sorted(fm_labels - all_agent_labels)
    
    # Determine status
    if global_missing:
        status = "BLOCK"
    elif global_superfluous:
        status = "ATTENTION"
    else:
        status = "OK"
    
    # Build neutral notes
    neutral_notes: List[str] = []
    if global_missing:
        neutral_notes.append(f"{len(global_missing)} label(s) used by agents but not defined in Field Manual")
    if global_superfluous:
        neutral_notes.append(f"{len(global_superfluous)} Field Manual label(s) not referenced by any agent")
    if not global_missing and not global_superfluous:
        neutral_notes.append("All agent labels align with Field Manual definitions")
    
    return {
        "schema_version": "1.0.0",
        "agents_checked": agents_checked,
        "missing_labels": global_missing,
        "superfluous_labels": global_superfluous,
        "status": status,
        "neutral_notes": neutral_notes,
    }


def main() -> int:
    """Main entry point for agent compliance linter."""
    parser = argparse.ArgumentParser(
        description="Agent Compliance Linter for Field Manual Label Contract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--label-index",
        type=Path,
        required=True,
        help="Path to label contract index JSON",
    )
    parser.add_argument(
        "--agents",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to agent config files to check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for compliance report JSON",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON only",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE IV — Agent Compliance Linter")
    print("=" * 70)
    
    try:
        # Load label contract index
        if not args.label_index.exists():
            print(f"ERROR: Label index not found: {args.label_index}")
            return 2
        
        with open(args.label_index, "r", encoding="utf-8") as f:
            label_index = json.load(f)
        
        # Check agent configs
        agent_paths = [Path(p) for p in args.agents]
        compliance = lint_agent_compliance(label_index, agent_paths)
        
        # Output results
        if args.json:
            print(json.dumps(compliance, indent=2))
        else:
            status_icon = {
                "OK": "✅",
                "ATTENTION": "⚠️",
                "BLOCK": "🔴",
            }.get(compliance["status"], "?")
            
            print(f"\n{status_icon} Compliance Status: {compliance['status']}")
            print(f"\nAgents Checked: {len(compliance['agents_checked'])}")
            
            for agent_info in compliance["agents_checked"]:
                print(f"\n  Agent: {Path(agent_info['agent']).name}")
                print(f"    Labels found: {len(agent_info['labels_found'])}")
                if agent_info["missing_labels"]:
                    print(f"    ⚠️  Missing labels: {len(agent_info['missing_labels'])}")
                    for label in agent_info["missing_labels"][:5]:
                        print(f"      - {label}")
                    if len(agent_info["missing_labels"]) > 5:
                        print(f"      ... and {len(agent_info['missing_labels']) - 5} more")
                if agent_info["superfluous_labels"]:
                    print(f"    ℹ️  Superfluous labels: {len(agent_info['superfluous_labels'])}")
            
            if compliance["missing_labels"]:
                print(f"\n⚠️  Global Missing Labels ({len(compliance['missing_labels'])}):")
                for label in compliance["missing_labels"][:10]:
                    print(f"    - {label}")
                if len(compliance["missing_labels"]) > 10:
                    print(f"    ... and {len(compliance['missing_labels']) - 10} more")
            
            if compliance["superfluous_labels"]:
                print(f"\nℹ️  Global Superfluous Labels ({len(compliance['superfluous_labels'])}):")
                for label in compliance["superfluous_labels"][:10]:
                    print(f"    - {label}")
                if len(compliance["superfluous_labels"]) > 10:
                    print(f"    ... and {len(compliance['superfluous_labels']) - 10} more")
            
            print(f"\nNotes:")
            for note in compliance["neutral_notes"]:
                print(f"  - {note}")
        
        # Save output if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(compliance, f, indent=2, sort_keys=True)
            if not args.json:
                print(f"\n📄 Compliance report written to: {args.output}")
        
        print("=" * 70)
        
        # Return code based on status
        if compliance["status"] == "BLOCK":
            return 2
        elif compliance["status"] == "ATTENTION":
            return 1
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

