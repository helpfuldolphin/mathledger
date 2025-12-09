"""
Curriculum Diff Generator

Produces human-readable diagnostic reports comparing two curriculum configurations.
Used for pre-commit validation, post-mortem analysis, and curriculum review.
"""

from typing import Dict, Any, List
from backend.frontier.curriculum import CurriculumSystem, load as load_curriculum


def generate_curriculum_diff(old_system: CurriculumSystem, new_system: CurriculumSystem) -> Dict[str, Any]:
    """
    Generates a human-readable diff between two curriculum systems.
    
    Args:
        old_system: The baseline curriculum system
        new_system: The modified curriculum system
        
    Returns:
        A dictionary containing:
        - param_diffs: List of parameter changes
        - gate_diffs: List of gate threshold changes
        - structure_diffs: List of structural changes (added/removed slices)
        - teacher_facing_summary: Human-readable summary
    """
    diff = {
        "param_diffs": [],
        "gate_diffs": [],
        "structure_diffs": [],
        "teacher_facing_summary": ""
    }

    old_slices = {s.name: s for s in old_system.slices}
    new_slices = {s.name: s for s in new_system.slices}

    added_slices = new_slices.keys() - old_slices.keys()
    removed_slices = old_slices.keys() - new_slices.keys()
    common_slices = old_slices.keys() & new_slices.keys()

    # Structure diffs
    if added_slices:
        diff["structure_diffs"].append(f"Added slices: {sorted(list(added_slices))}")
    if removed_slices:
        diff["structure_diffs"].append(f"Removed slices: {sorted(list(removed_slices))}")

    # Param and gate diffs for common slices
    for name in sorted(list(common_slices)):
        old_s, new_s = old_slices[name], new_slices[name]
        
        # Param Diffs
        all_param_keys = old_s.params.keys() | new_s.params.keys()
        for key in sorted(all_param_keys):
            old_val = old_s.params.get(key)
            new_val = new_s.params.get(key)
            if old_val != new_val:
                diff["param_diffs"].append(f"{name}.params.{key}: {old_val} -> {new_val}")
        
        # Gate Diffs
        old_gates = old_s.gates.to_dict()
        new_gates = new_s.gates.to_dict()
        for gate_name in ["coverage", "abstention", "velocity", "caps"]:
            old_spec = old_gates.get(gate_name, {})
            new_spec = new_gates.get(gate_name, {})
            all_gate_keys = old_spec.keys() | new_spec.keys()
            for key in sorted(all_gate_keys):
                old_val = old_spec.get(key)
                new_val = new_spec.get(key)
                if old_val != new_val:
                    diff["gate_diffs"].append(f"{name}.gates.{gate_name}.{key}: {old_val} -> {new_val}")

    # System-level diffs
    if old_system.version != new_system.version:
        diff["structure_diffs"].append(f"Version changed: {old_system.version} -> {new_system.version}")
    
    if old_system.invariants != new_system.invariants:
        diff["structure_diffs"].append(f"Invariants changed: {old_system.invariants} -> {new_system.invariants}")

    # Generate Summary
    summary_lines = []
    if not any([diff["structure_diffs"], diff["param_diffs"], diff["gate_diffs"]]):
        summary_lines.append("No functional change detected.")
    else:
        summary_lines.append("Curriculum has been modified:")
        if diff["structure_diffs"]:
            summary_lines.append("\nStructural Changes:")
            summary_lines.extend(f"  - {d}" for d in diff["structure_diffs"])
        if diff["param_diffs"]:
            summary_lines.append("\nParameter Changes:")
            summary_lines.extend(f"  - {d}" for d in diff["param_diffs"])
        if diff["gate_diffs"]:
            summary_lines.append("\nGate Threshold Changes:")
            summary_lines.extend(f"  - {d}" for d in diff["gate_diffs"])
    
    diff["teacher_facing_summary"] = "\n".join(summary_lines)
    return diff


def main():
    """CLI interface for curriculum diff generator."""
    import sys
    import yaml

    if len(sys.argv) != 4:
        print("Usage: python -m backend.tools.curriculum_differ <old_config.yaml> <new_config.yaml> <slug>")
        sys.exit(1)

    old_config_path = sys.argv[1]
    new_config_path = sys.argv[2]
    slug = sys.argv[3]

    with open(old_config_path, 'r', encoding='utf-8') as f:
        old_config = yaml.safe_load(f)
    
    with open(new_config_path, 'r', encoding='utf-8') as f:
        new_config = yaml.safe_load(f)

    old_sys = CurriculumSystem.from_config(slug, old_config)
    new_sys = CurriculumSystem.from_config(slug, new_config)

    diff_report = generate_curriculum_diff(old_sys, new_sys)
    print(diff_report["teacher_facing_summary"])


if __name__ == "__main__":
    main()
