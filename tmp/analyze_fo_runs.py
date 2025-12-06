import json
import sys
from pathlib import Path

def analyze_jsonl_file(file_path: Path):
    if not file_path.exists():
        return f"File not found: {file_path}", None

    if file_path.stat().st_size == 0:
        return f"File is empty: {file_path}", None

    line_count = 0
    cycles = []
    modes = set()
    abstentions = []
    has_roots_ht = True
    has_roots_rt = True
    has_roots_ut = True

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line_count += 1
                try:
                    data = json.loads(line)
                    cycles.append(data.get("cycle"))
                    modes.add(data.get("mode"))
                    abstentions.append(data.get("abstention", False))

                    if "roots" not in data:
                        has_roots_ht = False
                        has_roots_rt = False
                        has_roots_ut = False
                    else:
                        if "h_t" not in data["roots"]: has_roots_ht = False
                        if "r_t" not in data["roots"]: has_roots_rt = False
                        if "u_t" not in data["roots"]: has_roots_ut = False

                except json.JSONDecodeError:
                    return f"Invalid JSON on line {line_num + 1} in {file_path}", None
    except Exception as e:
        return f"Error reading file {file_path}: {e}", None

    # Sanity checks
    if not cycles:
        return f"No cycle data found in {file_path}", None

    min_cycle = min(cycles) if cycles else None
    max_cycle = max(cycles) if cycles else None
    
    # Check if cycles are sequential from 0 to line_count - 1
    # Assuming cycles are integers
    expected_cycles = list(range(line_count))
    missing_cycles = set(expected_cycles) - set(cycles)
    extra_cycles = set(cycles) - set(expected_cycles)
    cycle_check_msg = None
    if missing_cycles or extra_cycles or (len(set(cycles)) != line_count):
        cycle_check_msg = (
            f"Cycle sequence check failed. Expected 0 to {line_count - 1} unique cycles. "
            f"Found {len(set(cycles))} unique cycles. "
            f"Missing: {sorted(list(missing_cycles))[:5]}... " if missing_cycles else ""
            f"Extra/Duplicates: {sorted(list(extra_cycles))[:5]}... " if extra_cycles else ""
        )
    elif min_cycle != 0 or max_cycle != (line_count - 1):
         cycle_check_msg = f"Cycle range check failed. Expected 0 to {line_count - 1}, but found {min_cycle} to {max_cycle}."

    # Mean abstention
    mean_abstention = sum(1 for a in abstentions if a) / line_count if line_count > 0 else 0.0

    summary = {
        "file_path": str(file_path),
        "line_count": line_count,
        "cycles_range": f"{min_cycle} to {max_cycle}",
        "cycle_sequence_valid": cycle_check_msg is None,
        "cycle_check_message": cycle_check_msg,
        "modes_found": list(modes),
        "mean_abstention": mean_abstention,
        "has_roots_ht": has_roots_ht,
        "has_roots_rt": has_roots_rt,
        "has_roots_ut": has_roots_ut,
        "all_roots_present": has_roots_ht and has_roots_rt and has_roots_ut,
    }
    return None, summary

if __name__ == "__main__":
    baseline_file = Path("results/fo_baseline.jsonl")
    rfl_file = Path("results/fo_rfl.jsonl")

    baseline_error, baseline_summary = analyze_jsonl_file(baseline_file)
    rfl_error, rfl_summary = analyze_jsonl_file(rfl_file)

    summary_md = "### FO_RUN_SUMMARY.md\n\n"
    summary_md += "Summary of First Organism (FO) Runner experiments:\n\n"
    summary_md += "| Run                   | Cycles | Mean Abstention | Mode Check | Cycle Range | Roots Present |\n"
    summary_md += "| :-------------------- | :----- | :-------------- | :--------- | :---------- | :------------ |\n"

    # Baseline Summary
    if baseline_error:
        summary_md += f"| {baseline_file.name} | N/A    | N/A             | N/A        | N/A         | N/A           |\n"
        summary_md += f"**Error processing {baseline_file.name}:** {baseline_error}\n\n"
    elif baseline_summary:
        mode_check = "Pass" if len(baseline_summary["modes_found"]) == 1 and "baseline" in baseline_summary["modes_found"] else f"Fail: {baseline_summary['modes_found']}"
        cycle_check = "Pass" if baseline_summary["cycle_sequence_valid"] else "Fail"
        roots_check = "Pass" if baseline_summary["all_roots_present"] else "Fail"
        summary_md += (
            f"| {baseline_summary['file_path'].split('/')[-1]} "
            f"| {baseline_summary['line_count']} "
            f"| {baseline_summary['mean_abstention']:.3f} "
            f"| {mode_check} "
            f"| {baseline_summary['cycles_range']} ({cycle_check}) "
            f"| {roots_check} |\n"
        )
        if not baseline_summary["cycle_sequence_valid"]:
            summary_md += f"**Cycle Sequence Details for {baseline_file.name}:** {baseline_summary['cycle_check_message']}\n\n"
        
    # RFL Summary
    if rfl_error:
        summary_md += f"| {rfl_file.name} | N/A    | N/A             | N/A        | N/A         | N/A           |\n"
        summary_md += f"**Error processing {rfl_file.name}:** {rfl_error}\n\n"
    elif rfl_summary:
        mode_check = "Pass" if len(rfl_summary["modes_found"]) == 1 and "rfl" in rfl_summary["modes_found"] else f"Fail: {rfl_summary['modes_found']}"
        cycle_check = "Pass" if rfl_summary["cycle_sequence_valid"] else "Fail"
        roots_check = "Pass" if rfl_summary["all_roots_present"] else "Fail"
        summary_md += (
            f"| {rfl_summary['file_path'].split('/')[-1]} "
            f"| {rfl_summary['line_count']} "
            f"| {rfl_summary['mean_abstention']:.3f} "
            f"| {mode_check} "
            f"| {rfl_summary['cycles_range']} ({cycle_check}) "
            f"| {roots_check} |\n"
        )
        if not rfl_summary["cycle_sequence_valid"]:
            summary_md += f"**Cycle Sequence Details for {rfl_file.name}:** {rfl_summary['cycle_check_message']}\n\n"
    
    print(summary_md)

    if baseline_summary and baseline_summary["line_count"] != 1000:
        sys.stderr.write(f"Warning: {baseline_file.name} did not contain 1000 cycles.\n")
    if rfl_summary and rfl_summary["line_count"] != 1000:
        sys.stderr.write(f"Warning: {rfl_file.name} did not contain 1000 cycles.\n")
    if rfl_error and "empty" in rfl_error:
        sys.stderr.write(f"Critical: {rfl_file.name} was empty.\n")
