import re
import json
import os
import sys
import argparse
from datetime import datetime

# --- CONFIGURATION ---
SCHEMA_VERSION = "1.0.0"
CHECKLIST_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'system_law', 'Phase_X', 'Phase_X_P3_P4_Evidence_Checklist.md')

# --- PARSING LOGIC ---
def parse_checklist(markdown_content):
    """Parses the checklist to extract artifact status, gate, and dependencies."""
    lines = markdown_content.splitlines()
    table_lines = [line for line in lines if line.strip().startswith('| **P')]
    
    artifacts = []
    backtick_re = re.compile(r"`(.+?)`")

    for line in table_lines:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) > 5:
            gate = parts[1].replace('*', '')
            status = (backtick_re.search(parts[2]).group(1) if backtick_re.search(parts[2]) else "UNKNOWN").strip()
            artifact = (backtick_re.search(parts[4]).group(1) if backtick_re.search(parts[4]) else "UNKNOWN").strip()
            artifacts.append({"gate": gate, "status": status, "artifact": artifact})

    in_dag_section = False
    dag_lines = []
    for line in lines:
        if "Dependency DAG" in line: in_dag_section = True
        elif "Sprint Schedule" in line: in_dag_section = False
        elif in_dag_section and "->" in line: dag_lines.append(line.strip())

    return artifacts, dag_lines

# --- ANALYSIS LOGIC ---
def analyze_progress(artifacts, dag_lines):
    """Analyzes artifacts to calculate readiness and identify blocked items."""
    p3_artifacts = [a for a in artifacts if a['gate'] == 'P3']
    p4_artifacts = [a for a in artifacts if a['gate'] == 'P4']

    status_counts = {"READY": 0, "IN PROGRESS": 0, "BLOCKED": 0, "TODO": 0}
    for item in artifacts:
        if item['status'] in status_counts:
            status_counts[item['status']] += 1

    def calculate_readiness(artifact_list):
        if not artifact_list: return 0.0
        ready_count = sum(1 for a in artifact_list if a['status'] == 'READY')
        return round((ready_count / len(artifact_list)) * 100, 2)

    summary = {
        "readiness_percentage": {
            "overall": calculate_readiness(artifacts),
            "p3": calculate_readiness(p3_artifacts),
            "p4": calculate_readiness(p4_artifacts),
        },
        "status_counts": status_counts,
        "blocked_items": []
    }

    blocked_artifacts = [a for a in artifacts if a['status'] == 'BLOCKED']
    for blocked in blocked_artifacts:
        dependencies = []
        for dag_line in dag_lines:
            dependency, dependent = [part.strip() for part in dag_line.split('->', 1)]
            if f"`{blocked['artifact']}`" in dependent:
                dependencies.extend([f"`{d.strip()}`" for d in re.findall(r'`([^`]+)`', dependency)])
        summary["blocked_items"].append({"artifact": blocked['artifact'], "dependencies": dependencies})
        
    return summary

def calculate_trend(current_summary, previous_summary):
    """Calculates the diff between the current and previous summary."""
    if not previous_summary:
        return {status: {"change": 0, "direction": "none"} for status in current_summary["status_counts"]}

    trend = {}
    for status, current_count in current_summary["status_counts"].items():
        previous_count = previous_summary.get("status_counts", {}).get(status, 0)
        change = current_count - previous_count
        direction = "increase" if change > 0 else "decrease" if change < 0 else "none"
        trend[status] = {"change": change, "direction": direction}
    return trend

# --- OUTPUT GENERATION ---
def generate_markdown_summary(summary, trend, ascii_only=False):
    """Generates a human-readable markdown summary."""
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    symbols = {
        "READY": "[OK]", "IN PROGRESS": "[IN PROGRESS]", "BLOCKED": "[BLOCKED]", "TODO": "[TODO]"
    } if ascii_only else {
        "READY": "✅", "IN PROGRESS": "🚧", "BLOCKED": "🛑", "TODO": "📋"
    }

    def get_trend_symbol(status):
        t = trend.get(status, {"change": 0, "direction": "none"})
        change = t['change']
        if change == 0: return "" if ascii_only else "&nbsp;"
        
        if ascii_only:
            return f"(+{change})" if change > 0 else f"({change})"
        else:
            return f"▲ (+{change})" if change > 0 else f"▼ ({change})"

    markdown = f"""
# Evidence Checklist Readiness Report
*Last generated: {now}*

## Readiness Dashboard
- **Overall Readiness:** `{summary['readiness_percentage']['overall']}%`
- **P3 Gate Readiness:** `{summary['readiness_percentage']['p3']}%`
- **P4 Gate Readiness:** `{summary['readiness_percentage']['p4']}%`

## Sprint Status
| Status | Count | Trend |
|---|---|---|
| {symbols['READY']} READY | `{summary['status_counts']['READY']}` | {get_trend_symbol('READY')} |
| {symbols['IN PROGRESS']} IN PROGRESS | `{summary['status_counts']['IN PROGRESS']}` | {get_trend_symbol('IN PROGRESS')} |
| {symbols['BLOCKED']} BLOCKED | `{summary['status_counts']['BLOCKED']}` | {get_trend_symbol('BLOCKED')} |
| {symbols['TODO']} TODO | `{summary['status_counts']['TODO']}` | {get_trend_symbol('TODO')} |

"""
    if summary["blocked_items"]:
        block_symbol = symbols['BLOCKED']
        markdown += f"\n## {block_symbol} Blocked Items\n"
        for item in summary["blocked_items"]:
            deps = ', '.join(item['dependencies']) if item['dependencies'] else 'None identified'
            markdown += f"- **{item['artifact']}**: Blocked by {deps}\n"
            
    attestation = """

---

## Attestation

This report provides an automated summary of the evidence checklist status.

-   **Verified**: The *status* (`READY`, `BLOCKED`, etc.) and *count* of artifacts as parsed from the checklist markdown file.
-   **Not Verified**: The *correctness* or *validity* of the artifacts themselves. A `READY` status indicates the item is marked complete, not that its content has been validated by this script.
-   **Intended Consumer**: Project leadership and team leads for a high-level overview of sprint progress and blockers.
"""
    markdown += attestation
            
    return markdown.strip()

# --- MAIN EXECUTION ---
def main():
    """Main function to orchestrate the summary generation."""
    parser = argparse.ArgumentParser(description="Summarizes the Evidence Checklist.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.dirname(__file__),
        help="Directory to write the output files. Defaults to the script's directory."
    )
    parser.add_argument(
        '--ascii-only', 
        action=argparse.BooleanOptionalAction,
        default=sys.platform == "win32",
        help="Use ASCII symbols instead of emojis. Defaults to True on Windows."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    json_output_path = os.path.join(args.output_dir, 'readiness.json')
    markdown_output_path = os.path.join(args.output_dir, 'readiness_summary.md')

    previous_summary = None
    if os.path.exists(json_output_path):
        with open(json_output_path, 'r', encoding='utf-8') as f:
            try: previous_summary = json.load(f)
            except json.JSONDecodeError: pass 

    with open(CHECKLIST_PATH_DEFAULT, 'r', encoding='utf-8') as f:
        content = f.read()
    artifacts, dag_lines = parse_checklist(content)

    current_summary = analyze_progress(artifacts, dag_lines)
    trend = calculate_trend(current_summary, previous_summary)
    
    # Finalize JSON payload
    current_summary["schema_version"] = SCHEMA_VERSION
    current_summary["trend_since_last_run"] = trend
    current_summary["last_run_timestamp"] = previous_summary.get("current_run_timestamp") if previous_summary else None
    current_summary["current_run_timestamp"] = datetime.utcnow().isoformat()

    markdown_output = generate_markdown_summary(current_summary, trend, ascii_only=args.ascii_only)

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(current_summary, f, indent=4, sort_keys=True)
        
    with open(markdown_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_output)

    print(f"Successfully generated readiness summary in '{args.output_dir}'")

if __name__ == "__main__":
    main()