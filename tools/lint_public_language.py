import json
import argparse
import re

# Based on the SHADOW MODE doctrine and regulatory neutrality ruleset.
BANNED_WORDS = [
    # Verbs implying control or prevention
    "prevent", "prevents", "preventing",
    "guarantee", "guarantees", "guaranteeing",
    "ensure", "ensures", "ensuring",
    "secure", "secures", "securing",
    "protect", "protects", "protecting",
    "enforce", "enforces", "enforcing",
    "mitigate", "mitigates", "mitigating",
    "stop", "stops", "stopping",
    "block", "blocks", "blocking",
    "govern", "governs", "governing",
    "police", "polices", "policing",

    # Verbs implying finality or a "solution"
    "solve", "solves", "solving",
    "fix", "fixes", "fixing",
    "eliminate", "eliminates", "eliminating",
    "resolve", "resolves", "resolving",
    "eradicate", "eradicates", "eradicating",

    # Verds implying subjective understanding
    "understand", "understands", "understanding",
    "think", "thinks", "thinking",
    "know", "knows", "knowing",
    "believe", "believes", "believing",
    "interpret", "interprets", "interpreting",
    "contextualize", "contextualizes", "contextualizing",

    # Words implying absolute guarantees
    "proof",  # Noun, but used in a guarantee context
    "always",
    "never",
    "infallible",
    "perfect",
]

def scan_file(filepath):
    """Scans a single file for banned words."""
    violations = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                for word in BANNED_WORDS:
                    # Use regex to find whole words only, case-insensitive
                    if re.search(r'\b' + re.escape(word) + r'\b', line, re.IGNORECASE):
                        violations.append({
                            "line": i + 1,
                            "word": word,
                            "context": line.strip()
                        })
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    return violations

def main():
    """Main function to parse arguments and run the scan."""
    parser = argparse.ArgumentParser(description="Scan public-facing documents for non-neutral language.")
    parser.add_argument("files", nargs='+', help="List of file paths to scan.")
    parser.add_argument("--json-report", default="lint_report.json", help="Output file for the JSON report.")
    parser.add_argument("--md-summary", default="lint_summary.md", help="Output file for the Markdown summary.")
    args = parser.parse_args()

    report = {}
    total_violations = 0

    for f in args.files:
        violations = scan_file(f)
        if violations is not None:
            report[f] = violations
            total_violations += len(violations)

    # Write JSON report
    with open(args.json_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    # Write Markdown summary
    with open(args.md_summary, 'w', encoding='utf-8') as f:
        f.write("# Public Language Linting Report\n\n")
        f.write(f"**Total Violations Found:** {total_violations}\n\n")
        if total_violations == 0:
            f.write("✅ No non-neutral language detected in the scanned files.\n")
        else:
            for filepath, violations in report.items():
                if violations:
                    f.write(f"### 📄 `{filepath}`\n")
                    f.write("| Line | Banned Word | Context |\n")
                    f.write("|------|-------------|---------|\n")
                    for v in violations:
                        f.write(f"| {v['line']} | `{v['word']}` | `{v['context']}` |\n")
                    f.write("\n")

    print(f"Scan complete. Found {total_violations} violations.")
    print(f"JSON report saved to: {args.json_report}")
    print(f"Markdown summary saved to: {args.md_summary}")

if __name__ == "__main__":
    main()
