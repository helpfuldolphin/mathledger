import json
from pathlib import Path
from collections import defaultdict
import datetime

def analyze_census():
    manifest_path = Path("ops/spanning_set_manifest.json")
    output_path = Path("docs/SPANNING_SET_CENSUS_V1.md")
    
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    entries = data["entries"]
    
    stats = defaultdict(lambda: {"files": 0, "loc": 0, "size": 0})
    details = defaultdict(list)
    
    for path, info in entries.items():
        if info.get("type") == "directory":
            continue
            
        cls = info["classification"]
        stats[cls]["files"] += 1
        stats[cls]["loc"] += info.get("lines", 0)
        stats[cls]["size"] += info["size"]
        details[cls].append((path, info["lines"], info["justification"]))

    # Sort details by path for cleaner output
    for cls in details:
        details[cls].sort(key=lambda x: x[0])

    # Generate Markdown
    lines = []
    lines.append(f"# Spanning Set Census V1")
    lines.append(f"**Generated:** {datetime.datetime.now().isoformat()}")
    lines.append("")
    lines.append("## 1. Topline Statistics")
    lines.append("| Classification | Files | LOC | Size (Bytes) |")
    lines.append("|---|---|---|---|")
    
    total_files = 0
    total_loc = 0
    
    for cls in ["core", "supporting", "experimental", "archive-candidate"]:
        s = stats[cls]
        lines.append(f"| **{cls.upper()}** | {s['files']} | {s['loc']:,} | {s['size']:,} |")
        total_files += s['files']
        total_loc += s['loc']
        
    lines.append(f"| **TOTAL** | {total_files} | {total_loc:,} | - |")
    lines.append("")
    
    lines.append("## 2. Classification Map")
    
    lines.append("### Core (Minimal Basis Candidate)")
    lines.append("> **Definition:** Must survive into `basis/` (e.g., canonical hashing, dual_root, RFL core).")
    lines.append("")
    lines.append("| Module/Path | LOC | Justification |")
    lines.append("|---|---|---|")
    # Aggregate by top-level directory for brevity, but list individual files if root
    seen_dirs = set()
    for path, loc, just in details["core"]:
        parts = Path(path).parts
        if len(parts) > 1:
            top_dir = parts[0]
            if top_dir not in seen_dirs:
                 # Calculate total LOC for this dir
                 dir_loc = sum(l for p, l, j in details["core"] if p.startswith(top_dir))
                 lines.append(f"| `{top_dir}/` | {dir_loc:,} | {just} |")
                 seen_dirs.add(top_dir)
        else:
            lines.append(f"| `{path}` | {loc} | {just} |")
            
    lines.append("")

    lines.append("### Supporting (Infrastructure)")
    lines.append("> **Definition:** Ops, UI, tests, and config that support the organism but are not the organism itself.")
    lines.append("")
    # Group supporting
    seen_dirs_supp = set()
    lines.append("| Area | Est. LOC | Notes |")
    lines.append("|---|---|---|")
    for path, loc, just in details["supporting"]:
        parts = Path(path).parts
        top_level = parts[0]
        if top_level not in seen_dirs_supp:
             dir_loc = sum(l for p, l, j in details["supporting"] if p.startswith(top_level))
             lines.append(f"| `{top_level}` | {dir_loc:,} | {just} |")
             seen_dirs_supp.add(top_level)
    lines.append("")

    lines.append("### Experimental (Quarantine Zone)")
    lines.append("> **Definition:** Proto-scripts, scratchpads, and patch files. High risk of entropy.")
    lines.append("")
    for path, loc, just in details["experimental"]:
        lines.append(f"- `{path}` ({loc} lines): {just}")
    lines.append("")
    
    lines.append("### Archive Candidates")
    lines.append("> **Definition:** Obsolete sludge to be moved to `archive/` or deleted.")
    lines.append("")
    for path, loc, just in details["archive-candidate"]:
        lines.append(f"- `{path}`: {just}")
    lines.append("")

    lines.append("## 3. Basis Nucleus Proposal")
    lines.append("The following modules constitute the minimal viable organism:")
    lines.append("1. **`basis/`**: The existing formalized core.")
    lines.append("2. **`backend/crypto`**: Canonical hashing and signatures.")
    lines.append("3. **`backend/ledger`**: Immutable ledger structures.")
    lines.append("4. **`backend/logic`**: Normalization and canonicalization.")
    lines.append("5. **`backend/rfl`**: The Reinforced Feedback Loop logic.")
    lines.append("6. **`attestation/`**: Dual root attestation (Law).")
    lines.append("7. **`substrate/`**: The Lean/formal verification substrate.")
    lines.append("8. **`curriculum/`**: The learning schedule.")
    lines.append("")
    lines.append("**Action:** These should be consolidated into the `basis/` namespace to enforce the separation between 'The Organism' and 'The Lab'.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    analyze_census()
    print("Analysis complete.")

