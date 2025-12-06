#!/usr/bin/env python3
"""
Docs Regenerator - Auto-generate toolbox docs (ASCII-only)

Scans toolbox commands and regenerates an AUTO_DOCS.md file and updates the
README.md section between markers:

<!-- BEGIN AUTO DOCS -->
... generated content ...
<!-- END AUTO DOCS -->

Usage:
    python docs_regen.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent.parent
README = ROOT / 'README.md'
AUTO = ROOT / 'AUTO_DOCS.md'


def extract_docstring(path: Path) -> str:
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        return ''
    m = re.search(r'^[\s\S]*?\"\"\"([\s\S]*?)\"\"\"', text)
    if m:
        return m.group(1).strip()
    m2 = re.match(r'(?s)^#\!.*?\n(.*)$', text)
    return (m2.group(1).strip() if m2 else '')


def build_auto_docs() -> str:
    lines = []
    lines.append('ASCII Auto-Generated Docs')
    lines.append('==========================')
    lines.append('')

    ml_path = ROOT / 'ml'
    ml_doc = extract_docstring(ml_path)
    if ml_doc:
        lines.append('ml (MathLedger CLI)')
        lines.append('--------------------')
        lines.append(ml_doc)
        lines.append('')

    for tool in ['deterministic_build.py', 'seed_replay.py', 'artifact_verifier.py', 'pr-helper.py', 'workflow-validator.py']:
        p = ROOT / tool
        if p.exists():
            doc = extract_docstring(p)
            title = tool
            lines.append(title)
            lines.append('-' * len(title))
            if doc:
                lines.append(doc)
            else:
                lines.append('(no docstring found)')
            lines.append('')

    return '\n'.join(lines)


def update_readme(auto_text: str) -> None:
    content = README.read_text(encoding='utf-8')
    start = '<!-- BEGIN AUTO DOCS -->'
    end = '<!-- END AUTO DOCS -->'
    if start in content and end in content:
        before = content.split(start)[0].rstrip()
        after = content.split(end)[1].lstrip()
        new = f"{before}\n{start}\n\n{auto_text}\n\n{end}\n{after}"
        README.write_text(new, encoding='utf-8')
    else:
        new = f"{content.rstrip()}\n\n{start}\n\n{auto_text}\n\n{end}\n"
        README.write_text(new, encoding='utf-8')


def main() -> int:
    auto = build_auto_docs()
    AUTO.write_text(auto, encoding='utf-8')
    update_readme(auto)
    print('Auto-docs regenerated:')
    print(f'  - {AUTO}')
    print(f'  - {README} (updated)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
