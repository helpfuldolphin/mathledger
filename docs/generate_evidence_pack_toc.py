#!/usr/bin/env python3
"""
Evidence Pack v1 Table of Contents Generator

Generates machine- and human-readable TOC describing all documents
that constitute Evidence Pack v1.

Usage:
    python docs/generate_evidence_pack_toc.py [--config CONFIG_FILE] [--output-dir DIR]

Examples:
    python docs/generate_evidence_pack_toc.py
    python docs/generate_evidence_pack_toc.py --config docs/evidence_pack_config.yaml
    python docs/generate_evidence_pack_toc.py --output-dir artifacts/evidence
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import yaml


class EvidencePackTOCGenerator:
    """Generates TOC for Evidence Pack v1."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.entries: List[Dict[str, Any]] = []
        
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        content = config_path.read_text(encoding='utf-8')
        
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(content)
        elif config_path.suffix == '.json':
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate a single entry in the config."""
        required_fields = ['path', 'description']
        for field in required_fields:
            if field not in entry:
                print(f"Warning: Entry missing required field '{field}': {entry}", file=sys.stderr)
                return False
        
        # Check if file exists
        file_path = self.root_dir / entry['path']
        if not file_path.exists():
            print(f"Warning: File does not exist: {entry['path']}", file=sys.stderr)
            return False
        
        return True
    
    def add_metadata(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to entry (file size, modification time, etc.)."""
        file_path = self.root_dir / entry['path']
        
        enriched = entry.copy()
        
        if file_path.exists():
            stat = file_path.stat()
            enriched['file_size'] = stat.st_size
            enriched['modified_time'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        return enriched
    
    def process_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process configuration and build entries list."""
        entries = []
        
        # Get documents list from config
        documents = config.get('documents', [])
        
        if not documents:
            print("Warning: No documents found in config", file=sys.stderr)
            return entries
        
        for doc in documents:
            if self.validate_entry(doc):
                enriched = self.add_metadata(doc)
                entries.append(enriched)
        
        return entries
    
    def generate_json(self, entries: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON TOC."""
        return {
            'format_version': metadata.get('format_version', '1.0'),
            'generated_at': datetime.now().astimezone().replace(microsecond=0).isoformat(),
            'metadata': metadata,
            'documents': entries
        }
    
    def generate_markdown(self, entries: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Generate Markdown TOC."""
        lines = []
        
        # Header
        lines.append("# Evidence Pack v1 - Table of Contents")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        lines.append(f"**Version:** {metadata.get('version', '1.0')}")
        lines.append(f"**Description:** {metadata.get('description', 'Evidence Pack v1 Documentation Index')}")
        lines.append("")
        
        # Group by phase if available
        by_phase: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            phase = entry.get('phase', 'General')
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(entry)
        
        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for phase in sorted(by_phase.keys()):
            lines.append(f"- [{phase}](#{phase.lower().replace(' ', '-')})")
        lines.append("")
        
        # Documents by phase
        for phase in sorted(by_phase.keys()):
            lines.append(f"## {phase}")
            lines.append("")
            
            for entry in sorted(by_phase[phase], key=lambda e: e['path']):
                path = entry['path']
                desc = entry['description']
                category = entry.get('category', 'documentation')
                
                lines.append(f"### `{path}`")
                lines.append("")
                lines.append(f"**Description:** {desc}")
                lines.append("")
                lines.append(f"**Category:** {category}")
                
                if 'file_size' in entry:
                    size_kb = entry['file_size'] / 1024
                    lines.append(f"**Size:** {size_kb:.1f} KB")
                
                if entry.get('tags'):
                    tags = ', '.join(entry['tags'])
                    lines.append(f"**Tags:** {tags}")
                
                lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*This TOC is automatically generated. Do not edit manually.*")
        lines.append("")
        lines.append(f"*Generated by: `{Path(__file__).name}`*")
        lines.append("")
        
        return '\n'.join(lines)
    
    def generate(self, config_path: Path, output_dir: Path) -> bool:
        """
        Generate TOC files.
        Returns True if successful.
        """
        # Load config
        try:
            config = self.load_config(config_path)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return False
        
        # Process entries
        entries = self.process_config(config)
        
        if not entries:
            print("Error: No valid entries to generate TOC", file=sys.stderr)
            return False
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get metadata from config
        metadata = config.get('metadata', {})
        
        # Generate JSON
        json_output = output_dir / 'evidence_pack_v1_toc.json'
        json_data = self.generate_json(entries, metadata)
        json_output.write_text(json.dumps(json_data, indent=2), encoding='utf-8')
        print(f"✓ Generated: {json_output}")
        
        # Generate Markdown
        md_output = output_dir / 'evidence_pack_v1_toc.md'
        md_content = self.generate_markdown(entries, metadata)
        md_output.write_text(md_content, encoding='utf-8')
        print(f"✓ Generated: {md_output}")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Evidence Pack v1 Table of Contents"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('docs/evidence_pack_config.yaml'),
        help='Path to config file (YAML or JSON)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs'),
        help='Output directory for TOC files'
    )
    
    args = parser.parse_args()
    
    # Find repository root
    root_dir = Path(__file__).parent.parent
    
    # Resolve paths
    config_path = root_dir / args.config
    output_dir = root_dir / args.output_dir
    
    generator = EvidencePackTOCGenerator(root_dir)
    
    success = generator.generate(config_path, output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
