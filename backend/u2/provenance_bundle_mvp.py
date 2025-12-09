"""
Provenance Bundle Engine MVP

This module provides an MVP implementation of the provenance bundle engine.
Creates reproducibility bundles without cryptographic signing (MVP scope).

Author: Manus-F
Date: 2025-12-06
Status: Phase V MVP Implementation
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# MANIFEST BUILDER
# ============================================================================

@dataclass
class FileEntry:
    """Entry in the bundle manifest."""
    path: str  # Relative path in bundle
    sha256: str  # SHA-256 hash of file content
    size_bytes: int  # File size


class ManifestBuilder:
    """
    Builds a manifest of all files in a bundle.
    """
    
    def __init__(self):
        self.entries: List[FileEntry] = []
    
    def add_file(self, path: Path, relative_to: Path):
        """
        Add a file to the manifest.
        
        Args:
            path: Absolute path to file
            relative_to: Base directory for relative path
        """
        # Compute relative path
        rel_path = str(path.relative_to(relative_to))
        
        # Compute SHA-256 hash
        sha256 = self._compute_file_hash(path)
        
        # Get file size
        size_bytes = path.stat().st_size
        
        # Add entry
        entry = FileEntry(
            path=rel_path,
            sha256=sha256,
            size_bytes=size_bytes,
        )
        self.entries.append(entry)
    
    def add_directory(self, dir_path: Path, relative_to: Path):
        """
        Add all files in a directory to the manifest.
        
        Args:
            dir_path: Directory to add
            relative_to: Base directory for relative paths
        """
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                self.add_file(file_path, relative_to)
    
    def build(self) -> Dict[str, Any]:
        """
        Build manifest dictionary.
        
        Returns:
            Manifest dictionary
        """
        return {
            "files": [
                {
                    "path": entry.path,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                }
                for entry in sorted(self.entries, key=lambda e: e.path)
            ],
            "total_files": len(self.entries),
            "total_bytes": sum(e.size_bytes for e in self.entries),
        }
    
    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()


# ============================================================================
# TRACE MERGER
# ============================================================================

class TraceMerger:
    """
    Merges multiple trace files into a single canonical trace.
    """
    
    def merge(
        self,
        trace_paths: List[Path],
        output_path: Path,
    ):
        """
        Merge traces into single file with canonical ordering.
        
        Args:
            trace_paths: List of trace JSONL files
            output_path: Output path for merged trace
        """
        # Read all events
        all_events = []
        for trace_path in trace_paths:
            with open(trace_path, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    all_events.append(event)
        
        # Sort canonically by (cycle, worker_id, timestamp_ms)
        all_events.sort(key=lambda e: (
            e.get("cycle", 0),
            e.get("worker_id", 0),
            e.get("timestamp_ms", 0),
        ))
        
        # Write merged trace
        with open(output_path, 'w') as f:
            for event in all_events:
                f.write(json.dumps(event, sort_keys=True) + '\n')
    
    def compute_per_cycle_hashes(
        self,
        trace_path: Path,
    ) -> Dict[int, str]:
        """
        Compute per-cycle trace hashes.
        
        Args:
            trace_path: Path to trace JSONL file
            
        Returns:
            Dictionary mapping cycle → hash
        """
        # Group events by cycle
        by_cycle = {}
        with open(trace_path, 'r') as f:
            for line in f:
                event = json.loads(line)
                cycle = event.get("cycle", 0)
                if cycle not in by_cycle:
                    by_cycle[cycle] = []
                by_cycle[cycle].append(event)
        
        # Compute hash for each cycle
        cycle_hashes = {}
        for cycle, events in sorted(by_cycle.items()):
            cycle_hashes[cycle] = self._compute_events_hash(events)
        
        return cycle_hashes
    
    def _compute_events_hash(self, events: List[Dict[str, Any]]) -> str:
        """Compute SHA-256 hash of events."""
        hasher = hashlib.sha256()
        
        for event in sorted(events, key=lambda e: json.dumps(e, sort_keys=True)):
            # Remove non-deterministic fields
            event_copy = event.copy()
            if "timestamp_ms" in event_copy:
                del event_copy["timestamp_ms"]
            if "data" in event_copy and "result" in event_copy["data"]:
                if "timestamp_ms" in event_copy["data"]["result"]:
                    del event_copy["data"]["result"]["timestamp_ms"]
            
            # Canonical serialization
            canonical = json.dumps(event_copy, sort_keys=True, separators=(",", ":"))
            hasher.update(canonical.encode())
        
        return hasher.hexdigest()


# ============================================================================
# MERKLE TREE BUILDER
# ============================================================================

class MerkleTreeBuilder:
    """
    Builds a Merkle tree from file hashes.
    """
    
    def build_root(self, file_hashes: List[str]) -> str:
        """
        Build Merkle root from file hashes.
        
        Args:
            file_hashes: List of SHA-256 hashes (hex strings)
            
        Returns:
            Merkle root hash
        """
        if not file_hashes:
            return hashlib.sha256(b"").hexdigest()
        
        # Sort hashes for determinism
        hashes = sorted(file_hashes)
        
        # Build tree bottom-up
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    # Pair exists
                    combined = hashes[i] + hashes[i + 1]
                else:
                    # Odd node, duplicate
                    combined = hashes[i] + hashes[i]
                
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)
            
            hashes = next_level
        
        return hashes[0]


# ============================================================================
# PROVENANCE BUNDLE
# ============================================================================

@dataclass
class ProvenanceBundle:
    """
    Complete provenance bundle for an experiment.
    """
    experiment_id: str
    slice_name: str
    total_cycles: int
    master_seed: str
    
    # Manifest
    manifest: Dict[str, Any]
    
    # Trace hashes
    trace_hash: str
    per_cycle_hashes: Dict[int, str]
    
    # Merkle root
    merkle_root: str
    
    # Metadata
    substrate_version: str
    bundle_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "total_cycles": self.total_cycles,
            "master_seed": self.master_seed,
            "manifest": self.manifest,
            "trace_hash": self.trace_hash,
            "per_cycle_hashes": {str(k): v for k, v in self.per_cycle_hashes.items()},
            "merkle_root": self.merkle_root,
            "substrate_version": self.substrate_version,
            "bundle_version": self.bundle_version,
        }
    
    def save(self, output_path: Path):
        """Save bundle to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


# ============================================================================
# BUNDLE GENERATOR
# ============================================================================

class BundleGenerator:
    """
    Generates provenance bundles from experiment artifacts.
    """
    
    def __init__(self):
        self.manifest_builder = ManifestBuilder()
        self.trace_merger = TraceMerger()
        self.merkle_builder = MerkleTreeBuilder()
    
    def generate(
        self,
        experiment_id: str,
        slice_name: str,
        total_cycles: int,
        master_seed: str,
        artifacts_dir: Path,
        output_path: Path,
        substrate_version: str = "1.0.0-mvp",
    ) -> ProvenanceBundle:
        """
        Generate provenance bundle from experiment artifacts.
        
        Args:
            experiment_id: Experiment ID
            slice_name: Slice name
            total_cycles: Total cycles executed
            master_seed: Master seed used
            artifacts_dir: Directory containing experiment artifacts
            output_path: Path to save bundle
            substrate_version: Substrate version string
            
        Returns:
            ProvenanceBundle object
        """
        # Build manifest
        self.manifest_builder.add_directory(artifacts_dir, artifacts_dir)
        manifest = self.manifest_builder.build()
        
        # Merge traces (if multiple)
        trace_files = list(artifacts_dir.glob("**/trace*.jsonl"))
        if len(trace_files) > 1:
            merged_trace_path = artifacts_dir / "merged_trace.jsonl"
            self.trace_merger.merge(trace_files, merged_trace_path)
        elif len(trace_files) == 1:
            merged_trace_path = trace_files[0]
        else:
            raise ValueError("No trace files found in artifacts directory")
        
        # Compute trace hash
        trace_hash = self._compute_file_hash(merged_trace_path)
        
        # Compute per-cycle hashes
        per_cycle_hashes = self.trace_merger.compute_per_cycle_hashes(
            merged_trace_path
        )
        
        # Build Merkle root
        file_hashes = [entry["sha256"] for entry in manifest["files"]]
        merkle_root = self.merkle_builder.build_root(file_hashes)
        
        # Create bundle
        bundle = ProvenanceBundle(
            experiment_id=experiment_id,
            slice_name=slice_name,
            total_cycles=total_cycles,
            master_seed=master_seed,
            manifest=manifest,
            trace_hash=trace_hash,
            per_cycle_hashes=per_cycle_hashes,
            merkle_root=merkle_root,
            substrate_version=substrate_version,
            bundle_version="1.0.0-mvp",
        )
        
        # Save bundle
        bundle.save(output_path)
        
        return bundle
    
    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def generate_bundle_from_artifacts(
    artifacts_dir: Path,
    output_path: Path,
):
    """
    Generate provenance bundle from experiment artifacts.
    
    Args:
        artifacts_dir: Directory containing experiment artifacts
        output_path: Path to save bundle
    """
    generator = BundleGenerator()
    
    bundle = generator.generate(
        experiment_id="test_experiment",
        slice_name="test_slice",
        total_cycles=5,
        master_seed="0xmaster",
        artifacts_dir=artifacts_dir,
        output_path=output_path,
    )
    
    print(f"✓ Bundle generated: {output_path}")
    print(f"  Merkle root: {bundle.merkle_root}")
    print(f"  Trace hash: {bundle.trace_hash}")
    print(f"  Total files: {bundle.manifest['total_files']}")
    print(f"  Total bytes: {bundle.manifest['total_bytes']}")
    print(f"  Per-cycle hashes: {len(bundle.per_cycle_hashes)}")


if __name__ == "__main__":
    # Example: generate bundle from /tmp artifacts
    artifacts_dir = Path("/tmp/u2_artifacts")
    output_path = Path("/tmp/provenance_bundle.json")
    
    if artifacts_dir.exists():
        generate_bundle_from_artifacts(artifacts_dir, output_path)
    else:
        print(f"Artifacts directory not found: {artifacts_dir}")
        print("Run distributed_frontier_mvp.py first to generate artifacts.")
