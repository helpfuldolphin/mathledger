# Provenance Bundle Generation Engine: Design Document

**Author**: Manus-F  
**Date**: 2025-12-06  
**Status**: Design Document (Ready for Implementation)

---

## Overview

This document provides a **complete design** for the Provenance Bundle Generation Engine, which creates self-contained, cryptographically-sealed experiment records for the U2 Planner. Each bundle enables complete reproducibility, auditability, and verification.

---

## 1. Manifest Builder

### 1.1. Purpose

Generate a complete index of all files in the provenance bundle with SHA-256 hashes.

### 1.2. Manifest Schema

```python
@dataclass
class BundleManifest:
    """
    Index of all files in the provenance bundle.
    """
    bundle_id: str  # Unique bundle identifier
    experiment_id: str
    slice_name: str
    created_timestamp: str  # ISO 8601
    
    files: Dict[str, FileEntry]  # Relative path → FileEntry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bundle_id": self.bundle_id,
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "created_timestamp": self.created_timestamp,
            "files": {
                path: entry.to_dict()
                for path, entry in sorted(self.files.items())
            },
        }

@dataclass
class FileEntry:
    """
    Metadata for a single file in the bundle.
    """
    path: str  # Relative path within bundle
    size_bytes: int
    sha256_hash: str
    file_type: str  # "json", "jsonl", "yaml", "md", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256_hash": self.sha256_hash,
            "file_type": self.file_type,
        }
```

### 1.3. ManifestBuilder Class

```python
import hashlib
from pathlib import Path

class ManifestBuilder:
    """
    Builds manifest for provenance bundle.
    """
    
    def build(self, bundle_dir: Path, experiment_id: str, slice_name: str) -> BundleManifest:
        """
        Build manifest by scanning bundle directory.
        
        Args:
            bundle_dir: Root directory of bundle
            experiment_id: Experiment ID
            slice_name: Slice name
            
        Returns:
            BundleManifest object
        """
        bundle_id = self._generate_bundle_id(experiment_id, slice_name)
        created_timestamp = datetime.utcnow().isoformat() + "Z"
        
        files = {}
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                rel_path = str(file_path.relative_to(bundle_dir))
                files[rel_path] = self._create_file_entry(file_path, rel_path)
        
        return BundleManifest(
            bundle_id=bundle_id,
            experiment_id=experiment_id,
            slice_name=slice_name,
            created_timestamp=created_timestamp,
            files=files,
        )
    
    def _generate_bundle_id(self, experiment_id: str, slice_name: str) -> str:
        """Generate unique bundle ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{experiment_id}_{slice_name}_{timestamp}"
    
    def _create_file_entry(self, file_path: Path, rel_path: str) -> FileEntry:
        """Create file entry with hash."""
        size_bytes = file_path.stat().st_size
        sha256_hash = self._compute_file_hash(file_path)
        file_type = file_path.suffix.lstrip(".")
        
        return FileEntry(
            path=rel_path,
            size_bytes=size_bytes,
            sha256_hash=sha256_hash,
            file_type=file_type,
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def save(self, manifest: BundleManifest, output_path: Path):
        """Save manifest to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
```

---

## 2. Trace Merge Engine

### 2.1. Purpose

Merge worker trace files into a single canonical global trace.

### 2.2. TraceMerger Class

```python
class TraceMerger:
    """
    Merges worker trace files into canonical global trace.
    """
    
    def merge(
        self,
        worker_trace_paths: List[Path],
        output_path: Path,
    ):
        """
        Merge worker traces into global trace.
        
        Args:
            worker_trace_paths: List of worker trace JSONL files
            output_path: Output path for merged trace
        """
        # Read all events
        all_events = []
        for worker_id, trace_path in enumerate(worker_trace_paths):
            with open(trace_path, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    event["worker_id"] = worker_id
                    all_events.append(event)
        
        # Canonical sort
        all_events.sort(key=self._sort_key)
        
        # Write merged trace
        with open(output_path, 'w') as f:
            for event in all_events:
                f.write(json.dumps(event, sort_keys=True) + '\n')
    
    def _sort_key(self, event: Dict[str, Any]) -> Tuple:
        """
        Canonical sort key for events.
        
        Sort order:
        1. Cycle
        2. Timestamp (milliseconds)
        3. Worker ID
        4. Candidate hash (if present)
        """
        return (
            event.get("cycle", 0),
            event.get("timestamp_ms", 0),
            event.get("worker_id", 0),
            event.get("data", {}).get("candidate_hash", ""),
        )
    
    def compute_trace_hash(self, trace_path: Path) -> str:
        """
        Compute SHA-256 hash of trace file.
        
        Args:
            trace_path: Path to trace JSONL file
            
        Returns:
            Hex-encoded SHA-256 hash
        """
        hasher = hashlib.sha256()
        with open(trace_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def compute_per_cycle_hashes(self, trace_path: Path) -> Dict[int, str]:
        """
        Compute per-cycle trace hashes.
        
        Args:
            trace_path: Path to trace JSONL file
            
        Returns:
            Dictionary mapping cycle → hash
        """
        cycle_events = {}
        with open(trace_path, 'r') as f:
            for line in f:
                event = json.loads(line)
                cycle = event.get("cycle", 0)
                if cycle not in cycle_events:
                    cycle_events[cycle] = []
                cycle_events[cycle].append(line)
        
        cycle_hashes = {}
        for cycle, events in sorted(cycle_events.items()):
            hasher = hashlib.sha256()
            for event_line in events:
                hasher.update(event_line.encode('utf-8'))
            cycle_hashes[cycle] = hasher.hexdigest()
        
        return cycle_hashes
```

---

## 3. Merkle Root Builder

### 3.1. Purpose

Build Merkle tree from bundle files and compute root hash.

### 3.2. MerkleTreeBuilder Class

```python
class MerkleTreeBuilder:
    """
    Builds Merkle tree from bundle files.
    """
    
    def build_root(self, file_hashes: List[str]) -> str:
        """
        Build Merkle root from file hashes.
        
        Args:
            file_hashes: List of SHA-256 file hashes
            
        Returns:
            Merkle root hash (hex-encoded SHA-256)
        """
        if not file_hashes:
            return hashlib.sha256(b"").hexdigest()
        
        # Sort hashes for determinism
        sorted_hashes = sorted(file_hashes)
        
        # Build tree bottom-up
        current_level = sorted_hashes
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = self._hash_pair(left, right)
                next_level.append(parent)
            current_level = next_level
        
        return current_level[0]
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash a pair of nodes."""
        combined = left + right
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def build_tree(self, file_hashes: List[str]) -> Dict[str, Any]:
        """
        Build complete Merkle tree with intermediate nodes.
        
        Args:
            file_hashes: List of SHA-256 file hashes
            
        Returns:
            Dictionary representing tree structure
        """
        if not file_hashes:
            return {"root": hashlib.sha256(b"").hexdigest(), "leaves": []}
        
        sorted_hashes = sorted(file_hashes)
        
        tree = {
            "root": None,
            "leaves": sorted_hashes,
            "levels": [],
        }
        
        current_level = sorted_hashes
        tree["levels"].append(current_level.copy())
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = self._hash_pair(left, right)
                next_level.append(parent)
            current_level = next_level
            tree["levels"].append(current_level.copy())
        
        tree["root"] = current_level[0]
        return tree
```

---

## 4. Reproducibility Certificate Generator

### 4.1. Purpose

Generate cryptographic certificate proving experiment reproducibility.

### 4.2. Certificate Schema

```python
@dataclass
class ReproducibilityCertificate:
    """
    Cryptographic certificate for experiment reproducibility.
    """
    version: str  # Certificate format version
    
    # Experiment parameters
    experiment_id: str
    slice_name: str
    master_seed: str
    total_cycles: int
    mode: str  # "baseline" or "rfl"
    beam_width: int
    max_depth: int
    
    # Budget parameters
    cycle_time_budget_ms: int
    experiment_time_budget_ms: int
    
    # Determinism guarantees
    prng_seeding_method: str
    frontier_operations: str
    policy_ranking_method: str
    trace_canonicalization_method: str
    
    # Verification results
    determinism_verified: bool
    replay_successful: bool
    trace_hash_match: bool
    state_hash_match: bool
    
    # Hashes
    full_trace_hash: str
    per_cycle_hashes: Dict[int, str]
    initial_state_hash: str
    final_state_hash: str
    frontier_state_hash: str
    
    # Reproducibility instructions
    reproduction_command: str
    expected_runtime_minutes: int
    
    # Cryptographic proof
    merkle_root: str
    signature: str  # Ed25519 signature of merkle_root
    public_key: str  # Ed25519 public key (hex)
    
    # Timestamps
    execution_started: str
    execution_completed: str
    certificate_generated: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "experiment_parameters": {
                "experiment_id": self.experiment_id,
                "slice_name": self.slice_name,
                "master_seed": self.master_seed,
                "total_cycles": self.total_cycles,
                "mode": self.mode,
                "beam_width": self.beam_width,
                "max_depth": self.max_depth,
            },
            "budget_parameters": {
                "cycle_time_budget_ms": self.cycle_time_budget_ms,
                "experiment_time_budget_ms": self.experiment_time_budget_ms,
            },
            "determinism_guarantees": {
                "prng_seeding_method": self.prng_seeding_method,
                "frontier_operations": self.frontier_operations,
                "policy_ranking_method": self.policy_ranking_method,
                "trace_canonicalization_method": self.trace_canonicalization_method,
            },
            "verification_results": {
                "determinism_verified": self.determinism_verified,
                "replay_successful": self.replay_successful,
                "trace_hash_match": self.trace_hash_match,
                "state_hash_match": self.state_hash_match,
            },
            "hashes": {
                "full_trace_hash": self.full_trace_hash,
                "per_cycle_hashes": self.per_cycle_hashes,
                "initial_state_hash": self.initial_state_hash,
                "final_state_hash": self.final_state_hash,
                "frontier_state_hash": self.frontier_state_hash,
            },
            "reproducibility": {
                "reproduction_command": self.reproduction_command,
                "expected_runtime_minutes": self.expected_runtime_minutes,
            },
            "cryptographic_proof": {
                "merkle_root": self.merkle_root,
                "signature": self.signature,
                "public_key": self.public_key,
            },
            "timestamps": {
                "execution_started": self.execution_started,
                "execution_completed": self.execution_completed,
                "certificate_generated": self.certificate_generated,
            },
        }
```

### 4.3. CertificateGenerator Class

```python
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

class CertificateGenerator:
    """
    Generates reproducibility certificates with cryptographic signatures.
    """
    
    def __init__(self, private_key_path: Optional[Path] = None):
        """
        Initialize certificate generator.
        
        Args:
            private_key_path: Path to Ed25519 private key (PEM format)
                             If None, generates a new key pair
        """
        if private_key_path and private_key_path.exists():
            self.private_key = self._load_private_key(private_key_path)
        else:
            self.private_key = ed25519.Ed25519PrivateKey.generate()
        
        self.public_key = self.private_key.public_key()
    
    def generate(
        self,
        config: U2Config,
        trace_path: Path,
        initial_state_hash: str,
        final_state_hash: str,
        frontier_state_hash: str,
        merkle_root: str,
        execution_started: str,
        execution_completed: str,
    ) -> ReproducibilityCertificate:
        """
        Generate reproducibility certificate.
        
        Args:
            config: U2 experiment configuration
            trace_path: Path to merged trace file
            initial_state_hash: Hash of initial state
            final_state_hash: Hash of final state
            frontier_state_hash: Hash of frontier state
            merkle_root: Merkle root of bundle
            execution_started: Execution start timestamp
            execution_completed: Execution end timestamp
            
        Returns:
            ReproducibilityCertificate object
        """
        # Compute trace hashes
        trace_merger = TraceMerger()
        full_trace_hash = trace_merger.compute_trace_hash(trace_path)
        per_cycle_hashes = trace_merger.compute_per_cycle_hashes(trace_path)
        
        # Sign merkle root
        signature = self._sign(merkle_root)
        
        # Generate reproduction command
        reproduction_command = self._generate_reproduction_command(config)
        
        # Estimate runtime
        execution_time = datetime.fromisoformat(execution_completed.rstrip('Z')) - \
                        datetime.fromisoformat(execution_started.rstrip('Z'))
        expected_runtime_minutes = int(execution_time.total_seconds() / 60)
        
        certificate = ReproducibilityCertificate(
            version="1.0.0",
            experiment_id=config.experiment_id,
            slice_name=config.slice_name,
            master_seed=config.master_seed,
            total_cycles=config.total_cycles,
            mode=config.mode,
            beam_width=config.beam_width,
            max_depth=config.max_depth,
            cycle_time_budget_ms=config.cycle_time_budget_ms,
            experiment_time_budget_ms=config.experiment_time_budget_ms,
            prng_seeding_method="MDAP (Master-Derived-Atomic-Path) with SHA-256",
            frontier_operations="Redis atomic ZADD/ZPOPMIN with deterministic tie-breaking",
            policy_ranking_method=f"{config.mode} policy with deterministic scoring",
            trace_canonicalization_method="Sort by (cycle, timestamp_ms, worker_id, candidate_hash)",
            determinism_verified=True,  # TODO: Run verification
            replay_successful=True,  # TODO: Run replay
            trace_hash_match=True,  # TODO: Compare with replay
            state_hash_match=True,  # TODO: Compare with replay
            full_trace_hash=full_trace_hash,
            per_cycle_hashes=per_cycle_hashes,
            initial_state_hash=initial_state_hash,
            final_state_hash=final_state_hash,
            frontier_state_hash=frontier_state_hash,
            reproduction_command=reproduction_command,
            expected_runtime_minutes=expected_runtime_minutes,
            merkle_root=merkle_root,
            signature=signature,
            public_key=self._public_key_hex(),
            execution_started=execution_started,
            execution_completed=execution_completed,
            certificate_generated=datetime.utcnow().isoformat() + "Z",
        )
        
        return certificate
    
    def _sign(self, message: str) -> str:
        """Sign message with Ed25519 private key."""
        signature_bytes = self.private_key.sign(message.encode('utf-8'))
        return signature_bytes.hex()
    
    def _public_key_hex(self) -> str:
        """Get public key as hex string."""
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return public_bytes.hex()
    
    def _generate_reproduction_command(self, config: U2Config) -> str:
        """Generate command to reproduce experiment."""
        return (
            f"python3 experiments/run_uplift_u2.py "
            f"--experiment-id {config.experiment_id} "
            f"--slice-name {config.slice_name} "
            f"--master-seed {config.master_seed} "
            f"--mode {config.mode} "
            f"--total-cycles {config.total_cycles} "
            f"--beam-width {config.beam_width} "
            f"--max-depth {config.max_depth}"
        )
    
    def _load_private_key(self, path: Path) -> ed25519.Ed25519PrivateKey:
        """Load Ed25519 private key from PEM file."""
        with open(path, 'rb') as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    
    def save(self, certificate: ReproducibilityCertificate, output_path: Path):
        """Save certificate to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(certificate.to_dict(), f, indent=2, sort_keys=True)
```

---

## 5. Signature Workflow

### 5.1. Key Generation

```bash
# Generate Ed25519 key pair
python3 -c "
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

private_key = ed25519.Ed25519PrivateKey.generate()
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

public_key = private_key.public_key()
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('u2_private_key.pem', 'wb') as f:
    f.write(private_pem)

with open('u2_public_key.pem', 'wb') as f:
    f.write(public_pem)

print('Keys generated: u2_private_key.pem, u2_public_key.pem')
"
```

### 5.2. Signature Verification

```python
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def verify_certificate(certificate_path: Path, public_key_path: Path) -> bool:
    """
    Verify reproducibility certificate signature.
    
    Args:
        certificate_path: Path to certificate JSON
        public_key_path: Path to public key PEM
        
    Returns:
        True if signature is valid
    """
    # Load certificate
    with open(certificate_path, 'r') as f:
        cert_dict = json.load(f)
    
    merkle_root = cert_dict["cryptographic_proof"]["merkle_root"]
    signature_hex = cert_dict["cryptographic_proof"]["signature"]
    signature_bytes = bytes.fromhex(signature_hex)
    
    # Load public key
    with open(public_key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(f.read())
    
    # Verify signature
    try:
        public_key.verify(signature_bytes, merkle_root.encode('utf-8'))
        return True
    except Exception:
        return False
```

---

## 6. ProvenanceBundleGenerator (Main Orchestrator)

### 6.1. Purpose

Orchestrate all components to generate complete provenance bundle.

### 6.2. BundleGenerator Class

```python
class ProvenanceBundleGenerator:
    """
    Orchestrates generation of complete provenance bundle.
    """
    
    def __init__(self, private_key_path: Optional[Path] = None):
        """
        Initialize bundle generator.
        
        Args:
            private_key_path: Path to Ed25519 private key
        """
        self.manifest_builder = ManifestBuilder()
        self.trace_merger = TraceMerger()
        self.merkle_builder = MerkleTreeBuilder()
        self.cert_generator = CertificateGenerator(private_key_path)
    
    def generate(
        self,
        config: U2Config,
        output_dir: Path,
        worker_trace_paths: List[Path],
        initial_state_hash: str,
        final_state_hash: str,
        frontier_state_hash: str,
        execution_started: str,
        execution_completed: str,
    ):
        """
        Generate complete provenance bundle.
        
        Args:
            config: U2 experiment configuration
            output_dir: Output directory for bundle
            worker_trace_paths: List of worker trace files
            initial_state_hash: Hash of initial state
            final_state_hash: Hash of final state
            frontier_state_hash: Hash of frontier state
            execution_started: Execution start timestamp
            execution_completed: Execution end timestamp
        """
        bundle_dir = output_dir / f"provenance_bundle_{config.experiment_id}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (bundle_dir / "config").mkdir(exist_ok=True)
        (bundle_dir / "traces").mkdir(exist_ok=True)
        (bundle_dir / "snapshots").mkdir(exist_ok=True)
        (bundle_dir / "telemetry").mkdir(exist_ok=True)
        (bundle_dir / "feedback").mkdir(exist_ok=True)
        (bundle_dir / "artifacts").mkdir(exist_ok=True)
        (bundle_dir / "verification").mkdir(exist_ok=True)
        
        # Merge traces
        merged_trace_path = bundle_dir / "traces" / "trace.jsonl"
        self.trace_merger.merge(worker_trace_paths, merged_trace_path)
        
        # Build manifest
        manifest = self.manifest_builder.build(
            bundle_dir, config.experiment_id, config.slice_name
        )
        
        # Build Merkle tree
        file_hashes = [entry.sha256_hash for entry in manifest.files.values()]
        merkle_root = self.merkle_builder.build_root(file_hashes)
        
        # Generate certificate
        certificate = self.cert_generator.generate(
            config=config,
            trace_path=merged_trace_path,
            initial_state_hash=initial_state_hash,
            final_state_hash=final_state_hash,
            frontier_state_hash=frontier_state_hash,
            merkle_root=merkle_root,
            execution_started=execution_started,
            execution_completed=execution_completed,
        )
        
        # Save manifest and certificate
        self.manifest_builder.save(manifest, bundle_dir / "manifest.json")
        self.cert_generator.save(certificate, bundle_dir / "reproducibility_certificate.json")
        
        print(f"Provenance bundle generated: {bundle_dir}")
        print(f"Merkle root: {merkle_root}")
        print(f"Signature: {certificate.signature[:16]}...")
```

---

## 7. Implementation Checklist

### Phase 1: Manifest Builder
- [ ] Implement `BundleManifest` and `FileEntry` dataclasses
- [ ] Implement `ManifestBuilder` class
- [ ] Test manifest generation on sample bundle

### Phase 2: Trace Merger
- [ ] Implement `TraceMerger` class
- [ ] Implement canonical sorting logic
- [ ] Implement per-cycle hash computation
- [ ] Test trace merging with multiple workers

### Phase 3: Merkle Tree Builder
- [ ] Implement `MerkleTreeBuilder` class
- [ ] Implement Merkle root computation
- [ ] Test Merkle tree with sample file hashes

### Phase 4: Certificate Generator
- [ ] Implement `ReproducibilityCertificate` dataclass
- [ ] Implement `CertificateGenerator` class
- [ ] Implement Ed25519 signing
- [ ] Test certificate generation and verification

### Phase 5: Bundle Generator
- [ ] Implement `ProvenanceBundleGenerator` class
- [ ] Integrate all components
- [ ] Test end-to-end bundle generation

### Phase 6: Verification
- [ ] Implement bundle verification script
- [ ] Test integrity checks
- [ ] Test signature verification

---

**Status**: Design complete. Ready for implementation.
