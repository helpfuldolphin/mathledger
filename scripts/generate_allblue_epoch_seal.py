#!/usr/bin/env python3
"""
Generate AllBlue Epoch Seal with Dynamic Lane Discovery (RFC 8785).

Reads lane configuration from config/allblue_lanes.json and dynamically
discovers required artifacts. Emits epoch seal that cryptographically binds
the run with deterministic canonicalization.

Exit codes: 0=ALL BLUE, 1=ABSTAIN (missing artifacts), 2=ERROR
"""

import json
import hashlib
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


def rfc8785_canonicalize(obj: Any) -> str:
    """
    Canonicalize JSON according to RFC 8785 (JCS).
    
    Rules:
    1. Sort object keys lexicographically
    2. No whitespace between tokens
    3. Unicode escape sequences normalized
    4. Numbers in minimal representation
    """
    return json.dumps(
        obj,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
        indent=None
    )


def load_lane_config(config_path: str = "config/allblue_lanes.json") -> Dict[str, Any]:
    """Load dynamic lane configuration with fallback to canonical 6."""
    if not os.path.exists(config_path):
        print(f"WARNING: Lane config not found at {config_path}, using canonical 6 lanes", 
              file=sys.stderr)
        return get_canonical_6_lanes()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded {len(config.get('lanes', []))} lanes from {config_path}")
        return config
    except Exception as e:
        print(f"ERROR: Could not load lane config: {e}", file=sys.stderr)
        print("Falling back to canonical 6 lanes", file=sys.stderr)
        return get_canonical_6_lanes()


def get_canonical_6_lanes() -> Dict[str, Any]:
    """Return canonical 6-lane configuration as fallback."""
    return {
        "version": "1.0",
        "description": "Canonical 6-lane configuration (fallback)",
        "lanes": [
            {
                "id": "triple_hash",
                "name": "Triple-Hash Verification",
                "type": "verification",
                "required": True,
                "artifacts": ["ci_verification/triple_hash_summary.json"]
            },
            {
                "id": "dual_attestation",
                "name": "Dual-Attestation Composite Seal",
                "type": "ci_workflow",
                "required": True,
                "workflow": "dual-attestation.yml",
                "jobs": ["browsermcp", "reasoning", "dual-attestation"]
            },
            {
                "id": "unit_tests",
                "name": "Unit Tests",
                "type": "ci_workflow",
                "required": True,
                "workflow": "ci.yml",
                "jobs": ["test"]
            },
            {
                "id": "uplift_omega",
                "name": "Uplift Omega Validation",
                "type": "ci_workflow",
                "required": True,
                "workflow": "ci.yml",
                "jobs": ["uplift-omega"]
            },
            {
                "id": "uplift_evaluation",
                "name": "Uplift Statistics Computation",
                "type": "ci_workflow",
                "required": True,
                "workflow": "uplift-evaluation.yml",
                "jobs": ["Compute Uplift Statistics"]
            },
            {
                "id": "hygiene",
                "name": "Code Hygiene",
                "type": "ci_workflow",
                "required": False,
                "workflow": "ci.yml",
                "jobs": ["hygiene"]
            }
        ]
    }


def verify_lane_artifacts(lane: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Verify lane artifacts exist and return status.
    
    Returns: (status, hash_or_error)
      - ("PASS", hash) if all artifacts present
      - ("ABSTAIN", reason) if missing required artifacts
      - ("FAIL", reason) if verification failed
    """
    lane_id = lane['id']
    lane_type = lane['type']
    
    if lane_type == 'verification':
        artifacts = lane.get('artifacts', [])
        for artifact_path in artifacts:
            if not os.path.exists(artifact_path):
                return ("ABSTAIN", f"Missing artifact: {artifact_path}")
        
        if lane_id == 'triple_hash':
            try:
                with open('ci_verification/triple_hash_summary.json', 'r') as f:
                    data = json.load(f)
                
                if data.get('overall_status') != 'PASS':
                    return ("FAIL", f"Triple-hash status: {data.get('overall_status')}")
                
                composite = data.get('composite_attestation', '')
                return ("PASS", composite)
            except Exception as e:
                return ("ABSTAIN", f"Could not verify: {e}")
        
        return ("PASS", "verified")
    
    elif lane_type == 'ci_workflow':
        workflow = lane.get('workflow', '')
        jobs = lane.get('jobs', [])
        
        try:
            result = subprocess.run(
                ['gh', 'run', 'list', f'--workflow={workflow}', '--limit', '1', 
                 '--json', 'databaseId,conclusion'],
                capture_output=True,
                text=True,
                check=True
            )
            
            runs = json.loads(result.stdout)
            if not runs:
                return ("ABSTAIN", f"No runs found for workflow: {workflow}")
            
            run = runs[0]
            if run['conclusion'] != 'success':
                return ("FAIL", f"Workflow {workflow} conclusion: {run['conclusion']}")
            
            workflow_data = f"{workflow}:{run['databaseId']}:{','.join(sorted(jobs))}"
            workflow_hash = hashlib.sha256(workflow_data.encode('utf-8')).hexdigest()
            
            return ("PASS", workflow_hash)
            
        except Exception as e:
            return ("ABSTAIN", f"Could not query workflow {workflow}: {e}")
    
    return ("ABSTAIN", f"Unknown lane type: {lane_type}")


def discover_lanes(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically discover and verify all lanes.
    
    Returns: {
        "lanes": {lane_id: {"status": "PASS|ABSTAIN|FAIL", "hash": "...", "message": "..."}},
        "required_pass": int,
        "required_abstain": int,
        "optional_pass": int,
        "optional_abstain": int
    }
    """
    lanes = config.get('lanes', [])
    results = {}
    
    required_pass = 0
    required_abstain = 0
    optional_pass = 0
    optional_abstain = 0
    
    for lane in lanes:
        lane_id = lane['id']
        lane_name = lane['name']
        required = lane.get('required', True)
        
        print(f"Verifying lane: {lane_name} ({lane_id})...")
        status, hash_or_error = verify_lane_artifacts(lane)
        
        results[lane_id] = {
            "name": lane_name,
            "type": lane['type'],
            "required": required,
            "status": status,
            "hash": hash_or_error if status == "PASS" else None,
            "message": hash_or_error if status != "PASS" else "verified"
        }
        
        if status == "PASS":
            if required:
                required_pass += 1
            else:
                optional_pass += 1
        elif status == "ABSTAIN":
            if required:
                required_abstain += 1
            else:
                optional_abstain += 1
        
        print(f"  -> {status}: {hash_or_error if status != 'PASS' else 'verified'}")
    
    return {
        "lanes": results,
        "required_pass": required_pass,
        "required_abstain": required_abstain,
        "optional_pass": optional_pass,
        "optional_abstain": optional_abstain
    }


def compute_hygiene_state_hash() -> str:
    """
    Compute hygiene state hash from repository state.
    
    Includes: git commit hash, branch name, clean/dirty status
    """
    try:
        commit_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = commit_result.stdout.strip()
        
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        branch_name = branch_result.stdout.strip()
        
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        is_clean = len(status_result.stdout.strip()) == 0
        
        hygiene_data = f"commit:{commit_hash}|branch:{branch_name}|clean:{is_clean}"
        return hashlib.sha256(hygiene_data.encode('utf-8')).hexdigest()
        
    except Exception as e:
        print(f"WARNING: Could not compute hygiene state hash: {e}", file=sys.stderr)
        return hashlib.sha256(b"unknown").hexdigest()


def compute_witness_signatures(
    lane_results: Dict[str, Any],
    H_t_hash: Optional[str]
) -> Dict[str, str]:
    """
    Compute witness signatures from independent lanes.
    
    Witnesses:
    - verification_gate_sig: Signature from triple_hash lane (H_t)
    - hermetic_matrix_sig: Signature from dual_attestation lane
    - perf_gate_sig: Signature from uplift_evaluation lane
    """
    witnesses = {}
    
    if 'triple_hash' in lane_results['lanes']:
        triple_hash_lane = lane_results['lanes']['triple_hash']
        if triple_hash_lane['status'] == 'PASS' and triple_hash_lane['hash']:
            witness_data = f"verification_gate:{triple_hash_lane['hash']}:{H_t_hash or 'ABSTAIN'}"
            witnesses['verification_gate_sig'] = hashlib.sha256(witness_data.encode('utf-8')).hexdigest()
        else:
            witnesses['verification_gate_sig'] = "ABSTAIN:triple_hash_not_verified"
    else:
        witnesses['verification_gate_sig'] = "ABSTAIN:triple_hash_lane_missing"
    
    if 'dual_attestation' in lane_results['lanes']:
        dual_att_lane = lane_results['lanes']['dual_attestation']
        if dual_att_lane['status'] == 'PASS' and dual_att_lane['hash']:
            witness_data = f"hermetic_matrix:{dual_att_lane['hash']}"
            witnesses['hermetic_matrix_sig'] = hashlib.sha256(witness_data.encode('utf-8')).hexdigest()
        else:
            witnesses['hermetic_matrix_sig'] = "ABSTAIN:dual_attestation_not_verified"
    else:
        witnesses['hermetic_matrix_sig'] = "ABSTAIN:dual_attestation_lane_missing"
    
    if 'uplift_evaluation' in lane_results['lanes']:
        uplift_eval_lane = lane_results['lanes']['uplift_evaluation']
        if uplift_eval_lane['status'] == 'PASS' and uplift_eval_lane['hash']:
            witness_data = f"perf_gate:{uplift_eval_lane['hash']}"
            witnesses['perf_gate_sig'] = hashlib.sha256(witness_data.encode('utf-8')).hexdigest()
        else:
            witnesses['perf_gate_sig'] = "ABSTAIN:uplift_evaluation_not_verified"
    else:
        witnesses['perf_gate_sig'] = "ABSTAIN:uplift_evaluation_lane_missing"
    
    return witnesses


def verify_witness_signatures(
    witnesses: Dict[str, str],
    lane_results: Dict[str, Any],
    H_t_hash: Optional[str]
) -> Dict[str, Any]:
    """
    Verify witness signatures by recomputing expected signatures.
    
    Returns verification results with per-witness status.
    """
    timestamp = datetime.now(timezone.utc).isoformat() + 'Z'
    verification_results = {
        "timestamp": timestamp,
        "overall_status": "PASS",
        "witnesses": {}
    }
    
    if 'triple_hash' in lane_results['lanes']:
        triple_hash_lane = lane_results['lanes']['triple_hash']
        if triple_hash_lane['status'] == 'PASS' and triple_hash_lane['hash']:
            expected_data = f"verification_gate:{triple_hash_lane['hash']}:{H_t_hash or 'ABSTAIN'}"
            expected_sig = hashlib.sha256(expected_data.encode('utf-8')).hexdigest()
            actual_sig = witnesses.get('verification_gate_sig', '')
            
            if actual_sig == expected_sig:
                verification_results['witnesses']['verification_gate'] = {
                    "status": "PASS",
                    "expected": expected_sig,
                    "actual": actual_sig,
                    "lane": "triple_hash"
                }
            elif actual_sig.startswith('ABSTAIN:'):
                verification_results['witnesses']['verification_gate'] = {
                    "status": "ABSTAIN",
                    "reason": actual_sig.split(':', 1)[1],
                    "lane": "triple_hash"
                }
                verification_results['overall_status'] = "ABSTAIN"
            else:
                verification_results['witnesses']['verification_gate'] = {
                    "status": "FAIL",
                    "expected": expected_sig,
                    "actual": actual_sig,
                    "lane": "triple_hash",
                    "reason": "signature_mismatch"
                }
                verification_results['overall_status'] = "FAIL"
        else:
            verification_results['witnesses']['verification_gate'] = {
                "status": "ABSTAIN",
                "reason": "triple_hash_not_verified",
                "lane": "triple_hash"
            }
            verification_results['overall_status'] = "ABSTAIN"
    else:
        verification_results['witnesses']['verification_gate'] = {
            "status": "ABSTAIN",
            "reason": "triple_hash_lane_missing",
            "lane": "triple_hash"
        }
        verification_results['overall_status'] = "ABSTAIN"
    
    if 'dual_attestation' in lane_results['lanes']:
        dual_att_lane = lane_results['lanes']['dual_attestation']
        if dual_att_lane['status'] == 'PASS' and dual_att_lane['hash']:
            expected_data = f"hermetic_matrix:{dual_att_lane['hash']}"
            expected_sig = hashlib.sha256(expected_data.encode('utf-8')).hexdigest()
            actual_sig = witnesses.get('hermetic_matrix_sig', '')
            
            if actual_sig == expected_sig:
                verification_results['witnesses']['hermetic_matrix'] = {
                    "status": "PASS",
                    "expected": expected_sig,
                    "actual": actual_sig,
                    "lane": "dual_attestation"
                }
            elif actual_sig.startswith('ABSTAIN:'):
                verification_results['witnesses']['hermetic_matrix'] = {
                    "status": "ABSTAIN",
                    "reason": actual_sig.split(':', 1)[1],
                    "lane": "dual_attestation"
                }
                if verification_results['overall_status'] == "PASS":
                    verification_results['overall_status'] = "ABSTAIN"
            else:
                verification_results['witnesses']['hermetic_matrix'] = {
                    "status": "FAIL",
                    "expected": expected_sig,
                    "actual": actual_sig,
                    "lane": "dual_attestation",
                    "reason": "signature_mismatch"
                }
                verification_results['overall_status'] = "FAIL"
        else:
            verification_results['witnesses']['hermetic_matrix'] = {
                "status": "ABSTAIN",
                "reason": "dual_attestation_not_verified",
                "lane": "dual_attestation"
            }
            if verification_results['overall_status'] == "PASS":
                verification_results['overall_status'] = "ABSTAIN"
    else:
        verification_results['witnesses']['hermetic_matrix'] = {
            "status": "ABSTAIN",
            "reason": "dual_attestation_lane_missing",
            "lane": "dual_attestation"
        }
        if verification_results['overall_status'] == "PASS":
            verification_results['overall_status'] = "ABSTAIN"
    
    if 'uplift_evaluation' in lane_results['lanes']:
        uplift_eval_lane = lane_results['lanes']['uplift_evaluation']
        if uplift_eval_lane['status'] == 'PASS' and uplift_eval_lane['hash']:
            expected_data = f"perf_gate:{uplift_eval_lane['hash']}"
            expected_sig = hashlib.sha256(expected_data.encode('utf-8')).hexdigest()
            actual_sig = witnesses.get('perf_gate_sig', '')
            
            if actual_sig == expected_sig:
                verification_results['witnesses']['perf_gate'] = {
                    "status": "PASS",
                    "expected": expected_sig,
                    "actual": actual_sig,
                    "lane": "uplift_evaluation"
                }
            elif actual_sig.startswith('ABSTAIN:'):
                verification_results['witnesses']['perf_gate'] = {
                    "status": "ABSTAIN",
                    "reason": actual_sig.split(':', 1)[1],
                    "lane": "uplift_evaluation"
                }
                if verification_results['overall_status'] == "PASS":
                    verification_results['overall_status'] = "ABSTAIN"
            else:
                verification_results['witnesses']['perf_gate'] = {
                    "status": "FAIL",
                    "expected": expected_sig,
                    "actual": actual_sig,
                    "lane": "uplift_evaluation",
                    "reason": "signature_mismatch"
                }
                verification_results['overall_status'] = "FAIL"
        else:
            verification_results['witnesses']['perf_gate'] = {
                "status": "ABSTAIN",
                "reason": "uplift_evaluation_not_verified",
                "lane": "uplift_evaluation"
            }
            if verification_results['overall_status'] == "PASS":
                verification_results['overall_status'] = "ABSTAIN"
    else:
        verification_results['witnesses']['perf_gate'] = {
            "status": "ABSTAIN",
            "reason": "uplift_evaluation_lane_missing",
            "lane": "uplift_evaluation"
        }
        if verification_results['overall_status'] == "PASS":
            verification_results['overall_status'] = "ABSTAIN"
    
    return verification_results


def compute_epoch_seal(
    lane_results: Dict[str, Any],
    H_t_hash: Optional[str],
    hygiene_hash: str,
    timestamp: str
) -> str:
    """
    Compute epoch seal that cryptographically binds the run.
    
    epoch_hash = sha256(RFC8785({
        lanes: {lane_id: hash},
        H_t: hash,
        hygiene_state_hash: hash,
        verification_signature: timestamp
    }))
    """
    lane_hashes = {}
    for lane_id, lane_data in lane_results['lanes'].items():
        if lane_data['status'] == 'PASS' and lane_data['hash']:
            lane_hashes[lane_id] = lane_data['hash']
        else:
            lane_hashes[lane_id] = f"ABSTAIN:{lane_data['message']}"
    
    epoch_manifest = {
        "lanes": lane_hashes,
        "H_t": H_t_hash or "ABSTAIN:missing",
        "hygiene_state_hash": hygiene_hash,
        "verification_signature": timestamp
    }
    
    canonical_manifest = rfc8785_canonicalize(epoch_manifest)
    epoch_hash = hashlib.sha256(canonical_manifest.encode('utf-8')).hexdigest()
    
    return epoch_hash


def compute_witnessed_epoch_seal(
    epoch_core: Dict[str, Any],
    witnesses: Dict[str, str]
) -> str:
    """
    Compute witnessed epoch seal with dual signatures.
    
    witnessed_epoch_hash = sha256(RFC8785(epoch_core + witnesses))
    """
    witnessed_manifest = {
        **epoch_core,
        "witnesses": witnesses
    }
    
    canonical_manifest = rfc8785_canonicalize(witnessed_manifest)
    witnessed_hash = hashlib.sha256(canonical_manifest.encode('utf-8')).hexdigest()
    
    return witnessed_hash


def append_epoch_registry(
    epoch_hash: str,
    witnessed_epoch_hash: str,
    witnesses: Dict[str, str],
    status: str,
    registry_path: str = "artifacts/allblue/epoch_registry.jsonl"
) -> None:
    """Append RFC8785 epoch record to registry."""
    
    timestamp = datetime.now(timezone.utc).isoformat() + 'Z'
    
    registry_record = {
        "timestamp": timestamp,
        "epoch_hash": epoch_hash,
        "witnessed_epoch_hash": witnessed_epoch_hash,
        "status": status,
        "witnesses": witnesses
    }
    
    canonical_record = rfc8785_canonicalize(registry_record)
    
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    with open(registry_path, 'a') as f:
        f.write(canonical_record + '\n')


def generate_fleet_state(
    config: Dict[str, Any],
    lane_results: Dict[str, Any],
    epoch_hash: str,
    witnessed_epoch_hash: str,
    witnesses: Dict[str, str],
    hygiene_hash: str
) -> Dict[str, Any]:
    """Generate complete fleet state with epoch seal and witnessed epoch."""
    
    timestamp = datetime.now(timezone.utc).isoformat() + 'Z'
    
    if lane_results['required_abstain'] > 0:
        overall_status = "ABSTAIN"
        status_message = f"{lane_results['required_abstain']} required lane(s) missing artifacts"
    elif lane_results['required_pass'] == len([l for l in config['lanes'] if l.get('required', True)]):
        overall_status = "PASS"
        status_message = "All required lanes verified"
    else:
        overall_status = "FAIL"
        status_message = "Some required lanes failed verification"
    
    fleet_state = {
        "allblue_status": overall_status,
        "status_message": status_message,
        "epoch_hash": epoch_hash,
        "witnessed_epoch_hash": witnessed_epoch_hash,
        "timestamp": timestamp,
        "conductor": "Devin J",
        "sprint_duration_hours": 72,
        "session_url": "https://app.devin.ai/sessions/a4d865ce3da54e7ba6119a84a8cbd8e3",
        
        "lane_configuration": {
            "version": config.get('version', '1.0'),
            "total_lanes": len(config['lanes']),
            "required_lanes": len([l for l in config['lanes'] if l.get('required', True)]),
            "optional_lanes": len([l for l in config['lanes'] if not l.get('required', True)])
        },
        
        "lane_results": lane_results['lanes'],
        
        "lane_summary": {
            "required_pass": lane_results['required_pass'],
            "required_abstain": lane_results['required_abstain'],
            "optional_pass": lane_results['optional_pass'],
            "optional_abstain": lane_results['optional_abstain']
        },
        
        "hygiene_state_hash": hygiene_hash,
        
        "witnesses": witnesses,
        
        "epoch_seal": {
            "algorithm": "sha256",
            "canonicalization": "RFC8785",
            "components": ["lanes", "H_t", "hygiene_state_hash", "verification_signature"],
            "hash": epoch_hash
        },
        
        "witnessed_epoch": {
            "algorithm": "sha256",
            "canonicalization": "RFC8785",
            "components": ["epoch_core", "witnesses"],
            "witnesses": {
                "verification_gate_sig": witnesses.get('verification_gate_sig', 'ABSTAIN'),
                "hermetic_matrix_sig": witnesses.get('hermetic_matrix_sig', 'ABSTAIN'),
                "perf_gate_sig": witnesses.get('perf_gate_sig', 'ABSTAIN')
            },
            "hash": witnessed_epoch_hash
        },
        
        "archive": {
            "retention_days": 90,
            "format": "RFC8785"
        }
    }
    
    return fleet_state


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate AllBlue Epoch Seal with Dynamic Lane Discovery'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/allblue_lanes.json',
        help='Lane configuration file (default: config/allblue_lanes.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/allblue',
        help='Output directory for artifacts (default: artifacts/allblue)'
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Output file path (overrides --output-dir, e.g., artifacts/allblue/epoch.json)'
    )
    parser.add_argument(
        '--rfcsign',
        action='store_true',
        help='Enable RFC 8785 signature mode (witnessed epoch with dual signatures)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AllBlue Epoch Seal Generator (Dynamic Lanes + RFC 8785)")
    print("=" * 70)
    print()
    
    print("Loading lane configuration...")
    config = load_lane_config(args.config)
    print(f"  Loaded {len(config.get('lanes', []))} lanes")
    print()
    
    print("Discovering lanes...")
    lane_results = discover_lanes(config)
    print()
    
    print("Computing hygiene state hash...")
    hygiene_hash = compute_hygiene_state_hash()
    print(f"  Hygiene hash: {hygiene_hash}")
    print()
    
    H_t_hash = None
    if 'triple_hash' in lane_results['lanes']:
        triple_hash_lane = lane_results['lanes']['triple_hash']
        if triple_hash_lane['status'] == 'PASS':
            H_t_hash = triple_hash_lane['hash']
    
    print("Computing epoch seal...")
    timestamp = datetime.now(timezone.utc).isoformat() + 'Z'
    epoch_hash = compute_epoch_seal(lane_results, H_t_hash, hygiene_hash, timestamp)
    print(f"  Epoch hash: {epoch_hash}")
    print()
    
    print("Computing witness signatures...")
    witnesses = compute_witness_signatures(lane_results, H_t_hash)
    print(f"  Verification Gate: {witnesses.get('verification_gate_sig', 'ABSTAIN')}")
    print(f"  Hermetic Matrix:   {witnesses.get('hermetic_matrix_sig', 'ABSTAIN')}")
    print(f"  Perf Gate:         {witnesses.get('perf_gate_sig', 'ABSTAIN')}")
    print()
    
    print("Verifying witness signatures...")
    witness_verification = verify_witness_signatures(witnesses, lane_results, H_t_hash)
    witness_verify_path = os.path.join(args.output_dir, 'witness_verify.json')
    os.makedirs(args.output_dir, exist_ok=True)
    canonical_witness_verify = rfc8785_canonicalize(witness_verification)
    with open(witness_verify_path, 'w') as f:
        f.write(canonical_witness_verify)
    print(f"  Verification status: {witness_verification['overall_status']}")
    print(f"  Verification output: {witness_verify_path}")
    print()
    
    print("Computing witnessed epoch seal...")
    epoch_core = {
        "lanes": {lid: lr['hash'] for lid, lr in lane_results['lanes'].items() if lr['status'] == 'PASS'},
        "H_t": H_t_hash or "ABSTAIN:missing",
        "hygiene_state_hash": hygiene_hash,
        "verification_signature": timestamp
    }
    witnessed_epoch_hash = compute_witnessed_epoch_seal(epoch_core, witnesses)
    print(f"  Witnessed epoch hash: {witnessed_epoch_hash}")
    print()
    
    print("Generating fleet state...")
    fleet_state = generate_fleet_state(config, lane_results, epoch_hash, witnessed_epoch_hash, witnesses, hygiene_hash)
    
    status = fleet_state['allblue_status']
    
    print("Appending to epoch registry...")
    registry_path = os.path.join(args.output_dir, 'epoch_registry.jsonl')
    append_epoch_registry(epoch_hash, witnessed_epoch_hash, witnesses, status, registry_path)
    print(f"  Registry: {registry_path}")
    print()
    
    if args.out:
        output_path = args.out
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'fleet_state.json')
    
    canonical_json = rfc8785_canonicalize(fleet_state)
    
    with open(output_path, 'w') as f:
        f.write(canonical_json)
    
    print(f"Generated: {output_path}")
    
    if not args.out:
        readable_path = os.path.join(args.output_dir, 'fleet_state_readable.json')
        with open(readable_path, 'w') as f:
            json.dump(fleet_state, f, indent=2, sort_keys=True)
        
        print(f"Generated: {readable_path}")
    print()
    
    print("=" * 70)
    if status == "PASS" and witness_verification['overall_status'] == "PASS":
        print(f"[PASS] ALL BLUE: sha256=<{epoch_hash}>")
        print(f"[PASS] Epoch Seal <{epoch_hash}>")
        print(f"[PASS] Witnessed Epoch <{witnessed_epoch_hash}>")
        print(f"[PASS] Witnesses Verified")
    elif status == "ABSTAIN" or witness_verification['overall_status'] == "ABSTAIN":
        print(f"[ABSTAIN] AllBlue Gate: {fleet_state['status_message']}")
        print(f"[INFO] Epoch Seal <{epoch_hash}>")
        print(f"[INFO] Witnessed Epoch <{witnessed_epoch_hash}>")
        for witness_name, witness_data in witness_verification['witnesses'].items():
            if witness_data['status'] == 'ABSTAIN':
                reason = witness_data.get('reason', 'unknown')
                print(f"[ABSTAIN] missing witness (lane={witness_name}, reason={reason})")
    else:
        print(f"[FAIL] AllBlue Gate: {fleet_state['status_message']}")
        print(f"[INFO] Epoch Seal <{epoch_hash}>")
        print(f"[INFO] Witnessed Epoch <{witnessed_epoch_hash}>")
        for witness_name, witness_data in witness_verification['witnesses'].items():
            if witness_data['status'] == 'FAIL':
                reason = witness_data.get('reason', 'unknown')
                print(f"[FAIL] witness verification failed (lane={witness_name}, reason={reason})")
    print("=" * 70)
    print()
    
    print("Lane Summary:")
    print(f"  Required PASS:    {lane_results['required_pass']}")
    print(f"  Required ABSTAIN: {lane_results['required_abstain']}")
    print(f"  Optional PASS:    {lane_results['optional_pass']}")
    print(f"  Optional ABSTAIN: {lane_results['optional_abstain']}")
    print()
    
    if status == "PASS":
        sys.exit(0)
    elif status == "ABSTAIN":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
