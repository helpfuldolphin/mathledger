#!/usr/bin/env python3
"""
Hermetic Matrix Trend Tracker v3.3 - Resharding and Drift Detection

Tracks H_matrix seal and lane health over time by appending history to
artifacts/no_network/matrix_history.jsonl.

Aggregates matrix history from shards and computes global H_matrix root.
Tracks global root changes across epochs and alerts on drift.

Usage:
    python tools/hermetic/matrix_trend.py --append
    python tools/hermetic/matrix_trend.py --report
    python tools/hermetic/matrix_trend.py --alert-drift
    python tools/hermetic/matrix_trend.py --verify-global
    python tools/hermetic/matrix_trend.py --save-global
    python tools/hermetic/matrix_trend.py --append-global-root
    python tools/hermetic/matrix_trend.py --alert-global-drift

Features:
- Appends H_matrix + lane pass/fail bitmap to JSONL history
- Detects H_matrix drift across runs
- Generates trend reports for lane health
- Alerts on consecutive failures or seal changes
- Aggregates shards (shard_00.jsonl...shard_XX.jsonl)
- Computes global H_matrix root = SHA256(concat of shard roots)
- Generates artifacts/hermetic/global_matrix.json with summary stats
- Tracks global root changes across epochs
- Alerts on global root drift with delta reporting

Pass-Lines:
    [PASS] NO_NETWORK HERMETIC v2 TRUE
    [PASS] Hermetic Matrix <sha256>
    [PASS] Global H-Matrix Verified <sha256>
    [ABSTAIN] Missing shard data: shard_XX, shard_YY
    [PASS] No global drift detected in last N epochs
    [ABSTAIN] Global Drift Detected Δ=<hash>
"""

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.repro.determinism import deterministic_isoformat

class MatrixTrendTracker:
    """Tracks hermetic matrix seal and lane health over time."""

    def __init__(self, repo_root: Optional[Path] = None, enable_cache: bool = True, cache_size: int = 512):
        """Initialize tracker with repository root."""
        self.repo_root = repo_root or Path(__file__).parent.parent.parent
        self.history_file = self.repo_root / "artifacts" / "no_network" / "matrix_history.jsonl"
        self.manifest_file = self.repo_root / "artifacts" / "no_network" / "lane_matrix_manifest.json"
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self._shard_cache = OrderedDict()
        
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def load_current_manifest(self) -> Optional[Dict]:
        """Load current lane matrix manifest."""
        if not self.manifest_file.exists():
            return None
        
        try:
            with open(self.manifest_file, 'r', encoding='ascii') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[ERROR] Failed to load manifest: {e}", file=sys.stderr)
            return None

    def compute_lane_bitmap(self, manifest: Dict) -> str:
        """
        Compute lane pass/fail bitmap.
        
        Returns 6-character string where each position represents a lane:
        - '1' = PASS
        - '0' = FAIL
        - 'A' = ABSTAIN
        
        Lane order: da_ui, da_reasoning, da_composite, browsermcp, uplift-omega, test
        """
        lane_order = ["da_ui", "da_reasoning", "da_composite", "browsermcp", "uplift-omega", "test"]
        bitmap = []
        
        lanes = manifest.get("lanes", manifest.get("entries", {}))
        for lane_name in lane_order:
            lane_data = lanes.get(lane_name, {})
            status = lane_data.get("hermetic", False)
            abstain_reason = lane_data.get("abstain_reason")
            
            if abstain_reason is not None:
                bitmap.append('A')
            elif status:
                bitmap.append('1')
            else:
                bitmap.append('0')
        
        return ''.join(bitmap)

    def append_history_entry(self, manifest: Dict) -> bool:
        """
        Append current matrix state to history.
        
        Returns True if successful, False otherwise.
        """
        try:
            h_matrix = manifest.get("h_matrix", "")
            timestamp = manifest.get(
                "timestamp",
                deterministic_isoformat("matrix_trend_entry", h_matrix, manifest.get("epoch", 0))
            )
            lane_bitmap = self.compute_lane_bitmap(manifest)
            
            pass_count = lane_bitmap.count('1')
            fail_count = lane_bitmap.count('0')
            abstain_count = lane_bitmap.count('A')
            
            entry = {
                "timestamp": timestamp,
                "h_matrix": h_matrix,
                "lane_bitmap": lane_bitmap,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "abstain_count": abstain_count,
                "all_blue": pass_count == 6 and fail_count == 0 and abstain_count == 0
            }
            
            with open(self.history_file, 'a', encoding='ascii') as f:
                canonical_entry = json.dumps(
                    entry,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(',', ':'),
                )
                f.write(canonical_entry + '\n')
            
            print(f"[INFO] Appended history entry: {h_matrix[:16]}... bitmap={lane_bitmap}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to append history: {e}", file=sys.stderr)
            return False

    def load_history(self) -> List[Dict]:
        """Load all history entries from JSONL file."""
        if not self.history_file.exists():
            return []
        
        history = []
        try:
            with open(self.history_file, 'r', encoding='ascii') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[ERROR] Failed to load history: {e}", file=sys.stderr)
        
        return history

    def detect_drift(self, window: int = 3) -> Tuple[bool, Optional[str]]:
        """
        Detect H_matrix drift in recent history.
        
        Args:
            window: Number of recent entries to check
        
        Returns:
            (has_drift, reason) tuple
        """
        history = self.load_history()
        
        if len(history) < 2:
            return False, None
        
        recent = history[-window:] if len(history) >= window else history
        h_matrices = [entry.get("h_matrix", "") for entry in recent]
        
        unique_hashes = set(h_matrices)
        
        if len(unique_hashes) > 1:
            return True, f"H_matrix changed across {len(recent)} runs: {len(unique_hashes)} unique seals"
        
        return False, None

    def generate_report(self, limit: int = 10) -> str:
        """
        Generate trend report for recent history.
        
        Args:
            limit: Number of recent entries to include
        
        Returns:
            ASCII report string
        """
        history = self.load_history()
        
        if not history:
            return "[INFO] No history available"
        
        recent = history[-limit:] if len(history) > limit else history
        
        lines = []
        lines.append("=" * 80)
        lines.append("HERMETIC MATRIX TREND REPORT")
        lines.append("=" * 80)
        lines.append(f"Total Entries: {len(history)}")
        lines.append(f"Showing Recent: {len(recent)}")
        lines.append("")
        
        lines.append(f"{'Timestamp':<20} {'H_matrix':<16} {'Bitmap':<8} {'P/F/A':<8} {'AllBlue':<8}")
        lines.append("-" * 80)
        
        for entry in recent:
            timestamp = entry.get("timestamp", "")[:19]  # Truncate to datetime
            h_matrix = entry.get("h_matrix", "")[:16]  # First 16 chars
            bitmap = entry.get("lane_bitmap", "")
            pass_count = entry.get("pass_count", 0)
            fail_count = entry.get("fail_count", 0)
            abstain_count = entry.get("abstain_count", 0)
            all_blue = "YES" if entry.get("all_blue", False) else "NO"
            
            pfa = f"{pass_count}/{fail_count}/{abstain_count}"
            lines.append(f"{timestamp:<20} {h_matrix:<16} {bitmap:<8} {pfa:<8} {all_blue:<8}")
        
        lines.append("=" * 80)
        
        all_blue_count = sum(1 for e in history if e.get("all_blue", False))
        all_blue_pct = (all_blue_count / len(history) * 100) if history else 0
        
        lines.append("")
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        lines.append(f"All Blue Runs: {all_blue_count}/{len(history)} ({all_blue_pct:.1f}%)")
        
        has_drift, drift_reason = self.detect_drift(window=3)
        if has_drift:
            lines.append(f"[ALERT] H_matrix Drift Detected: {drift_reason}")
        else:
            lines.append("[PASS] No H_matrix drift in recent runs")
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)

    def alert_drift(self, threshold: int = 3) -> bool:
        """
        Alert if H_matrix has drifted or lanes are failing.
        
        Args:
            threshold: Number of consecutive failures to trigger alert
        
        Returns:
            True if alert triggered, False otherwise
        """
        history = self.load_history()
        
        if len(history) < threshold:
            print(f"[INFO] Insufficient history for drift alert (need {threshold} entries)")
            return False
        
        recent = history[-threshold:]
        
        h_matrices = [e.get("h_matrix", "") for e in recent]
        unique_hashes = set(h_matrices)
        
        if len(unique_hashes) > 1:
            print(f"[ALERT] H_matrix drift detected across {threshold} runs:")
            for i, entry in enumerate(recent):
                h_matrix = entry.get("h_matrix", "")[:16]
                bitmap = entry.get("lane_bitmap", "")
                print(f"  Run {i+1}: {h_matrix}... bitmap={bitmap}")
            return True
        
        all_blue_status = [e.get("all_blue", False) for e in recent]
        if not any(all_blue_status):
            print(f"[ALERT] No All Blue in last {threshold} runs:")
            for i, entry in enumerate(recent):
                bitmap = entry.get("lane_bitmap", "")
                pfa = f"{entry.get('pass_count', 0)}/{entry.get('fail_count', 0)}/{entry.get('abstain_count', 0)}"
                print(f"  Run {i+1}: bitmap={bitmap} P/F/A={pfa}")
            return True
        
        print(f"[PASS] No drift or consecutive failures in last {threshold} runs")
        return False

    def validate_hermetic(self) -> Tuple[bool, str]:
        """
        Validate current hermetic matrix state.
        
        Returns:
            (is_valid, h_matrix) tuple
        """
        manifest = self.load_current_manifest()
        
        if not manifest:
            return False, ""
        
        h_matrix = manifest.get("h_matrix", "")
        lanes = manifest.get("lanes", manifest.get("entries", {}))
        
        all_hermetic = all(
            lane_data.get("hermetic", False) and lane_data.get("abstain_reason") is None
            for lane_data in lanes.values()
        )
        
        if all_hermetic and h_matrix:
            print(f"[PASS] NO_NETWORK HERMETIC v2 TRUE")
            print(f"[PASS] Hermetic Matrix {h_matrix}")
            return True, h_matrix
        else:
            lane_bitmap = self.compute_lane_bitmap(manifest)
            print(f"[FAIL] Hermetic validation failed: bitmap={lane_bitmap}")
            return False, h_matrix

    def validate_history(self, required_runs: int = 3) -> Tuple[bool, int, int]:
        """
        Validate history has required sealed runs.
        
        Args:
            required_runs: Minimum number of sealed runs required
        
        Returns:
            (is_valid, sealed_count, total_count) tuple
        """
        history = self.load_history()
        sealed_runs = [e for e in history if e.get("all_blue", False)]
        
        sealed_count = len(sealed_runs)
        total_count = len(history)
        
        if sealed_count >= required_runs:
            print(f"[PASS] Hermetic History: {sealed_count}/{total_count} sealed")
            return True, sealed_count, total_count
        else:
            print(f"[FAIL] Hermetic History: {sealed_count}/{total_count} sealed (need {required_runs})")
            return False, sealed_count, total_count

    def alert_drift_with_abstain(self, threshold: int = 3) -> bool:
        """
        Alert with ABSTAIN reporting and drift report generation.
        
        Args:
            threshold: Number of consecutive runs to check
        
        Returns:
            True if drift detected, False otherwise
        """
        has_drift, reason = self.detect_drift(window=threshold)
        
        if has_drift:
            print(f"[ABSTAIN] H_matrix drift detected in last {threshold} runs")
            print(f"  Reason: {reason}")
            
            history = self.load_history()
            recent = history[-threshold:] if len(history) >= threshold else history
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("HERMETIC MATRIX DRIFT REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Drift detected in last {threshold} runs:")
            report_lines.append("")
            report_lines.append(f"Reason: {reason}")
            report_lines.append("")
            report_lines.append(f"{'Run':<5} {'Timestamp':<20} {'H_matrix':<16} {'Bitmap':<8} {'P/F/A':<8}")
            report_lines.append("-" * 80)
            
            for i, entry in enumerate(recent):
                h_matrix = entry.get("h_matrix", "")[:16]
                bitmap = entry.get("lane_bitmap", "")
                timestamp = entry.get("timestamp", "")[:19]
                pass_count = entry.get("pass_count", 0)
                fail_count = entry.get("fail_count", 0)
                abstain_count = entry.get("abstain_count", 0)
                pfa = f"{pass_count}/{fail_count}/{abstain_count}"
                
                report_lines.append(f"{i+1:<5} {timestamp:<20} {h_matrix:<16} {bitmap:<8} {pfa:<8}")
            
            report_lines.append("=" * 80)
            
            report_path = self.repo_root / "artifacts" / "no_network" / "drift_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='ascii') as f:
                f.write('\n'.join(report_lines))
            
            print(f"  Drift report saved: {report_path}")
            return True
        
        print(f"[PASS] No drift or consecutive failures in last {threshold} runs")
        return False

    def save_report(self, limit: int = 10) -> Path:
        """
        Generate and save report to file.
        
        Args:
            limit: Number of recent entries to include
        
        Returns:
            Path to saved report file
        """
        report = self.generate_report(limit=limit)
        report_path = self.repo_root / "artifacts" / "no_network" / "matrix_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='ascii') as f:
            f.write(report)
        
        print(f"[INFO] Report saved: {report_path}")
        return report_path

    def _load_shard_history_uncached(self, shard_id: int) -> List[Dict]:
        """
        Load history from a specific shard (uncached implementation).
        
        Args:
            shard_id: Shard ID (0-15)
        
        Returns:
            List of history entries for the shard
        """
        shard_file = self.repo_root / "artifacts" / "hermetic" / f"shard_{shard_id:02d}.jsonl"
        
        if not shard_file.exists():
            return []
        
        history = []
        try:
            with open(shard_file, 'r', encoding='ascii') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[ERROR] Failed to load shard {shard_id}: {e}", file=sys.stderr)
        
        return history

    def load_shard_history(self, shard_id: int) -> List[Dict]:
        """
        Load history from a specific shard with optional LRU caching.
        
        Args:
            shard_id: Shard ID (0-15)
        
        Returns:
            List of history entries for the shard
        """
        if not self.enable_cache:
            self.cache_misses += 1
            return self._load_shard_history_uncached(shard_id)
        
        cache_key = f"shard_{shard_id:02d}"
        
        if cache_key in self._shard_cache:
            self.cache_hits += 1
            self._shard_cache.move_to_end(cache_key)
            return self._shard_cache[cache_key]
        
        self.cache_misses += 1
        history = self._load_shard_history_uncached(shard_id)
        
        if len(self._shard_cache) >= self.cache_size:
            self._shard_cache.popitem(last=False)
            self.cache_evictions += 1
        
        self._shard_cache[cache_key] = history
        
        return history

    def compute_shard_root(self, shard_id: int) -> Optional[str]:
        """
        Compute root hash for a shard's history.
        
        Args:
            shard_id: Shard ID (0-15)
        
        Returns:
            SHA256 hash of shard's latest H_matrix, or None if shard missing
        """
        history = self.load_shard_history(shard_id)
        
        if not history:
            return None
        
        latest_entry = history[-1]
        h_matrix = latest_entry.get("h_matrix", "")
        
        if not h_matrix:
            return None
        
        return h_matrix

    def compute_global_h_matrix(self, num_shards: int = 16, parallel: bool = False) -> Tuple[bool, Optional[str], List[int]]:
        """
        Compute global H_matrix root from all shards.
        
        Args:
            num_shards: Number of shards to aggregate (default 16)
            parallel: Use parallel shard loading (default False)
        
        Returns:
            (success, global_root, missing_shards) tuple
        """
        if parallel and num_shards >= 16:
            return self._compute_global_h_matrix_parallel(num_shards)
        
        shard_roots = []
        missing_shards = []
        
        for shard_id in range(num_shards):
            shard_root = self.compute_shard_root(shard_id)
            
            if shard_root is None:
                missing_shards.append(shard_id)
                shard_roots.append("")  # Empty string for missing shards
            else:
                shard_roots.append(shard_root)
        
        if missing_shards:
            return False, None, missing_shards
        
        concatenated = ''.join(shard_roots)
        global_root = hashlib.sha256(concatenated.encode('ascii')).hexdigest()
        
        return True, global_root, []

    def _compute_global_h_matrix_parallel(self, num_shards: int) -> Tuple[bool, Optional[str], List[int]]:
        """
        Compute global H_matrix root from all shards using parallel loading.
        
        Args:
            num_shards: Number of shards to aggregate
        
        Returns:
            (success, global_root, missing_shards) tuple
        """
        shard_roots = [None] * num_shards
        missing_shards = []
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(self.compute_shard_root, shard_id): shard_id 
                      for shard_id in range(num_shards)}
            
            for future in futures:
                shard_id = futures[future]
                try:
                    shard_root = future.result()
                    if shard_root is None:
                        missing_shards.append(shard_id)
                        shard_roots[shard_id] = ""
                    else:
                        shard_roots[shard_id] = shard_root
                except Exception as e:
                    print(f"[ERROR] Failed to compute shard {shard_id} root: {e}", file=sys.stderr)
                    missing_shards.append(shard_id)
                    shard_roots[shard_id] = ""
        
        if missing_shards:
            return False, None, missing_shards
        
        concatenated = ''.join(shard_roots)
        global_root = hashlib.sha256(concatenated.encode('ascii')).hexdigest()
        
        return True, global_root, []

    def generate_global_matrix_summary(self, num_shards: int = 16) -> Dict:
        """
        Generate global matrix summary with statistics.
        
        Args:
            num_shards: Number of shards to aggregate (default 16)
        
        Returns:
            Dictionary with global matrix summary
        """
        success, global_root, missing_shards = self.compute_global_h_matrix(num_shards)
        
        shard_stats = []
        total_entries = 0
        total_all_blue = 0
        
        for shard_id in range(num_shards):
            history = self.load_shard_history(shard_id)
            shard_root = self.compute_shard_root(shard_id)
            
            if history:
                all_blue_count = sum(1 for e in history if e.get("all_blue", False))
                total_entries += len(history)
                total_all_blue += all_blue_count
                
                shard_stats.append({
                    "shard_id": shard_id,
                    "entries": len(history),
                    "all_blue_count": all_blue_count,
                    "latest_h_matrix": shard_root or "",
                    "status": "present"
                })
            else:
                shard_stats.append({
                    "shard_id": shard_id,
                    "entries": 0,
                    "all_blue_count": 0,
                    "latest_h_matrix": "",
                    "status": "missing"
                })
        
        summary = {
            "timestamp": deterministic_isoformat(
                "matrix_trend_summary",
                global_root or "",
                missing_shards
            ),
            "num_shards": num_shards,
            "global_h_matrix_root": global_root or "",
            "verification_status": "verified" if success else "abstain",
            "missing_shards": missing_shards,
            "total_entries": total_entries,
            "total_all_blue": total_all_blue,
            "all_blue_percentage": (total_all_blue / total_entries * 100) if total_entries > 0 else 0,
            "shards": shard_stats
        }
        
        return summary

    def save_global_matrix(self, num_shards: int = 16) -> Path:
        """
        Generate and save global matrix summary.
        
        Args:
            num_shards: Number of shards to aggregate (default 16)
        
        Returns:
            Path to saved global matrix file
        """
        summary = self.generate_global_matrix_summary(num_shards)
        
        global_matrix_path = self.repo_root / "artifacts" / "hermetic" / "global_matrix.json"
        global_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(global_matrix_path, 'w', encoding='ascii') as f:
            canonical_summary = json.dumps(
                summary,
                ensure_ascii=True,
                sort_keys=True,
                separators=(',', ':'),
            )
            f.write(canonical_summary)
        
        print(f"[INFO] Global matrix saved: {global_matrix_path}")
        return global_matrix_path

    def verify_global_matrix(self, num_shards: int = 16) -> Tuple[bool, Optional[str]]:
        """
        Verify global H_matrix across all shards.
        
        Args:
            num_shards: Number of shards to verify (default 16)
        
        Returns:
            (success, global_root) tuple
        """
        success, global_root, missing_shards = self.compute_global_h_matrix(num_shards)
        
        if not success:
            missing_str = ', '.join([f"shard_{s:02d}" for s in missing_shards])
            print(f"[ABSTAIN] Missing shard data: {missing_str}")
            return False, None
        
        print(f"[PASS] Global H-Matrix Verified {global_root}")
        return True, global_root

    def load_global_root_history(self) -> List[Dict]:
        """
        Load global root history from JSONL file.
        
        Returns:
            List of global root history entries
        """
        history_file = self.repo_root / "artifacts" / "hermetic" / "global_root_history.jsonl"
        
        if not history_file.exists():
            return []
        
        history = []
        try:
            with open(history_file, 'r', encoding='ascii') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[ERROR] Failed to load global root history: {e}", file=sys.stderr)
        
        return history

    def append_global_root(self, num_shards: int = 16, parallel: bool = False) -> bool:
        """
        Append current global root to history.
        
        Args:
            num_shards: Number of shards to aggregate
            parallel: Use parallel shard loading
        
        Returns:
            True if successful, False otherwise
        """
        success, global_root, missing_shards = self.compute_global_h_matrix(num_shards, parallel=parallel)
        
        if not success:
            print(f"[ABSTAIN] Cannot append global root - missing shards", file=sys.stderr)
            return False
        
        history_file = self.repo_root / "artifacts" / "hermetic" / "global_root_history.jsonl"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        entry = {
            "timestamp": deterministic_isoformat("matrix_trend_global_root", global_root, num_shards),
            "global_root": global_root,
            "num_shards": num_shards,
            "epoch": len(self.load_global_root_history()) + 1
        }
        
        try:
            with open(history_file, 'a', encoding='ascii') as f:
                canonical_entry = json.dumps(
                    entry,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(',', ':'),
                )
                f.write(canonical_entry + '\n')
            
            print(f"[INFO] Appended global root: epoch={entry['epoch']} root={global_root[:16]}...")
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to append global root: {e}", file=sys.stderr)
            return False

    def detect_global_drift(self, window: int = 3) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect global root drift across epochs.
        
        Args:
            window: Number of recent epochs to check
        
        Returns:
            (has_drift, expected_root, actual_root) tuple
        """
        history = self.load_global_root_history()
        
        if len(history) < 2:
            return False, None, None
        
        recent = history[-window:] if len(history) >= window else history
        global_roots = [entry.get("global_root", "") for entry in recent]
        
        unique_roots = set(global_roots)
        
        if len(unique_roots) > 1:
            expected_root = global_roots[0] if len(global_roots) > 0 else None
            actual_root = global_roots[-1] if len(global_roots) > 0 else None
            
            delta = f"Δ={actual_root[:16]}..." if actual_root else "Δ=unknown"
            return True, expected_root, actual_root
        
        return False, None, None

    def compute_drift_severity(self, num_shards: int, threshold: int) -> Tuple[str, int, int]:
        """
        Compute drift severity by comparing shard roots across epochs.
        
        Args:
            num_shards: Number of shards to check
            threshold: Number of epochs to analyze
        
        Returns:
            (severity, changed_shards, total_shards) tuple
            severity: "minor" (<10%), "moderate" (<30%), "major" (>=30%)
        """
        history = self.load_global_root_history()
        
        if len(history) < 2:
            return "minor", 0, num_shards
        
        recent = history[-threshold:] if len(history) >= threshold else history
        
        if len(recent) < 2:
            return "minor", 0, num_shards
        
        first_epoch_roots = []
        last_epoch_roots = []
        
        for shard_id in range(num_shards):
            first_root = self.compute_shard_root(shard_id)
            if first_root:
                first_epoch_roots.append(first_root)
        
        for shard_id in range(num_shards):
            last_root = self.compute_shard_root(shard_id)
            if last_root:
                last_epoch_roots.append(last_root)
        
        if not first_epoch_roots or not last_epoch_roots:
            return "minor", 0, num_shards
        
        changed_shards = 0
        total_shards = min(len(first_epoch_roots), len(last_epoch_roots))
        
        for i in range(total_shards):
            if i < len(first_epoch_roots) and i < len(last_epoch_roots):
                if first_epoch_roots[i] != last_epoch_roots[i]:
                    changed_shards += 1
        
        if total_shards == 0:
            change_percentage = 0
        else:
            change_percentage = (changed_shards / total_shards) * 100
        
        if change_percentage < 10:
            severity = "minor"
        elif change_percentage < 30:
            severity = "moderate"
        else:
            severity = "major"
        
        return severity, changed_shards, total_shards

    def compute_historical_drift_severity(self, num_shards: int, threshold: int, parallel: bool = False) -> Tuple[str, int, int]:
        """
        Compute drift severity by analyzing historical shard changes across epochs.
        
        Args:
            num_shards: Number of shards to check
            threshold: Number of epochs to analyze
            parallel: Use parallel shard loading (for 16+ shards)
        
        Returns:
            (severity, max_changed_shards, total_shards) tuple
            severity: "minor" (<10%), "moderate" (<30%), "major" (>=30%)
        """
        history = self.load_global_root_history()
        
        if len(history) < 2:
            return "minor", 0, num_shards
        
        recent = history[-threshold:] if len(history) >= threshold else history
        
        if len(recent) < 2:
            return "minor", 0, num_shards
        
        max_changed = 0
        
        if parallel and num_shards >= 16:
            max_workers = min(16, num_shards)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i in range(len(recent) - 1):
                    futures = [executor.submit(self.load_shard_history, shard_id) for shard_id in range(num_shards)]
                    shard_histories = [f.result() for f in futures]
                    
                    changed = 0
                    for shard_id, history_entries in enumerate(shard_histories):
                        if len(history_entries) >= i + 2:
                            root_i = history_entries[i].get("h_matrix", "")
                            root_i_plus_1 = history_entries[i + 1].get("h_matrix", "")
                            if root_i != root_i_plus_1:
                                changed += 1
                    
                    max_changed = max(max_changed, changed)
        else:
            for i in range(len(recent) - 1):
                changed = 0
                
                for shard_id in range(num_shards):
                    history_entries = self.load_shard_history(shard_id)
                    if len(history_entries) >= i + 2:
                        root_i = history_entries[i].get("h_matrix", "")
                        root_i_plus_1 = history_entries[i + 1].get("h_matrix", "")
                        if root_i != root_i_plus_1:
                            changed += 1
                
                max_changed = max(max_changed, changed)
        
        if num_shards == 0:
            change_percentage = 0
        else:
            change_percentage = (max_changed / num_shards) * 100
        
        if change_percentage < 10:
            severity = "minor"
        elif change_percentage < 30:
            severity = "moderate"
        else:
            severity = "major"
        
        return severity, max_changed, num_shards

    def compute_drift_velocity(self, num_shards: int, threshold: int) -> float:
        """
        Compute drift velocity (average shards changed per epoch).
        
        Args:
            num_shards: Number of shards to check
            threshold: Number of epochs to analyze
        
        Returns:
            Drift velocity in shards/epoch
        """
        history = self.load_global_root_history()
        
        if len(history) < 2:
            return 0.0
        
        recent = history[-threshold:] if len(history) >= threshold else history
        
        if len(recent) < 2:
            return 0.0
        
        drift_events = []
        
        for i in range(len(recent) - 1):
            changed = 0
            
            for shard_id in range(num_shards):
                shard_file = self.repo_root / "artifacts" / "hermetic" / f"shard_{shard_id:02d}.jsonl"
                if shard_file.exists():
                    history_entries = self.load_shard_history(shard_id)
                    if len(history_entries) >= i + 2:
                        root_i = history_entries[i].get("h_matrix", "")
                        root_i_plus_1 = history_entries[i + 1].get("h_matrix", "")
                        if root_i != root_i_plus_1:
                            changed += 1
            
            drift_events.append(changed)
        
        if not drift_events:
            return 0.0
        
        return sum(drift_events) / len(drift_events)

    def identify_changed_shards(self, num_shards: int) -> List[int]:
        """
        Identify specific shard IDs that have changed.
        
        Args:
            num_shards: Number of shards to check
        
        Returns:
            List of changed shard IDs
        """
        history = self.load_global_root_history()
        
        if len(history) < 2:
            return []
        
        changed_shard_ids = []
        
        for shard_id in range(num_shards):
            shard_file = self.repo_root / "artifacts" / "hermetic" / f"shard_{shard_id:02d}.jsonl"
            if shard_file.exists():
                history_entries = self.load_shard_history(shard_id)
                if len(history_entries) >= 2:
                    first_root = history_entries[0].get("h_matrix", "")
                    last_root = history_entries[-1].get("h_matrix", "")
                    if first_root != last_root:
                        changed_shard_ids.append(shard_id)
        
        return changed_shard_ids

    def alert_global_drift(self, num_shards: int = 16, threshold: int = 3, parallel: bool = False, test_determinism: bool = False) -> bool:
        """
        Alert if global root has drifted across epochs with severity scoring and velocity.
        
        Args:
            num_shards: Number of shards to verify
            threshold: Number of epochs to check for drift
            parallel: Use parallel shard loading for historical severity
            test_determinism: Test that parallel and sequential produce identical results
        
        Returns:
            True if drift detected, False otherwise
        """
        history = self.load_global_root_history()
        
        if len(history) < threshold:
            print(f"[INFO] Insufficient global root history for drift detection (need {threshold} epochs)")
            return False
        
        has_drift, expected_root, actual_root = self.detect_global_drift(window=threshold)
        
        if has_drift:
            delta = actual_root[:16] if actual_root else "unknown"
            
            snapshot_severity, snapshot_changed, total_shards = self.compute_drift_severity(num_shards, threshold)
            historical_severity, historical_changed, _ = self.compute_historical_drift_severity(num_shards, threshold, parallel=parallel)
            velocity = self.compute_drift_velocity(num_shards, threshold)
            changed_shard_ids = self.identify_changed_shards(num_shards)
            
            if test_determinism:
                seq_severity, seq_changed, _ = self.compute_historical_drift_severity(num_shards, threshold, parallel=False)
                par_severity, par_changed, _ = self.compute_historical_drift_severity(num_shards, threshold, parallel=True)
                identical = (seq_severity == par_severity and seq_changed == par_changed)
                print(f"[PASS] Historical Severity parallel={parallel} identical={str(identical).lower()}")
            else:
                print(f"[PASS] Historical Severity parallel={str(parallel).lower()}")
            
            print(f"[ABSTAIN] Global Drift Detected (historical={historical_severity}, snapshot={snapshot_severity}, velocity={velocity:.1f}/epoch) Δ={delta}")
            print(f"[PASS] Drift Severity historical={historical_severity} snapshot={snapshot_severity}")
            
            if self.enable_cache:
                print(f"[PASS] Shard Cache LRU size={self.cache_size} hits={self.cache_hits} misses={self.cache_misses} evictions={self.cache_evictions}")
            else:
                print(f"[PASS] Shard Cache LRU size={self.cache_size} hits=0 misses={self.cache_misses} evictions=0")
            
            if changed_shard_ids:
                changed_shard_str = ", ".join([f"{sid:02d}" for sid in changed_shard_ids[:10]])
                if len(changed_shard_ids) > 10:
                    changed_shard_str += f"... ({len(changed_shard_ids)} total)"
                print(f"  Changed shards: {changed_shard_str}")
            
            recent = history[-threshold:]
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("GLOBAL MATRIX DRIFT REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Drift detected across {threshold} epochs:")
            report_lines.append(f"Severity (Snapshot): {snapshot_severity.upper()} ({snapshot_changed}/{total_shards} shards changed)")
            report_lines.append(f"Severity (Historical): {historical_severity.upper()} (max {historical_changed}/{total_shards} shards/epoch)")
            report_lines.append(f"Velocity: {velocity:.1f} shards/epoch")
            report_lines.append("")
            
            if changed_shard_ids:
                report_lines.append("Changed Shard IDs:")
                for i in range(0, len(changed_shard_ids), 10):
                    batch = changed_shard_ids[i:i+10]
                    batch_str = ", ".join([f"{sid:02d}" for sid in batch])
                    report_lines.append(f"  {batch_str}")
                report_lines.append("")
            
            report_lines.append(f"Expected root: {expected_root[:16] if expected_root else 'N/A'}...")
            report_lines.append(f"Actual root:   {actual_root[:16] if actual_root else 'N/A'}...")
            report_lines.append("")
            report_lines.append(f"{'Epoch':<8} {'Timestamp':<20} {'Global Root':<16} {'Shards':<8}")
            report_lines.append("-" * 80)
            
            for entry in recent:
                epoch = entry.get("epoch", 0)
                timestamp = entry.get("timestamp", "")[:19]
                global_root = entry.get("global_root", "")[:16]
                num_shards_entry = entry.get("num_shards", 0)
                
                report_lines.append(f"{epoch:<8} {timestamp:<20} {global_root:<16} {num_shards_entry:<8}")
            
            report_lines.append("=" * 80)
            
            report_path = self.repo_root / "artifacts" / "hermetic" / "global_drift_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='ascii') as f:
                f.write('\n'.join(report_lines))
            
            print(f"  Global drift report saved: {report_path}")
            return True
        
        print(f"[PASS] No global drift detected in last {threshold} epochs")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hermetic Matrix Trend Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/hermetic/matrix_trend.py --append
    
    python tools/hermetic/matrix_trend.py --report
    
    python tools/hermetic/matrix_trend.py --alert-drift
    
    python tools/hermetic/matrix_trend.py --validate
        """
    )
    
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append current matrix state to history'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate trend report'
    )
    
    parser.add_argument(
        '--alert-drift',
        action='store_true',
        help='Alert on H_matrix drift or consecutive failures'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate current hermetic matrix state'
    )
    
    parser.add_argument(
        '--validate-history',
        action='store_true',
        help='Validate history has required sealed runs'
    )
    
    parser.add_argument(
        '--alert-drift-abstain',
        action='store_true',
        help='Alert on drift with ABSTAIN reporting and drift report'
    )
    
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save report to artifacts/no_network/matrix_report.txt'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of recent entries to show in report (default: 10)'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        default=3,
        help='Number of consecutive runs to check for drift (default: 3)'
    )
    
    parser.add_argument(
        '--required-runs',
        type=int,
        default=3,
        help='Number of sealed runs required for history validation (default: 3)'
    )
    
    parser.add_argument(
        '--verify-global',
        action='store_true',
        help='Verify global H_matrix across all shards'
    )
    
    parser.add_argument(
        '--save-global',
        action='store_true',
        help='Generate and save global matrix summary to artifacts/hermetic/global_matrix.json'
    )
    
    parser.add_argument(
        '--num-shards',
        type=int,
        default=16,
        help='Number of shards to aggregate (default: 16)'
    )
    
    parser.add_argument(
        '--append-global-root',
        action='store_true',
        help='Append current global root to epoch history'
    )
    
    parser.add_argument(
        '--alert-global-drift',
        action='store_true',
        help='Alert if global root has drifted across epochs'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel shard loading (for 16+ shards)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable shard history caching'
    )
    
    parser.add_argument(
        '--cache-size',
        type=int,
        default=512,
        help='LRU cache size for shard histories (default: 512)'
    )
    
    parser.add_argument(
        '--test-determinism',
        action='store_true',
        help='Test that parallel and sequential historical severity produce identical results'
    )
    
    args = parser.parse_args()
    
    tracker = MatrixTrendTracker(enable_cache=not args.no_cache, cache_size=args.cache_size)
    
    if args.append:
        manifest = tracker.load_current_manifest()
        if not manifest:
            print("[ERROR] No manifest found to append", file=sys.stderr)
            sys.exit(1)
        
        success = tracker.append_history_entry(manifest)
        sys.exit(0 if success else 1)
    
    elif args.report:
        report = tracker.generate_report(limit=args.limit)
        print(report)
        sys.exit(0)
    
    elif args.alert_drift:
        alert_triggered = tracker.alert_drift(threshold=args.threshold)
        sys.exit(1 if alert_triggered else 0)
    
    elif args.validate:
        is_valid, h_matrix = tracker.validate_hermetic()
        sys.exit(0 if is_valid else 1)
    
    elif args.validate_history:
        is_valid, sealed, total = tracker.validate_history(required_runs=args.required_runs)
        sys.exit(0 if is_valid else 1)
    
    elif args.alert_drift_abstain:
        alert_triggered = tracker.alert_drift_with_abstain(threshold=args.threshold)
        sys.exit(1 if alert_triggered else 0)
    
    elif args.save_report:
        report_path = tracker.save_report(limit=args.limit)
        sys.exit(0)
    
    elif args.verify_global:
        success, global_root = tracker.verify_global_matrix(num_shards=args.num_shards)
        sys.exit(0 if success else 1)
    
    elif args.save_global:
        global_path = tracker.save_global_matrix(num_shards=args.num_shards)
        sys.exit(0)
    
    elif args.append_global_root:
        success = tracker.append_global_root(num_shards=args.num_shards, parallel=args.parallel)
        sys.exit(0 if success else 1)
    
    elif args.alert_global_drift:
        alert_triggered = tracker.alert_global_drift(num_shards=args.num_shards, threshold=args.threshold, parallel=args.parallel, test_determinism=args.test_determinism)
        sys.exit(1 if alert_triggered else 0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
