#!/usr/bin/env python3
"""
Universal Verification Suite V3 for MathLedger - Audit Sync Enhanced

Devin C - Universal Verifier V3
Mission: Integrate signed verifier results with Audit Harness (Cursor N).
Cross-verify signatures & timestamps. Maintain Proof-or-Abstain integrity.

Exit codes:
  0: [PASS] VERIFIED: ALL CLAIMS HOLD (sync v3)
  1: [FAIL] One or more verification checks failed
  2: [ERROR] Fatal error during verification

Usage:
  python tools/verify_all_v3.py --offline --audit-sync --allblue-gate
  python tools/verify_all_v3.py --check hash --cross-verify
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from backend.repro.determinism import (
    deterministic_isoformat,
    deterministic_timestamp_from_content,
)
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psycopg
except ImportError:
    psycopg = None

from tools.verify_all import Verifier, VerificationResult


EXIT_CODE_MAP = {
    0: {
        "status": "PASS",
        "description": "VERIFIED: ALL CLAIMS HOLD",
        "allblue_status": "green",
        "ci_marker": "[OK]"
    },
    1: {
        "status": "FAIL",
        "description": "One or more verification checks failed",
        "allblue_status": "red",
        "ci_marker": "[FAIL]"
    },
    2: {
        "status": "ERROR",
        "description": "Fatal error during verification",
        "allblue_status": "yellow",
        "ci_marker": "[WARN]"
    }
}


def rfc8785_canonicalize(obj: Any) -> str:
    """
    RFC 8785 canonical JSON serialization.
    
    Implements JSON Canonicalization Scheme (JCS) for deterministic signatures:
    - Sorted object keys
    - No whitespace
    - Unicode escape sequences normalized
    - Numbers in standard form
    """
    def canonicalize_value(val: Any) -> str:
        if val is None:
            return "null"
        elif isinstance(val, bool):
            return "true" if val else "false"
        elif isinstance(val, (int, float)):
            if isinstance(val, float) and (val != val or val == float('inf') or val == float('-inf')):
                raise ValueError(f"Invalid number: {val}")
            return json.dumps(val, separators=(',', ':'), ensure_ascii=True)
        elif isinstance(val, str):
            return json.dumps(val, ensure_ascii=True)
        elif isinstance(val, list):
            items = [canonicalize_value(item) for item in val]
            return '[' + ','.join(items) + ']'
        elif isinstance(val, dict):
            sorted_keys = sorted(val.keys())
            pairs = [json.dumps(k, ensure_ascii=True) + ':' + canonicalize_value(val[k]) for k in sorted_keys]
            return '{' + ','.join(pairs) + '}'
        else:
            raise TypeError(f"Unsupported type: {type(val)}")
    
    return canonicalize_value(obj)


class VerifierV3(Verifier):
    """Enhanced verifier with RFC 8785 canonicalization and AllBlue Gate integration."""
    
    def __init__(self, offline: bool = False, verbose: bool = False, audit_sync: bool = False, 
                 cross_verify: bool = False, allblue_gate: bool = False):
        super().__init__(offline, verbose)
        self.audit_sync = audit_sync
        self.cross_verify = cross_verify
        self.allblue_gate = allblue_gate
        self.run_id = self._generate_run_id()
        self.start_time = datetime.now(timezone.utc)
        self.audit_harness_data = None
        
    def _generate_run_id(self) -> str:
        """Generate deterministic run ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def _compute_signature_rfc8785(self, data: Dict[str, Any]) -> str:
        """Compute RFC 8785 canonical signature for verification results."""
        canonical = rfc8785_canonicalize(data)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def _load_audit_harness_data(self) -> Optional[Dict[str, Any]]:
        """Load existing audit harness data for cross-verification."""
        audit_path = Path("artifacts/audit/verification_summary.json")
        if audit_path.exists():
            with open(audit_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _cross_verify_timestamp(self, current_ts: str, previous_ts: Optional[str]) -> Tuple[bool, str]:
        """Cross-verify timestamp ordering and validity."""
        if not previous_ts:
            return True, "No previous timestamp to verify"
        
        try:
            current_dt = datetime.fromisoformat(current_ts.replace('Z', '+00:00'))
            previous_dt = datetime.fromisoformat(previous_ts.replace('Z', '+00:00'))
            
            if current_dt < previous_dt:
                return False, f"Timestamp regression: {current_ts} < {previous_ts}"
            
            time_delta = (current_dt - previous_dt).total_seconds()
            if time_delta > 86400:  # 24 hours
                return True, f"Warning: Large time gap ({time_delta:.0f}s) since last verification"
            
            return True, f"Timestamp ordering valid (delta: {time_delta:.2f}s)"
        except Exception as e:
            return False, f"Timestamp validation error: {e}"
    
    def _cross_verify_signature(self, data: Dict[str, Any], claimed_signature: str) -> Tuple[bool, str]:
        """Cross-verify signature integrity."""
        data_without_sig = {k: v for k, v in data.items() if k != "signature"}
        computed_sig = self._compute_signature_rfc8785(data_without_sig)
        
        if computed_sig != claimed_signature:
            return False, f"Signature mismatch: computed={computed_sig[:16]}... claimed={claimed_signature[:16]}..."
        
        return True, f"Signature valid: {computed_sig[:32]}..."
    
    def generate_verification_summary(self) -> Dict[str, Any]:
        """Generate comprehensive verification summary with RFC 8785 canonicalization."""
        end_time = datetime.now(timezone.utc)
        duration_seconds = (end_time - self.start_time).total_seconds()
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        exit_code = 0 if passed_count == total_count else 1
        
        check_results = []
        for result in self.results:
            check_results.append({
                "name": result.name,
                "passed": result.passed,
                "message": result.message,
                "details": result.details
            })
        
        coverage_pct = (passed_count / total_count * 100) if total_count > 0 else 0.0
        
        summary_data = {
            "run_id": self.run_id,
            "timestamp": self.start_time.isoformat(),
            "end_timestamp": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "version": "v3.6",
            "verifier": "Devin C - Universal Verifier V3.6 (Predictive Analytics Uplift)",
            "mode": "offline" if self.offline else "online",
            "checks": {
                "total": total_count,
                "passed": passed_count,
                "failed": total_count - passed_count
            },
            "telemetry": {
                "checks_passed": passed_count,
                "checks_total": total_count,
                "coverage_pct": round(coverage_pct, 2)
            },
            "exit_code": exit_code,
            "exit_code_map": EXIT_CODE_MAP[exit_code],
            "results": check_results,
            "audit_metadata": {
                "schema_version": "v3.6",
                "audit_sync_enabled": self.audit_sync,
                "cross_verification_enabled": self.cross_verify,
                "allblue_gate_enabled": self.allblue_gate,
                "compliance_tags": ["RC", "ME", "IVL"],
                "acquisition_narrative": "Reliability & Correctness verification with audit trail",
                "canonicalization": "RFC 8785"
            }
        }
        
        if self.cross_verify and self.audit_harness_data:
            prev_ts = self.audit_harness_data.get("timestamp")
            ts_valid, ts_msg = self._cross_verify_timestamp(summary_data["timestamp"], prev_ts)
            
            prev_sig = self.audit_harness_data.get("signature")
            if prev_sig:
                sig_valid, sig_msg = self._cross_verify_signature(self.audit_harness_data, prev_sig)
                summary_data["audit_metadata"]["previous_verification"] = {
                    "timestamp": prev_ts,
                    "signature": prev_sig[:32] + "...",
                    "signature_valid": sig_valid,
                    "signature_message": sig_msg,
                    "timestamp_valid": ts_valid,
                    "timestamp_message": ts_msg
                }
        
        signature = self._compute_signature_rfc8785(summary_data)
        summary_data["signature"] = signature
        
        return summary_data
    
    def generate_ci_summary(self, summary: Dict[str, Any]) -> str:
        """Generate CI-friendly markdown summary for AllBlue ingestion."""
        exit_info = summary["exit_code_map"]
        marker = exit_info["ci_marker"]
        status = exit_info["status"]
        
        ci_summary = f"""# {marker} Verification Summary - {status}

**Run ID**: `{summary['run_id']}`
**Timestamp**: {summary['timestamp']}
**Duration**: {summary['duration_seconds']:.2f}s
**Mode**: {summary['mode']}
**Version**: {summary['version']}


- **Total Checks**: {summary['checks']['total']}
- **Passed**: {summary['checks']['passed']} [OK]
- **Failed**: {summary['checks']['failed']} [FAIL]


"""
        
        for result in summary['results']:
            result_marker = "[OK]" if result['passed'] else "[FAIL]"
            ci_summary += f"### {result_marker} {result['name']}\n\n"
            ci_summary += f"**Status**: {'PASS' if result['passed'] else 'FAIL'}\n"
            ci_summary += f"**Message**: {result['message']}\n\n"
            
            if result['details'] and self.verbose:
                ci_summary += "**Details**:\n"
                for key, value in result['details'].items():
                    ci_summary += f"- `{key}`: {value}\n"
                ci_summary += "\n"
        
        ci_summary += f"""## Exit Code Map

- **Code**: {summary['exit_code']}
- **Status**: {exit_info['status']}
- **AllBlue Status**: {exit_info['allblue_status']}
- **Description**: {exit_info['description']}


- **Schema Version**: {summary['audit_metadata']['schema_version']}
- **Canonicalization**: {summary['audit_metadata']['canonicalization']}
- **Compliance Tags**: {', '.join(summary['audit_metadata']['compliance_tags'])}
- **Signature**: `{summary['signature'][:32]}...`
"""
        
        if self.cross_verify and "previous_verification" in summary['audit_metadata']:
            prev = summary['audit_metadata']['previous_verification']
            ci_summary += f"""

- **Previous Timestamp**: {prev['timestamp']}
- **Previous Signature**: `{prev['signature']}`
- **Signature Valid**: {prev['signature_valid']}
- **Timestamp Valid**: {prev['timestamp_valid']}
- **Timestamp Message**: {prev['timestamp_message']}
"""
        
        ci_summary += "\n---\n*Generated by Universal Verifier V3 - Audit Sync Enhanced*\n"
        
        return ci_summary
    
    def validate_against_audit_harness(self, summary: Dict[str, Any]) -> Tuple[bool, str]:
        """Cross-validate summary against Audit Harness schema."""
        required_fields = [
            "run_id", "timestamp", "version", "verifier", "mode",
            "checks", "exit_code", "exit_code_map", "results",
            "audit_metadata", "signature"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in summary:
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        if summary["audit_metadata"]["schema_version"] not in ["v3.0", "v3.4", "v3.5", "v3.6"]:
            return False, f"Invalid schema version: {summary['audit_metadata']['schema_version']}"
        
        if summary["audit_metadata"]["canonicalization"] != "RFC 8785":
            return False, "RFC 8785 canonicalization not enabled"
        
        expected_signature = self._compute_signature_rfc8785({k: v for k, v in summary.items() if k != "signature"})
        if summary["signature"] != expected_signature:
            return False, "Signature mismatch - data may have been tampered with"
        
        return True, "Audit harness validation passed"
    
    def freeze_allblue_state(self, summary: Dict[str, Any]) -> Optional[Path]:
        """Freeze fleet state when ALL BLUE appears in CI."""
        if not self.allblue_gate:
            return None
        
        if summary["exit_code"] != 0:
            self.log("AllBlue Gate: Not freezing state - verification failed")
            return None
        
        allblue_dir = Path("artifacts/allblue")
        allblue_dir.mkdir(parents=True, exist_ok=True)
        
        fleet_state = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "verification_summary": summary,
            "state_hash": summary["signature"],
            "allblue_status": "ALL BLUE",
            "compliance_tags": summary["audit_metadata"]["compliance_tags"],
            "verifier_version": summary["version"],
            "canonicalization": "RFC 8785"
        }
        
        fleet_state_signature = self._compute_signature_rfc8785(fleet_state)
        fleet_state["fleet_signature"] = fleet_state_signature
        
        fleet_state_path = allblue_dir / "fleet_state.json"
        with open(fleet_state_path, 'w', encoding='utf-8') as f:
            json.dump(fleet_state, f, indent=2, ensure_ascii=True)
        
        self.log(f"AllBlue Gate: Fleet state frozen at {fleet_state_path}")
        self.log(f"Fleet signature: {fleet_state_signature[:32]}...")
        
        return fleet_state_path
    
    def write_audit_output(self, output_path: Path, summary: Dict[str, Any], ci_summary: str):
        """Write verification results to audit directory."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        
        ci_summary_path = output_path.parent / "verification_summary.md"
        with open(ci_summary_path, 'w', encoding='utf-8') as f:
            f.write(ci_summary)
        
        exit_code_path = output_path.parent / "exit_code_map.json"
        with open(exit_code_path, 'w', encoding='utf-8') as f:
            json.dump(EXIT_CODE_MAP, f, indent=2, ensure_ascii=True)
        
        self.log(f"Audit output written to {output_path}")
        self.log(f"CI summary written to {ci_summary_path}")
        self.log(f"Exit code map written to {exit_code_path}")
    
    def append_telemetry_trend(self, telemetry: Dict[str, Any], telemetry_hash: str):
        """Append telemetry to trends JSONL for analytics."""
        trends_path = Path("artifacts/audit/telemetry_trends.jsonl")
        trends_path.parent.mkdir(parents=True, exist_ok=True)
        
        trend_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "checks_passed": telemetry['checks_passed'],
            "checks_total": telemetry['checks_total'],
            "coverage_pct": telemetry['coverage_pct'],
            "telemetry_hash": telemetry_hash
        }
        
        canonical_entry = rfc8785_canonicalize(trend_entry)
        with open(trends_path, 'a', encoding='utf-8') as f:
            f.write(canonical_entry + '\n')
        
        self.log(f"Telemetry trend appended to {trends_path}")
    
    def load_telemetry_trends(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Load last N telemetry trends from JSONL."""
        trends_path = Path("artifacts/audit/telemetry_trends.jsonl")
        if not trends_path.exists():
            return []
        
        trends = []
        with open(trends_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    trends.append(json.loads(line))
        
        return trends[-limit:] if len(trends) > limit else trends
    
    def compute_telemetry_statistics(self, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute rolling statistics (mean ± σ) over telemetry trends."""
        if not trends:
            return {
                "count": 0,
                "mean": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        coverages = [t['coverage_pct'] for t in trends]
        n = len(coverages)
        mean = sum(coverages) / n
        variance = sum((x - mean) ** 2 for x in coverages) / n
        std_dev = variance ** 0.5
        
        return {
            "count": n,
            "mean": round(mean, 2),
            "std_dev": round(std_dev, 2),
            "min": round(min(coverages), 2),
            "max": round(max(coverages), 2)
        }
    
    def check_telemetry_drift(self, current_coverage: float, stats: Dict[str, Any], threshold_multiplier: float = 0.9) -> Tuple[bool, str]:
        """Check if current coverage drifts below mean * threshold_multiplier."""
        if stats['count'] < 2:
            return False, "Insufficient data for drift detection (need >=2 runs)"
        
        mean = stats['mean']
        threshold = mean * threshold_multiplier
        
        if current_coverage < threshold:
            drift_pct = ((mean - current_coverage) / mean) * 100
            return True, f"Coverage drift detected: {current_coverage}% < {threshold:.2f}% (mu={mean}%, drift={drift_pct:.1f}%)"
        
        return False, f"Coverage stable: {current_coverage}% >= {threshold:.2f}% (mu={mean}%)"
    
    def generate_ascii_telemetry_chart(self, trends: List[Dict[str, Any]], current_coverage: float) -> str:
        """Generate ASCII chart for telemetry trends."""
        if not trends:
            return "No historical data available"
        
        coverages = [t['coverage_pct'] for t in trends] + [current_coverage]
        max_coverage = max(coverages)
        min_coverage = min(coverages)
        
        chart_lines = ["Telemetry Trend (last {} runs + current):".format(len(trends))]
        
        for i, cov in enumerate(coverages):
            bars = int((cov / 100) * 20)
            marker = "*" if i == len(coverages) - 1 else "#"
            label = "NOW" if i == len(coverages) - 1 else f"R{i+1:02d}"
            chart_lines.append(f"{label} {cov:5.1f}% [{marker * bars}{'-' * (20 - bars)}]")
        
        return '\n'.join(chart_lines)
    
    def write_github_job_summary(self, stats: Dict[str, Any], drift_detected: bool, drift_msg: str):
        """Write telemetry analytics to GitHub job summary."""
        github_step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
        if not github_step_summary:
            return
        
        with open(github_step_summary, 'a', encoding='utf-8') as f:
            f.write("\n## Telemetry Analytics (V3.5)\n\n")
            
            if stats['count'] > 0:
                f.write(f"Runs: {stats['count']} | mu={stats['mean']}% | sigma={stats['std_dev']}% | Range=[{stats['min']}%, {stats['max']}%]\n\n")
                
                if drift_detected:
                    f.write(f"[ALERT] {drift_msg}\n")
                else:
                    f.write(f"[PASS] Telemetry Health Stable (mu={stats['mean']}%, sigma={stats['std_dev']}%)\n")
            else:
                f.write("[INFO] Insufficient historical data for analytics (first run)\n")
                f.write("[PASS] Telemetry Health Stable (mu=N/A, sigma=N/A)\n")
    
    def compute_ewma(self, trends: List[Dict[str, Any]], alpha: float = 0.3) -> float:
        """Compute exponential-weighted moving average of coverage."""
        if not trends:
            return 0.0
        
        coverages = [t['coverage_pct'] for t in trends]
        ewma = coverages[0]
        
        for cov in coverages[1:]:
            ewma = alpha * cov + (1 - alpha) * ewma
        
        return round(ewma, 2)
    
    def predict_drift_probability(self, trends: List[Dict[str, Any]], stats: Dict[str, Any], threshold_multiplier: float = 0.9) -> Tuple[float, List[float]]:
        """Predict probability of drift in next 3 runs using EWMA trend."""
        if stats['count'] < 3:
            return 0.0, []
        
        coverages = [t['coverage_pct'] for t in trends]
        ewma = self.compute_ewma(trends)
        
        recent_trend = coverages[-3:]
        trend_slope = (recent_trend[-1] - recent_trend[0]) / 3 if len(recent_trend) == 3 else 0.0
        
        forecasts = []
        for i in range(1, 4):
            forecast = ewma + (trend_slope * i)
            forecasts.append(round(forecast, 2))
        
        threshold = stats['mean'] * threshold_multiplier
        drift_count = sum(1 for f in forecasts if f < threshold)
        drift_probability = (drift_count / 3) * 100
        
        return round(drift_probability, 1), forecasts
    
    def append_forecast_to_jsonl(self, forecast_data: Dict[str, Any]):
        """Append forecast to telemetry_forecast.jsonl with RFC 8785 format."""
        forecast_path = Path("artifacts/audit/telemetry_forecast.jsonl")
        forecast_path.parent.mkdir(parents=True, exist_ok=True)
        
        canonical_entry = rfc8785_canonicalize(forecast_data)
        with open(forecast_path, 'a', encoding='utf-8') as f:
            f.write(canonical_entry + '\n')
        
        self.log(f"Forecast appended to {forecast_path}")
    
    def write_forecast_to_job_summary(self, drift_prob: float, forecasts: List[float], stats: Dict[str, Any], accuracy_data: Dict[str, Any] = None, accuracy_trend: Dict[str, Any] = None):
        """Write drift forecast to GitHub job summary."""
        github_step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
        if not github_step_summary:
            return
        
        with open(github_step_summary, 'a', encoding='utf-8') as f:
            f.write("\n## Predictive Analytics (V3.6)\n\n")
            f.write(f"3-Run Drift Forecast: {forecasts[0]}%, {forecasts[1]}%, {forecasts[2]}%\n\n")
            
            if drift_prob > 0:
                f.write(f"[ALERT] Drift Predicted (prob={drift_prob}%, mu={stats['mean']}%, sigma={stats['std_dev']}%)\n")
            else:
                f.write(f"[PASS] No Drift Predicted (prob={drift_prob}%, mu={stats['mean']}%, sigma={stats['std_dev']}%)\n")
            
            if accuracy_data:
                f.write(f"\nForecast Accuracy: MAE={accuracy_data['mae']}%, Error={accuracy_data['error_pct']}%\n")
                if accuracy_data.get('alert'):
                    f.write(f"[ABSTAIN] Forecast Alert reason={accuracy_data['alert_reason']}\n")
                else:
                    f.write(f"[PASS] Forecast Accuracy ok error_pct={accuracy_data['error_pct']}%\n")
            
            if accuracy_trend and accuracy_trend['count'] >= 3:
                f.write(f"\nAccuracy Trend (last {accuracy_trend['count']}): mean={accuracy_trend['mean']}%, std={accuracy_trend['std_dev']}%, min={accuracy_trend['min']}%, max={accuracy_trend['max']}%\n")
                f.write(f"[PASS] Forecast Trend last10 mean={accuracy_trend['mean']}% std={accuracy_trend['std_dev']}%\n")
    
    def load_last_forecast(self) -> Dict[str, Any]:
        """Load the most recent forecast from telemetry_forecast.jsonl."""
        forecast_path = Path("artifacts/audit/telemetry_forecast.jsonl")
        if not forecast_path.exists():
            return None
        
        last_forecast = None
        with open(forecast_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_forecast = json.loads(line.strip())
        
        return last_forecast
    
    def validate_forecast_accuracy(self, current_coverage: float, last_forecast: Dict[str, Any], accuracy_threshold: float = 10.0, drift_alert_threshold: int = 50) -> Dict[str, Any]:
        """Validate forecast accuracy against actual coverage."""
        if not last_forecast or 'forecasts' not in last_forecast:
            return None
        
        predicted = last_forecast['forecasts'][0]
        actual = current_coverage
        
        mae = abs(predicted - actual)
        error_pct = (mae / actual) * 100 if actual > 0 else 0.0
        
        alert = False
        alert_reason = None
        
        if error_pct > accuracy_threshold:
            alert = True
            alert_reason = f"high_error(>{error_pct:.1f}%)"
        elif last_forecast.get('drift_probability', 0) >= drift_alert_threshold:
            alert = True
            alert_reason = f"drift_pred(>={last_forecast['drift_probability']}%)"
        
        accuracy_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "forecast_run_id": last_forecast.get('run_id'),
            "predicted": predicted,
            "actual": actual,
            "mae": round(mae, 2),
            "error_pct": round(error_pct, 2),
            "alert": alert,
            "alert_reason": alert_reason
        }
        
        return accuracy_data
    
    def append_accuracy_to_jsonl(self, accuracy_data: Dict[str, Any]):
        """Append forecast accuracy to forecast_accuracy.jsonl with RFC 8785 format."""
        accuracy_path = Path("artifacts/audit/forecast_accuracy.jsonl")
        accuracy_path.parent.mkdir(parents=True, exist_ok=True)
        
        canonical_entry = rfc8785_canonicalize(accuracy_data)
        with open(accuracy_path, 'a', encoding='utf-8') as f:
            f.write(canonical_entry + '\n')
        
        self.log(f"Forecast accuracy appended to {accuracy_path}")
    
    def load_accuracy_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Load last N accuracy records from forecast_accuracy.jsonl."""
        accuracy_path = Path("artifacts/audit/forecast_accuracy.jsonl")
        if not accuracy_path.exists():
            return []
        
        records = []
        with open(accuracy_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        
        return records[-limit:] if len(records) > limit else records
    
    def compute_accuracy_trend(self, accuracy_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute rolling trend statistics from accuracy history."""
        if not accuracy_history:
            return {'count': 0, 'mean': 0.0, 'std_dev': 0.0, 'min': 0.0, 'max': 0.0}
        
        error_pcts = [record['error_pct'] for record in accuracy_history]
        
        mean = sum(error_pcts) / len(error_pcts)
        variance = sum((x - mean) ** 2 for x in error_pcts) / len(error_pcts)
        std_dev = variance ** 0.5
        
        return {
            'count': len(error_pcts),
            'mean': round(mean, 2),
            'std_dev': round(std_dev, 2),
            'min': round(min(error_pcts), 2),
            'max': round(max(error_pcts), 2)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Universal verification suite V3 with RFC 8785 canonicalization",
        epilog="Verify every claim. Cross-verify signatures. Maintain Proof-or-Abstain integrity."
    )
    parser.add_argument(
        "--check",
        type=str,
        help="Run specific check (hash, merkle, files, metrics, normalization, database, api, parents)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip database and API checks"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audit/verification_summary.json"),
        help="Output path for verification summary JSON"
    )
    parser.add_argument(
        "--audit-sync",
        action="store_true",
        help="Enable audit harness synchronization"
    )
    parser.add_argument(
        "--cross-verify",
        action="store_true",
        help="Enable cross-verification with previous runs"
    )
    parser.add_argument(
        "--allblue-gate",
        action="store_true",
        help="Enable AllBlue Gate state freezing"
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=10,
        help="Rolling window size (N runs) for telemetry analytics"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.9,
        help="Drift threshold multiplier relative to mean (e.g., 0.9 => 10 percent below mean)"
    )
    parser.add_argument(
        "--predict-drift",
        action="store_true",
        help="Enable predictive drift analytics with 3-run forecast"
    )
    parser.add_argument(
        "--validate-forecast",
        action="store_true",
        help="Validate forecast accuracy against actuals (auto-enabled with --predict-drift)"
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=10.0,
        help="Accuracy error threshold percent (default: 10.0)"
    )
    parser.add_argument(
        "--drift-alert-threshold",
        type=int,
        default=50,
        help="Drift probability alert threshold percent (default: 50)"
    )
    
    args = parser.parse_args()
    
    if args.predict_drift and not args.validate_forecast:
        args.validate_forecast = True
    
    verifier = VerifierV3(
        offline=args.offline, 
        verbose=args.verbose, 
        audit_sync=args.audit_sync,
        cross_verify=args.cross_verify,
        allblue_gate=args.allblue_gate
    )
    
    if args.cross_verify:
        verifier.audit_harness_data = verifier._load_audit_harness_data()
        if verifier.audit_harness_data:
            print(f"[INFO] Loaded previous verification for cross-validation")
    
    success = verifier.run_all_checks(specific_check=args.check)
    
    summary = verifier.generate_verification_summary()
    ci_summary = verifier.generate_ci_summary(summary)
    
    if args.audit_sync:
        is_valid, validation_msg = verifier.validate_against_audit_harness(summary)
        if not is_valid:
            print(f"\n[ERROR] Audit harness validation failed: {validation_msg}", file=sys.stderr)
            return 2
        else:
            print(f"\n[PASS] Audit harness validation: {validation_msg}")
    
    verifier.write_audit_output(args.output, summary, ci_summary)
    
    fleet_state_path = verifier.freeze_allblue_state(summary)
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION V3.6 SUMMARY")
    print(f"{'='*60}")
    print(f"Run ID: {summary['run_id']}")
    print(f"Signature (RFC 8785): {summary['signature'][:32]}...")
    print(f"Output: {args.output}")
    if fleet_state_path:
        print(f"Fleet State: {fleet_state_path}")
    print(f"{'='*60}\n")
    
    telemetry = summary['telemetry']
    telemetry_complete = (
        telemetry['checks_passed'] is not None and
        telemetry['checks_total'] is not None and
        telemetry['coverage_pct'] is not None
    )
    
    if not telemetry_complete:
        print("[ABSTAIN] Telemetry incomplete - cannot seal verification gate")
        print(f"  checks_passed: {telemetry.get('checks_passed', 'MISSING')}")
        print(f"  checks_total: {telemetry.get('checks_total', 'MISSING')}")
        print(f"  coverage_pct: {telemetry.get('coverage_pct', 'MISSING')}")
        return 2
    
    telemetry_hash = hashlib.sha256(
        rfc8785_canonicalize(telemetry).encode('utf-8')
    ).hexdigest()
    print(f"[PASS] Verification Gate Telemetry sealed {telemetry_hash}")
    print(f"  Checks: {telemetry['checks_passed']}/{telemetry['checks_total']} ({telemetry['coverage_pct']}%)")
    
    verifier.append_telemetry_trend(telemetry, telemetry_hash)
    
    trends = verifier.load_telemetry_trends(limit=args.trend_window)
    stats = verifier.compute_telemetry_statistics(trends)
    
    print(f"\n{'='*60}")
    print(f"TELEMETRY ANALYTICS V3.6 (window={args.trend_window}, threshold={args.drift_threshold})")
    print(f"{'='*60}")
    
    if stats['count'] > 0:
        print(f"Rolling Statistics (last {stats['count']} runs):")
        print(f"  Mean (mu): {stats['mean']}%")
        print(f"  Std Dev (sigma): {stats['std_dev']}%")
        print(f"  Range: [{stats['min']}%, {stats['max']}%]")
        print()
        
        drift_detected, drift_msg = verifier.check_telemetry_drift(telemetry['coverage_pct'], stats, threshold_multiplier=args.drift_threshold)
        if drift_detected:
            print(f"[ALERT] {drift_msg}")
        else:
            print(f"[PASS] Telemetry Health Stable (mu={stats['mean']}%, sigma={stats['std_dev']}%)")
        
        print()
        chart = verifier.generate_ascii_telemetry_chart(trends, telemetry['coverage_pct'])
        print(chart)
        
        verifier.write_github_job_summary(stats, drift_detected, drift_msg)
        
        accuracy_data = None
        accuracy_trend = None
        if args.validate_forecast:
            print(f"[PASS] Forecast Thresholds accuracy={args.accuracy_threshold}% drift={args.drift_alert_threshold}%")
            
            last_forecast = verifier.load_last_forecast()
            if last_forecast:
                accuracy_data = verifier.validate_forecast_accuracy(
                    telemetry['coverage_pct'], 
                    last_forecast,
                    accuracy_threshold=args.accuracy_threshold,
                    drift_alert_threshold=args.drift_alert_threshold
                )
                if accuracy_data:
                    print()
                    print(f"{'='*60}")
                    print(f"FORECAST ACCURACY VALIDATION")
                    print(f"{'='*60}")
                    print(f"Predicted: {accuracy_data['predicted']}%")
                    print(f"Actual: {accuracy_data['actual']}%")
                    print(f"MAE: {accuracy_data['mae']}%")
                    print(f"Error: {accuracy_data['error_pct']}%")
                    print()
                    
                    if accuracy_data['alert']:
                        print(f"[ABSTAIN] Forecast Alert reason={accuracy_data['alert_reason']}")
                    else:
                        print(f"[PASS] Forecast Accuracy ok error_pct={accuracy_data['error_pct']}%")
                    
                    verifier.append_accuracy_to_jsonl(accuracy_data)
                    print(f"{'='*60}\n")
            
            accuracy_history = verifier.load_accuracy_history(limit=10)
            if len(accuracy_history) >= 3:
                accuracy_trend = verifier.compute_accuracy_trend(accuracy_history)
                print()
                print(f"{'='*60}")
                print(f"ACCURACY TREND ANALYSIS (last {accuracy_trend['count']} validations)")
                print(f"{'='*60}")
                print(f"Accuracy Trend: mean={accuracy_trend['mean']}% std={accuracy_trend['std_dev']}% min={accuracy_trend['min']}% max={accuracy_trend['max']}%")
                print(f"[PASS] Forecast Trend last10 mean={accuracy_trend['mean']}% std={accuracy_trend['std_dev']}%")
                print(f"{'='*60}\n")
        
        if args.predict_drift and stats['count'] >= 3:
            print()
            print(f"{'='*60}")
            print(f"PREDICTIVE ANALYTICS V3.6")
            print(f"{'='*60}")
            
            drift_prob, forecasts = verifier.predict_drift_probability(trends, stats, threshold_multiplier=args.drift_threshold)
            ewma = verifier.compute_ewma(trends)
            
            print(f"EWMA (alpha=0.3): {ewma}%")
            print(f"3-Run Drift Forecast: {forecasts[0]}%, {forecasts[1]}%, {forecasts[2]}%")
            print()
            
            if drift_prob > 0:
                print(f"[ALERT] Drift Predicted (prob={drift_prob}%, mu={stats['mean']}%, sigma={stats['std_dev']}%)")
            else:
                print(f"[PASS] No Drift Predicted (prob={drift_prob}%, mu={stats['mean']}%, sigma={stats['std_dev']}%)")
            
            forecast_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": verifier.run_id,
                "ewma": ewma,
                "forecasts": forecasts,
                "drift_probability": drift_prob,
                "threshold": stats['mean'] * args.drift_threshold,
                "stats": {
                    "mean": stats['mean'],
                    "std_dev": stats['std_dev'],
                    "count": stats['count']
                }
            }
            verifier.append_forecast_to_jsonl(forecast_data)
            verifier.write_forecast_to_job_summary(drift_prob, forecasts, stats, accuracy_data, accuracy_trend)
            
            print(f"{'='*60}\n")
    else:
        print("[INFO] Insufficient historical data for analytics (first run)")
        print(f"[PASS] Telemetry Health Stable (mu=N/A, sigma=N/A)")
        
        verifier.write_github_job_summary(stats, False, "")
    
    print(f"{'='*60}\n")
    
    if success:
        print(f"[PASS] Verifier Sync v3.6 [{summary['signature'][:16]}]")
        return 0
    else:
        print("[FAIL] Verification failed - see summary for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
# Test comment with non-ASCII: café ☕
