"""
Metrics Validator - Schema Compliance and Variance Checking

Validates canonical metrics against schema_v1.json and performs statistical
variance analysis with epsilon-tolerance checking.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of metrics validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    variance_check: Optional['VarianceCheck'] = None


@dataclass
class VarianceCheck:
    """Result of variance analysis"""
    coefficient_of_variation: float
    epsilon_tolerance: float
    within_tolerance: bool
    sample_values: List[float]
    mean: float
    stdev: float


class MetricsValidator:
    """Validate metrics against canonical schema and statistical tolerances"""

    def __init__(self, schema_path: Path):
        with open(schema_path) as f:
            self.schema = json.load(f)

    def validate_structure(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate metrics structure against schema"""
        errors = []

        # Check required top-level fields
        required = self.schema.get('required', [])
        for field in required:
            if field not in metrics:
                errors.append(f"Missing required field: {field}")

        # Check source enum
        source = metrics.get('source', '')
        valid_sources = self.schema['properties']['source'].get('enum', [])
        if source and source not in valid_sources:
            errors.append(f"Invalid source: {source}. Must be one of {valid_sources}")

        # Check timestamp format
        timestamp = metrics.get('timestamp', '')
        if timestamp and not self._is_iso8601(timestamp):
            errors.append(f"Invalid timestamp format: {timestamp}. Must be ISO 8601")

        # Check metrics object exists
        if 'metrics' not in metrics:
            errors.append("Missing 'metrics' object")
        else:
            # Validate metrics subsections
            metrics_obj = metrics['metrics']
            self._validate_throughput(metrics_obj.get('throughput', {}), errors)
            self._validate_success_rates(metrics_obj.get('success_rates', {}), errors)
            self._validate_performance(metrics_obj.get('performance', {}), errors)
            self._validate_uplift(metrics_obj.get('uplift', {}), errors)
            self._validate_blockchain(metrics_obj.get('blockchain', {}), errors)
            self._validate_queue(metrics_obj.get('queue', {}), errors)
            if 'trends' in metrics_obj:
                self._validate_trends(metrics_obj.get('trends', {}), errors)
            metadata = metrics_obj.get('metadata', {})
            if metadata and not isinstance(metadata, dict):
                errors.append("metrics.metadata must be an object")

        # Check provenance
        if 'provenance' not in metrics:
            errors.append("Missing 'provenance' object")
        else:
            prov = metrics['provenance']
            if 'collector' not in prov:
                errors.append("Missing provenance.collector")
            if 'merkle_hash' not in prov:
                errors.append("Missing provenance.merkle_hash")
            elif not self._is_sha256(prov['merkle_hash']):
                errors.append(f"Invalid merkle_hash format: {prov['merkle_hash']}")
            if 'history_merkle' in prov and prov['history_merkle'] and not self._is_sha256(prov['history_merkle']):
                errors.append(f"Invalid history_merkle format: {prov['history_merkle']}")
            if 'collectors' in prov and not isinstance(prov['collectors'], list):
                errors.append("provenance.collectors must be a list when present")
            if 'warnings' in prov and not isinstance(prov['warnings'], list):
                errors.append("provenance.warnings must be a list when present")

        return len(errors) == 0, errors

    def _validate_throughput(self, throughput: Dict[str, Any], errors: List[str]):
        """Validate throughput metrics"""
        for key in ['proofs_per_sec', 'proofs_per_hour']:
            if key in throughput:
                value = throughput[key]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"throughput.{key} must be non-negative number")
        for key in ['proof_count_total', 'proof_success_count', 'proof_failure_count']:
            if key in throughput:
                value = throughput[key]
                if not isinstance(value, int) or value < 0:
                    errors.append(f"throughput.{key} must be non-negative integer")

    def _validate_success_rates(self, rates: Dict[str, Any], errors: List[str]):
        """Validate success rate metrics"""
        for key in ['proof_success_rate', 'abstention_rate', 'verification_success_rate']:
            if key in rates:
                value = rates[key]
                if not isinstance(value, (int, float)) or not (0 <= value <= 100):
                    errors.append(f"success_rates.{key} must be between 0 and 100")

    def _validate_performance(self, perf: Dict[str, Any], errors: List[str]):
        """Validate performance metrics"""
        latency_keys = ['mean_latency_ms', 'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms', 'max_latency_ms']
        for key in latency_keys:
            if key in perf:
                value = perf[key]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"performance.{key} must be non-negative number")

        memory_keys = ['mean_memory_mb', 'max_memory_mb']
        for key in memory_keys:
            if key in perf:
                value = perf[key]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"performance.{key} must be non-negative number")
        if 'sample_size' in perf:
            value = perf['sample_size']
            if not isinstance(value, int) or value < 0:
                errors.append("performance.sample_size must be non-negative integer")

    def _validate_uplift(self, uplift: Dict[str, Any], errors: List[str]):
        """Validate uplift metrics"""
        if 'uplift_ratio' in uplift:
            value = uplift['uplift_ratio']
            if not isinstance(value, (int, float)) or value < 0:
                errors.append("uplift.uplift_ratio must be non-negative number")

        if 'p_value' in uplift:
            value = uplift['p_value']
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                errors.append("uplift.p_value must be between 0 and 1")
        if 'delta_from_baseline' in uplift:
            value = uplift['delta_from_baseline']
            if not isinstance(value, (int, float)):
                errors.append("uplift.delta_from_baseline must be numeric")

    def _validate_blockchain(self, blockchain: Dict[str, Any], errors: List[str]):
        """Validate blockchain metrics"""
        for key in ['block_height', 'total_blocks']:
            if key in blockchain:
                value = blockchain[key]
                if not isinstance(value, int) or value < 0:
                    errors.append(f"blockchain.{key} must be non-negative integer")

        if 'merkle_root' in blockchain and blockchain['merkle_root']:
            if not self._is_sha256(blockchain['merkle_root']):
                errors.append(f"blockchain.merkle_root must be 64-character hex string")

    def _validate_queue(self, queue: Dict[str, Any], errors: List[str]):
        """Validate queue metrics"""
        if not isinstance(queue, dict):
            errors.append("metrics.queue must be an object")
            return
        if 'queue_length' in queue and (not isinstance(queue['queue_length'], int) or queue['queue_length'] < 0):
            errors.append("queue.queue_length must be non-negative integer")
        if 'backlog_ratio' in queue and (not isinstance(queue['backlog_ratio'], (int, float)) or queue['backlog_ratio'] < 0):
            errors.append("queue.backlog_ratio must be non-negative number")
        if 'source' in queue and not isinstance(queue['source'], str):
            errors.append("queue.source must be string")

    def _validate_trends(self, trends: Dict[str, Any], errors: List[str]):
        """Validate trends metrics"""
        def _validate_series(name: str, value: Dict[str, Any]):
            if not isinstance(value, dict):
                errors.append(f"trends.{name} must be an object")
                return
            for field in ['latest', 'delta_from_previous', 'moving_average_short', 'moving_average_long']:
                if field in value:
                    if not isinstance(value[field], (int, float)):
                        errors.append(f"trends.{name}.{field} must be numeric")
            if 'samples' in value and (not isinstance(value['samples'], int) or value['samples'] < 0):
                errors.append(f"trends.{name}.samples must be non-negative integer")
            if 'trend' in value and value['trend'] not in ['up', 'down', 'flat']:
                errors.append(f"trends.{name}.trend must be one of ['up','down','flat']")

        if not isinstance(trends, dict):
            errors.append("metrics.trends must be an object")
            return

        for key in ['proofs_per_sec', 'proof_success_rate', 'p95_latency_ms']:
            if key in trends:
                _validate_series(key, trends[key])
        if 'retention' in trends and (not isinstance(trends['retention'], int) or trends['retention'] < 0):
            errors.append("trends.retention must be non-negative integer")

    def _is_iso8601(self, timestamp: str) -> bool:
        """Check if timestamp is valid ISO 8601 format"""
        try:
            from datetime import datetime
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except (ValueError, AttributeError):
            return False

    def _is_sha256(self, hash_str: str) -> bool:
        """Check if string is valid SHA-256 hash"""
        if not isinstance(hash_str, str):
            return False
        if len(hash_str) != 64:
            return False
        try:
            int(hash_str, 16)
            return True
        except ValueError:
            return False

    def check_variance(self, metrics: Dict[str, Any], epsilon: float = 0.01) -> VarianceCheck:
        """
        Perform variance analysis on throughput metrics

        Args:
            metrics: Metrics dictionary
            epsilon: Tolerance threshold (default 1%)

        Returns:
            VarianceCheck with CV and tolerance status
        """
        # Extract throughput values
        sample_values = []
        metrics_obj = metrics.get('metrics', {})

        throughput = metrics_obj.get('throughput', {})
        if 'proofs_per_sec' in throughput:
            sample_values.append(throughput['proofs_per_sec'])

        success_rates = metrics_obj.get('success_rates', {})
        if 'proof_success_rate' in success_rates:
            # Normalize to 0-1 scale for comparison
            sample_values.append(success_rates['proof_success_rate'] / 100.0)

        performance = metrics_obj.get('performance', {})
        if 'mean_latency_ms' in performance:
            sample_values.append(performance['mean_latency_ms'])

        # Need at least 2 values for variance
        if len(sample_values) < 2:
            return VarianceCheck(
                coefficient_of_variation=0.0,
                epsilon_tolerance=epsilon,
                within_tolerance=True,
                sample_values=sample_values,
                mean=sample_values[0] if sample_values else 0.0,
                stdev=0.0
            )

        # Calculate statistics
        mean_val = statistics.mean(sample_values)
        stdev_val = statistics.stdev(sample_values)
        cv = stdev_val / mean_val if mean_val > 0 else 0.0

        return VarianceCheck(
            coefficient_of_variation=cv,
            epsilon_tolerance=epsilon,
            within_tolerance=cv <= epsilon,
            sample_values=sample_values,
            mean=mean_val,
            stdev=stdev_val
        )

    def validate(self, metrics: Dict[str, Any], epsilon: float = 0.01) -> ValidationResult:
        """
        Full validation: structure + variance

        Args:
            metrics: Metrics dictionary to validate
            epsilon: Variance tolerance threshold

        Returns:
            ValidationResult with all checks
        """
        is_valid, errors = self.validate_structure(metrics)
        warnings = []

        # Check variance
        variance_check = self.check_variance(metrics, epsilon)
        if not variance_check.within_tolerance:
            warnings.append(
                f"Variance exceeds tolerance: CV={variance_check.coefficient_of_variation:.4f} > ε={epsilon}"
            )

        # Additional warnings
        metrics_obj = metrics.get('metrics', {})
        success_rates = metrics_obj.get('success_rates', {})
        if 'proof_success_rate' in success_rates:
            if success_rates['proof_success_rate'] < 90.0:
                warnings.append(f"Low proof success rate: {success_rates['proof_success_rate']:.1f}%")

        performance = metrics_obj.get('performance', {})
        if 'regression_detected' in performance and performance['regression_detected']:
            warnings.append("Performance regression detected")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            variance_check=variance_check
        )


def main():
    """CLI entry point for metrics validation"""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    schema_path = project_root / "artifacts" / "metrics" / "schema_v1.json"
    metrics_path = project_root / "artifacts" / "metrics" / "latest.json"

    if not schema_path.exists():
        print(f"Error: Schema not found: {schema_path}")
        return 1

    if not metrics_path.exists():
        print(f"Error: Metrics not found: {metrics_path}")
        return 1

    # Load metrics
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Validate
    validator = MetricsValidator(schema_path)
    result = validator.validate(metrics, epsilon=0.01)

    # Report
    print("=" * 70)
    print("METRICS VALIDATION REPORT")
    print("=" * 70)
    print()

    if result.is_valid:
        print("[PASS] Structure validation: OK")
    else:
        print("[FAIL] Structure validation: ERRORS FOUND")
        for error in result.errors:
            print(f"  ERROR: {error}")
        print()

    if result.warnings:
        print(f"[WARN] {len(result.warnings)} warning(s):")
        for warning in result.warnings:
            print(f"  WARN: {warning}")
        print()

    if result.variance_check:
        vc = result.variance_check
        print("Variance Check:")
        print(f"  Coefficient of Variation: {vc.coefficient_of_variation:.6f}")
        print(f"  Epsilon Tolerance: {vc.epsilon_tolerance:.6f}")
        print(f"  Within Tolerance: {vc.within_tolerance}")
        print(f"  Sample Size: {len(vc.sample_values)}")
        print(f"  Mean: {vc.mean:.4f}")
        print(f"  Stdev: {vc.stdev:.4f}")
        print()

    # Final verdict
    if result.is_valid and (not result.variance_check or result.variance_check.within_tolerance):
        print("[PASS] Metrics validated successfully")
        return 0
    else:
        print("[FAIL] Metrics validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
