#!/usr/bin/env python3
"""
Claude M — The Reflexive Metrologist
Reflexive Metabolism Gate (RFL Gate)
Production Bootstrap with BCa Confidence Intervals
"""

import os
import re
import json
import yaml
import sys
from datetime import datetime
from pathlib import Path
from backend.repro.determinism import deterministic_isoformat
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class BlockMetrics:
    """Ledger block metrics snapshot"""
    timestamp: str
    block_height: int
    statements: int
    proofs_total: int
    proofs_success: int
    success_rate: float
    merkle_root: str

    @property
    def efficiency(self) -> float:
        return self.proofs_success / max(1, self.proofs_total)

    @property
    def density(self) -> float:
        return self.proofs_total / max(1, self.statements)


@dataclass
class BootstrapResult:
    """Single bootstrap experiment result"""
    experiment_id: int
    metric_name: str
    mean: float
    ci_lower: float
    ci_upper: float
    std_dev: float
    coverage_lower: float
    uplift_lower: float
    n_replicates: int
    method: str
    passed: bool


# ============================================================================
# Data Extraction
# ============================================================================

def parse_progress_md(path: str = "docs/progress.md") -> List[BlockMetrics]:
    """Extract time-series ledger metrics from progress.md"""
    with open(path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    blocks = []

    # Pattern 1: Structured block entries
    block_pattern = r'## \[([^\]]+)\] Block (\d+).*?merkle_root: ([a-fx0-9]+).*?statements: (\d+).*?proofs_total: (\d+).*?proofs_success: (\d+)'

    for match in re.finditer(block_pattern, content, re.DOTALL):
        timestamp, height, merkle, stmts, total, success = match.groups()
        total_int = int(total)
        success_int = int(success)
        success_rate = success_int / max(1, total_int)

        blocks.append(BlockMetrics(
            timestamp=timestamp,
            block_height=int(height),
            statements=int(stmts),
            proofs_total=total_int,
            proofs_success=success_int,
            success_rate=success_rate,
            merkle_root=merkle
        ))

    # Pattern 2: Compact log entries
    log_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+BLOCK:\s*(\d+)\s+MERKLE:\s*([a-fx0-9]+)\s+PROOFS:\s*(\d+)/(\d+)\s+STATEMENTS:\s*(\d+)'

    for match in re.finditer(log_pattern, content):
        timestamp, height, merkle, total, success, stmts = match.groups()
        total_int = int(total)
        success_int = int(success)
        success_rate = success_int / max(1, total_int)

        blocks.append(BlockMetrics(
            timestamp=timestamp,
            block_height=int(height),
            statements=int(stmts),
            proofs_total=total_int,
            proofs_success=success_int,
            success_rate=success_rate,
            merkle_root=merkle
        ))

    blocks.sort(key=lambda b: b.block_height)
    return blocks


def fetch_from_db(config: dict) -> Optional[List[BlockMetrics]]:
    """Fetch metrics from PostgreSQL database"""
    try:
        import psycopg2
        db_cfg = config['data_source']['db_config']

        dsn = os.getenv("RFL_DATABASE_URL") or os.getenv("DATABASE_URL")
        if dsn:
            conn = psycopg2.connect(dsn)
        else:
            password = (
                db_cfg.get('password')
                or os.getenv("RFL_DB_PASSWORD")
                or os.getenv("POSTGRES_PASSWORD")
            )
            if not password:
                raise RuntimeError(
                    "Database password not provided. "
                    "Set RFL_DATABASE_URL, DATABASE_URL, RFL_DB_PASSWORD, or POSTGRES_PASSWORD."
                )
            conn = psycopg2.connect(
                host=db_cfg['host'],
                port=db_cfg['port'],
                database=db_cfg['database'],
                user=db_cfg['user'],
                password=password,
            )
        
        cursor = conn.cursor()
        query = """
        SELECT 
            b.created_at,
            b.height,
            COUNT(DISTINCT s.id) as statements,
            COUNT(p.id) as proofs_total,
            COUNT(CASE WHEN p.status = 'success' THEN 1 END) as proofs_success,
            b.root_hash
        FROM blocks b
        LEFT JOIN proofs p ON p.block_id = b.id
        LEFT JOIN statements s ON s.id = p.statement_id
        GROUP BY b.id, b.created_at, b.height, b.root_hash
        ORDER BY b.height
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        blocks = []
        for row in rows:
            timestamp, height, stmts, total, success, merkle = row
            success_rate = success / max(1, total)
            blocks.append(BlockMetrics(
                timestamp=str(timestamp),
                block_height=height,
                statements=stmts,
                proofs_total=total,
                proofs_success=success or 0,
                success_rate=success_rate,
                merkle_root=merkle or "unknown"
            ))
        
        cursor.close()
        conn.close()
        
        return blocks if blocks else None
        
    except Exception as e:
        print(f"⚠ Database connection failed: {e}")
        return None


# ============================================================================
# Bootstrap Methods
# ============================================================================

def bootstrap_resample(data: np.ndarray, n_replicates: int) -> np.ndarray:
    """Standard bootstrap resampling with replacement"""
    n = len(data)
    np.random.seed(42)  # Reproducibility
    resamples = np.random.choice(data, size=(n_replicates, n), replace=True)
    return resamples


def compute_bca_ci(
    data: np.ndarray,
    statistic_func,
    n_replicates: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute BCa (Bias-Corrected and accelerated) confidence interval
    
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    n = len(data)
    
    # Point estimate
    theta_hat = statistic_func(data)
    
    # Bootstrap replicates
    np.random.seed(42)
    boot_samples = np.random.choice(data, size=(n_replicates, n), replace=True)
    boot_stats = np.array([statistic_func(sample) for sample in boot_samples])
    
    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))
    
    # Acceleration (jackknife)
    jack_stats = np.array([
        statistic_func(np.delete(data, i)) for i in range(n)
    ])
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = num / den if den != 0 else 0
    
    # Adjusted percentiles
    alpha = 1 - confidence
    z_alpha = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    p_lower = stats.norm.cdf(z0 + (z0 + z_alpha[0]) / (1 - a * (z0 + z_alpha[0])))
    p_upper = stats.norm.cdf(z0 + (z0 + z_alpha[1]) / (1 - a * (z0 + z_alpha[1])))
    
    # Clip to valid percentile range
    p_lower = np.clip(p_lower, 0, 1)
    p_upper = np.clip(p_upper, 0, 1)
    
    ci_lower = np.percentile(boot_stats, 100 * p_lower)
    ci_upper = np.percentile(boot_stats, 100 * p_upper)
    
    return theta_hat, ci_lower, ci_upper


def run_bootstrap_experiment(
    experiment_id: int,
    metric_name: str,
    data: np.ndarray,
    baseline: float,
    n_replicates: int,
    method: str = "bca"
) -> BootstrapResult:
    """Run single bootstrap experiment with BCa or percentile CI"""

    if method == "bca":
        mean, ci_lower, ci_upper = compute_bca_ci(
            data, np.mean, n_replicates, confidence=0.95
        )
        resamples = bootstrap_resample(data, n_replicates)
        sample_means = np.mean(resamples, axis=1)
        std_dev = np.std(sample_means)
    else:  # percentile
        resamples = bootstrap_resample(data, n_replicates)
        sample_means = np.mean(resamples, axis=1)
        mean = np.mean(sample_means)
        ci_lower = np.percentile(sample_means, 2.5)
        ci_upper = np.percentile(sample_means, 97.5)
        std_dev = np.std(sample_means)

    # Coverage: proportion of bootstrap samples within CI
    in_ci = (sample_means >= ci_lower) & (sample_means <= ci_upper)
    coverage = np.mean(in_ci)

    # Uplift at lower bound relative to baseline
    uplift_lower = (ci_lower - baseline) / max(0.001, abs(baseline)) if baseline != 0 else ci_lower

    # Pass criteria: coverage ≥ 0.92 AND uplift_lower > 1.0
    passed = (coverage >= 0.92) and (uplift_lower > 1.0)

    return BootstrapResult(
        experiment_id=experiment_id,
        metric_name=metric_name,
        mean=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_dev=std_dev,
        coverage_lower=coverage,  # Now using actual coverage proportion
        uplift_lower=uplift_lower,
        n_replicates=n_replicates,
        method=method,
        passed=passed
    )


# ============================================================================
# Visualization
# ============================================================================

def plot_rfl_curves(results: List[BootstrapResult], output_path: str):
    """Generate RFL gate visualization curves"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RFL Gate: Reflexive Metabolism Analysis (BCa Bootstrap)', fontsize=16)
    
    # 1. Coverage distribution
    coverages = [r.coverage_lower for r in results]
    axes[0, 0].hist(coverages, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(0.92, color='red', linestyle='--', linewidth=2, label='Threshold (0.92)')
    axes[0, 0].axvline(np.mean(coverages), color='green', linestyle='-', linewidth=2,
                       label=f'Mean ({np.mean(coverages):.3f})')
    axes[0, 0].set_xlabel('Coverage (proportion in CI)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Bootstrap Coverage Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Uplift Lower Bound distribution
    uplift_lowers = [r.uplift_lower for r in results]
    axes[0, 1].hist(uplift_lowers, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Threshold (1.0)')
    axes[0, 1].axvline(np.mean(uplift_lowers), color='green', linestyle='-', linewidth=2,
                       label=f'Mean ({np.mean(uplift_lowers):.3f})')
    axes[0, 1].set_xlabel('Uplift Lower Bound')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uplift Lower Bound Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Coverage vs Uplift scatter
    colors = ['green' if r.passed else 'red' for r in results]
    axes[1, 0].scatter(coverages, uplift_lowers, c=colors, s=100, alpha=0.6, edgecolors='black')
    axes[1, 0].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0.92, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Coverage')
    axes[1, 0].set_ylabel('Uplift Lower Bound')
    axes[1, 0].set_title('Coverage vs Uplift (Green=PASS, Red=FAIL)')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Pass rate
    pass_count = sum(1 for r in results if r.passed)
    fail_count = len(results) - pass_count
    axes[1, 1].bar(['PASS', 'FAIL'], [pass_count, fail_count],
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Experiment Outcomes ({pass_count}/{len(results)} passed)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")


# ============================================================================
# Main Orchestrator
# ============================================================================

def main(config_path: str = "config/rfl/production.json"):
    """Main RFL gate orchestrator"""
    
    print("=" * 70)
    print("🜍 Claude M — The Reflexive Metrologist")
    print("Reflexive Metabolism Gate (RFL Gate)")
    print("=" * 70)
    print()
    
    # Load configuration
    with open(config_path) as f:
        if config_path.endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    print(f"Configuration: {config['experiment_name']}")
    print(f"Method: {config['bootstrap_config']['method'].upper()}")
    print(f"Experiments: {config['bootstrap_config']['n_experiments']}")
    print(f"Replicates: {config['bootstrap_config']['n_replicates']}")
    print()
    
    # Fetch data
    print("[1/6] Fetching ledger metrics...")
    blocks = None
    
    if config['data_source']['fallback_to_db']:
        print("      Attempting database connection...")
        blocks = fetch_from_db(config)
    
    if blocks is None:
        print("      Falling back to historical data...")
        blocks = parse_progress_md(config['data_source']['path'])
    
    print(f"      ✓ Loaded {len(blocks)} block snapshots")
    
    if len(blocks) < 5:
        print("⚠ Insufficient data for analysis")
        print("VERDICT: ABSTAIN — Infrastructure offline / insufficient data")
        return
    
    print()
    
    # Prepare metrics
    print("[2/6] Preparing metric arrays...")
    success_rates = np.array([b.success_rate for b in blocks if b.proofs_total > 0])
    proof_counts = np.array([b.proofs_success for b in blocks])
    statement_counts = np.array([b.statements for b in blocks])
    efficiencies = np.array([b.efficiency for b in blocks if b.proofs_total > 0])
    densities = np.array([b.density for b in blocks if b.statements > 0])
    
    print(f"      Success rates: {len(success_rates)} samples")
    print(f"      Proof counts: {len(proof_counts)} samples")
    print(f"      Statement counts: {len(statement_counts)} samples")
    print()
    
    # Build experiment schedule
    print("[3/6] Building experiment schedule...")
    experiments = []
    metric_configs = {m['name']: m for m in config['metrics']}
    
    for metric_cfg in config['metrics']:
        name = metric_cfg['name']
        baseline = metric_cfg['baseline']
        n_exp = metric_cfg['experiments']
        
        if name == "success_rate":
            data = success_rates
        elif name == "proof_velocity":
            data = proof_counts
        elif name == "statement_growth":
            data = statement_counts
        elif name == "efficiency":
            data = efficiencies
        elif name == "density":
            data = densities
        else:
            continue
        
        for i in range(n_exp):
            experiments.append((f"{name}_{i+1}", data, baseline))
    
    print(f"      Scheduled {len(experiments)} experiments")
    print()
    
    # Run experiments
    print(f"[4/6] Running {len(experiments)} bootstrap experiments...")
    results = []
    method = config['bootstrap_config']['method']
    n_replicates = config['bootstrap_config']['n_replicates']
    
    for exp_id, (metric_name, data, baseline) in enumerate(experiments, start=1):
        print(f"      [{exp_id:2d}/{len(experiments)}] {metric_name:30s} ", end="", flush=True)
        result = run_bootstrap_experiment(
            exp_id, metric_name, data, baseline, n_replicates, method
        )
        results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status}  (cov_lower={result.coverage_lower:.3f}, uplift_lower={result.uplift_lower:.2f})")
    
    print()
    
    # Generate artifacts
    print("[5/6] Generating artifacts...")
    artifacts_dir = Path(config['output']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    
    # Coverage JSON
    coverage_data = {
        "gate": "RFL - Reflexive Metabolism Gate",
        "metrologist": config['metrologist'],
        "timestamp": deterministic_isoformat("rfl_gate", method, results),
        "method": method.upper(),
        "pass_criteria": config['pass_criteria'],
        "aggregate": {
            "mean_coverage_lower": float(np.mean([r.coverage_lower for r in results])),
            "mean_uplift_lower": float(np.mean([r.uplift_lower for r in results])),
            "experiments_passed": sum(1 for r in results if r.passed),
            "experiments_total": len(results),
            "pass_rate": float(sum(1 for r in results if r.passed) / len(results))
        },
        "experiments": [
            {
                "id": r.experiment_id,
                "metric": r.metric_name,
                "mean": float(r.mean),
                "ci_95_lower": float(r.ci_lower),
                "ci_95_upper": float(r.ci_upper),
                "coverage_lower": float(r.coverage_lower),
                "uplift_lower": float(r.uplift_lower),
                "passed": bool(r.passed)
            }
            for r in results
        ]
    }
    
    coverage_path = artifacts_dir / config['output']['coverage_file']
    with open(coverage_path, 'w') as f:
        json.dump(coverage_data, f, indent=2)
    print(f"      ✓ {coverage_path}")
    
    # Full results JSON
    results_data = {
        **coverage_data,
        "config": config,
        "data_source": {
            "n_blocks": len(blocks),
            "block_height_range": [blocks[0].block_height, blocks[-1].block_height],
            "total_proofs": sum(b.proofs_total for b in blocks),
            "total_success": sum(b.proofs_success for b in blocks)
        }
    }
    
    results_path = artifacts_dir / config['output']['results_file']
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"      ✓ {results_path}")
    
    # Visualization
    viz_path = artifacts_dir / config['output']['visualization_file']
    plot_rfl_curves(results, str(viz_path))
    
    print()
    
    # Verdict
    print("[6/6] Computing verdict...")
    print("=" * 70)
    print("🜍 RFL GATE VERDICT")
    print("=" * 70)
    
    agg = coverage_data['aggregate']
    mean_cov = agg['mean_coverage_lower']
    mean_uplift = agg['mean_uplift_lower']
    pass_rate = agg['pass_rate']
    
    print(f"Mean Coverage Lower:  {mean_cov:.4f}  (threshold: ≥0.92)")
    print(f"Mean Uplift Lower:    {mean_uplift:.4f}  (threshold: >1.0)")
    print(f"Experiments Passed:   {agg['experiments_passed']}/{agg['experiments_total']} ({pass_rate:.1%})")
    print()
    
    if mean_cov >= 0.92 and mean_uplift > 1.0:
        verdict = "PASS"
        msg = config['commit_template']['pass'].format(coverage=mean_cov, uplift=mean_uplift)
        print(f"✓ VERDICT: {verdict}")
        print()
        print(msg)
        print()
        print("Metabolism alive — proofs breathe statistically.")
    else:
        verdict = "FAIL"
        msg = config['commit_template']['fail'].format(coverage=mean_cov, uplift=mean_uplift)
        print(f"✗ VERDICT: {verdict}")
        print()
        print(msg)
        if mean_cov < 0.92:
            print(f"  - Coverage lower bound insufficient: {mean_cov:.4f} < 0.92")
        if mean_uplift <= 1.0:
            print(f"  - Uplift lower bound insufficient: {mean_uplift:.4f} ≤ 1.0")
    
    print("=" * 70)
    
    # Write verdict file
    verdict_data = {
        "verdict": verdict,
        "message": msg,
        "mean_coverage_lower": mean_cov,
        "mean_uplift_lower": mean_uplift,
        "pass_rate": pass_rate
    }
    
    verdict_path = artifacts_dir / "verdict.json"
    with open(verdict_path, 'w') as f:
        json.dump(verdict_data, f, indent=2)
    
    return verdict, verdict_data


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/rfl/production.json"
    main(config_path)
