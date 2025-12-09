# Task C2: Calibration Runner Implementation Plan

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: Implementation Ready  
**Target**: `backend/verification/calibrate_noise.py`

---

## 1. Overview

The **Calibration Runner** orchestrates the complete noise model calibration pipeline: telemetry collection, ground truth labeling, statistical fitting, cross-tier validation, and model export.

**CLI Interface**: `calibrate_noise --tiers FAST BALANCED SLOW --n 10000 --export calibration.yaml`

**Key Features**:
- Parallel verification with worker pool
- Execution DAG for labeling pipeline stages
- Timeout escalation for ground truth labeling
- Validation suite with hypothesis testing
- Canonicalization for telemetry snapshotting

---

## 2. CLI Interface Design

### 2.1 Command Signature

```bash
calibrate_noise \
    --tiers FAST BALANCED SLOW \
    --n 10000 \
    --corpus mathlib \
    --export calibration.yaml \
    --workers 8 \
    --seed 12345 \
    --validation-split 0.2 \
    --label-sample 1000 \
    --timeout-multiplier 10 \
    --output-dir ./calibration_output \
    --verbose
```

### 2.2 Arguments Specification

```python
import argparse
from pathlib import Path
from typing import List

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for calibration runner."""
    
    parser = argparse.ArgumentParser(
        description="Calibrate noise models for verifier imperfection modeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--tiers",
        nargs="+",
        required=True,
        choices=["FAST", "BALANCED", "SLOW"],
        help="Verifier tiers to calibrate (space-separated)",
    )
    
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Total number of verifications to run per tier",
    )
    
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="Path to export calibrated noise models (YAML)",
    )
    
    # Corpus selection
    parser.add_argument(
        "--corpus",
        type=str,
        default="mathlib",
        choices=["mathlib", "curriculum", "synthetic", "custom"],
        help="Proof corpus to use for calibration",
    )
    
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Path to custom corpus (required if --corpus=custom)",
    )
    
    parser.add_argument(
        "--corpus-filter",
        type=str,
        default=None,
        help="Regex filter for corpus modules (e.g., 'Mathlib.Algebra.*')",
    )
    
    # Parallelization
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes",
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Master seed for reproducibility",
    )
    
    # Validation
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (0.0-1.0)",
    )
    
    # Ground truth labeling
    parser.add_argument(
        "--label-sample",
        type=int,
        default=1000,
        help="Number of proofs to label with ground truth",
    )
    
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=10.0,
        help="Timeout multiplier for extended timeout labeling",
    )
    
    parser.add_argument(
        "--consensus-verifiers",
        nargs="+",
        default=["lean"],
        help="Verifiers to use for consensus labeling",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./calibration_output"),
        help="Directory for calibration outputs (telemetry, models, reports)",
    )
    
    parser.add_argument(
        "--telemetry-log",
        type=Path,
        default=None,
        help="Path to telemetry log file (default: <output_dir>/telemetry.jsonl)",
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.corpus == "custom" and args.corpus_path is None:
        parser.error("--corpus-path is required when --corpus=custom")
    
    if not (0.0 <= args.validation_split <= 1.0):
        parser.error("--validation-split must be in range [0.0, 1.0]")
    
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    
    # Set defaults
    if args.telemetry_log is None:
        args.telemetry_log = args.output_dir / "telemetry.jsonl"
    
    return args
```

---

## 3. Main Calibration Pipeline

### 3.1 Pipeline Stages

```
Stage 1: Corpus Selection
    ↓
Stage 2: Telemetry Collection (parallel)
    ↓
Stage 3: Ground Truth Labeling (parallel)
    ↓
Stage 4: Statistical Fitting
    ↓
Stage 5: Cross-Tier Validation
    ↓
Stage 6: Model Export
```

### 3.2 Main Function

```python
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
import yaml

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for calibration runner."""
    
    # Parse arguments
    args = parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Noise Model Calibration Runner")
    logger.info("=" * 80)
    logger.info(f"Tiers: {args.tiers}")
    logger.info(f"Verifications per tier: {args.n}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Stage 1: Corpus Selection
    logger.info("Stage 1: Corpus Selection")
    corpus = select_corpus(args)
    logger.info(f"Selected {len(corpus)} modules from corpus")
    
    # Stage 2: Telemetry Collection
    logger.info("Stage 2: Telemetry Collection")
    telemetry_records = collect_telemetry(args, corpus)
    logger.info(f"Collected {len(telemetry_records)} telemetry records")
    
    # Save telemetry to log
    save_telemetry_log(telemetry_records, args.telemetry_log)
    logger.info(f"Saved telemetry to {args.telemetry_log}")
    
    # Stage 3: Ground Truth Labeling
    logger.info("Stage 3: Ground Truth Labeling")
    labeled_records = label_ground_truth(args, telemetry_records)
    logger.info(f"Labeled {len(labeled_records)} records with ground truth")
    
    # Stage 4: Statistical Fitting
    logger.info("Stage 4: Statistical Fitting")
    noise_models = fit_noise_models(args, labeled_records)
    logger.info(f"Fitted noise models for {len(noise_models)} tiers")
    
    # Stage 5: Cross-Tier Validation
    logger.info("Stage 5: Cross-Tier Validation")
    validation_report = validate_noise_models(args, noise_models, labeled_records)
    logger.info(f"Validation report: {validation_report['summary']}")
    
    # Stage 6: Model Export
    logger.info("Stage 6: Model Export")
    export_noise_models(args, noise_models, validation_report)
    logger.info(f"Exported noise models to {args.export}")
    
    logger.info("=" * 80)
    logger.info("Calibration complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
```

---

## 4. Stage 1: Corpus Selection

### 4.1 Corpus Loaders

```python
import re
from pathlib import Path
from typing import List

def select_corpus(args: argparse.Namespace) -> List[str]:
    """Select proof corpus based on arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        List of module names to verify
    """
    
    if args.corpus == "mathlib":
        modules = load_mathlib_corpus()
    elif args.corpus == "curriculum":
        modules = load_curriculum_corpus()
    elif args.corpus == "synthetic":
        modules = generate_synthetic_corpus(args.n)
    elif args.corpus == "custom":
        modules = load_custom_corpus(args.corpus_path)
    else:
        raise ValueError(f"Unknown corpus: {args.corpus}")
    
    # Apply filter if specified
    if args.corpus_filter:
        pattern = re.compile(args.corpus_filter)
        modules = [m for m in modules if pattern.match(m)]
    
    # Sample modules if needed (to reach target count)
    if len(modules) > args.n:
        import random
        random.seed(args.seed)
        modules = random.sample(modules, args.n)
    
    return modules


def load_mathlib_corpus() -> List[str]:
    """Load Mathlib corpus from installed Mathlib."""
    # Find Mathlib installation
    import subprocess
    
    result = subprocess.run(
        ["lake", "env", "printenv", "LEAN_PATH"],
        capture_output=True,
        text=True,
    )
    
    lean_path = result.stdout.strip()
    mathlib_dir = Path(lean_path) / "Mathlib"
    
    # Find all .lean files
    modules = []
    for lean_file in mathlib_dir.rglob("*.lean"):
        # Convert file path to module name
        rel_path = lean_file.relative_to(mathlib_dir.parent)
        module_name = str(rel_path.with_suffix("")).replace("/", ".")
        modules.append(module_name)
    
    return modules


def load_curriculum_corpus() -> List[str]:
    """Load curriculum corpus from experiments/curriculum."""
    curriculum_dir = Path("experiments/curriculum")
    
    modules = []
    for lean_file in curriculum_dir.rglob("*.lean"):
        rel_path = lean_file.relative_to(curriculum_dir)
        module_name = f"Curriculum.{str(rel_path.with_suffix('')).replace('/', '.')}"
        modules.append(module_name)
    
    return modules


def generate_synthetic_corpus(n: int) -> List[str]:
    """Generate synthetic proof corpus."""
    # Generate n synthetic module names
    modules = [f"Synthetic.Proof{i:06d}" for i in range(n)]
    return modules


def load_custom_corpus(corpus_path: Path) -> List[str]:
    """Load custom corpus from file.
    
    Expected format: one module name per line
    """
    with open(corpus_path, "r") as f:
        modules = [line.strip() for line in f if line.strip()]
    
    return modules
```

---

## 5. Stage 2: Telemetry Collection

### 5.1 Worker Pool Model

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from backend.verification.telemetry_runtime import run_lean_with_monitoring
from backend.verification.error_codes import VerifierTier

@dataclass
class TelemetryTask:
    """Task for telemetry collection."""
    module_name: str
    tier: VerifierTier
    timeout_s: float
    context: str
    master_seed: int


def collect_telemetry(
    args: argparse.Namespace,
    corpus: List[str],
) -> List[Dict[str, Any]]:
    """Collect telemetry from corpus using worker pool.
    
    Args:
        args: Parsed command-line arguments
        corpus: List of module names to verify
    
    Returns:
        List of telemetry records (as dicts)
    """
    
    # Create tasks for all tier-module combinations
    tasks = []
    for tier_name in args.tiers:
        tier = VerifierTier[tier_name]
        timeout_s = get_timeout_for_tier(tier)
        
        for i, module_name in enumerate(corpus):
            context = f"calibration_{tier_name}_{module_name}"
            task = TelemetryTask(
                module_name=module_name,
                tier=tier,
                timeout_s=timeout_s,
                context=context,
                master_seed=args.seed,
            )
            tasks.append(task)
    
    logger.info(f"Created {len(tasks)} telemetry collection tasks")
    
    # Execute tasks in parallel
    telemetry_records = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(run_telemetry_task, task): task
            for task in tasks
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(futures)):
            task = futures[future]
            
            try:
                telemetry = future.result()
                telemetry_records.append(asdict(telemetry))
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{len(tasks)} tasks completed")
                    
            except Exception as e:
                logger.error(f"Task failed for {task.module_name} ({task.tier}): {e}")
    
    logger.info(f"Completed {len(telemetry_records)}/{len(tasks)} tasks")
    
    return telemetry_records


def run_telemetry_task(task: TelemetryTask):
    """Run single telemetry collection task (executed in worker process)."""
    
    telemetry = run_lean_with_monitoring(
        module_name=task.module_name,
        tier=task.tier,
        timeout_s=task.timeout_s,
        context=task.context,
        master_seed=task.master_seed,
        noise_config=None,  # No noise injection during calibration
    )
    
    return telemetry


def get_timeout_for_tier(tier: VerifierTier) -> float:
    """Get default timeout for tier."""
    timeouts = {
        VerifierTier.FAST_NOISY: 30.0,
        VerifierTier.BALANCED: 60.0,
        VerifierTier.SLOW_PRECISE: 120.0,
    }
    return timeouts[tier]


def save_telemetry_log(
    telemetry_records: List[Dict[str, Any]],
    log_path: Path,
) -> None:
    """Save telemetry records to JSONL file."""
    
    with open(log_path, "w") as f:
        for record in telemetry_records:
            f.write(json.dumps(record) + "\n")
```

---

## 6. Stage 3: Ground Truth Labeling

### 6.1 Labeling Pipeline DAG

```
                    ┌─────────────────┐
                    │ Sample Selection│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────┐
    │  Consensus  │  │   Extended   │  │  Manual  │
    │Verification │  │   Timeout    │  │ Labeling │
    └──────┬──────┘  └──────┬───────┘  └────┬─────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Label Merging  │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │ Labeled Records│
                    └────────────────┘
```

### 6.2 Implementation

```python
from enum import Enum

class GroundTruthMethod(Enum):
    """Ground truth labeling method."""
    CONSENSUS = "consensus"
    EXTENDED_TIMEOUT = "extended_timeout"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class LabeledRecord:
    """Telemetry record with ground truth label."""
    telemetry: Dict[str, Any]
    ground_truth: str  # "VERIFIED" or "INVALID"
    labeling_method: GroundTruthMethod
    confidence: float  # 0.0 to 1.0


def label_ground_truth(
    args: argparse.Namespace,
    telemetry_records: List[Dict[str, Any]],
) -> List[LabeledRecord]:
    """Label telemetry records with ground truth.
    
    Args:
        args: Parsed command-line arguments
        telemetry_records: List of telemetry records
    
    Returns:
        List of labeled records
    """
    
    # Sample records for labeling
    sample_records = sample_for_labeling(telemetry_records, args.label_sample, args.seed)
    logger.info(f"Sampled {len(sample_records)} records for ground truth labeling")
    
    # Apply labeling methods in parallel
    labeled_records = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Method 1: Consensus verification
        logger.info("Applying consensus verification...")
        consensus_futures = {
            executor.submit(label_with_consensus, record, args): record
            for record in sample_records
            if record["outcome"] in ["VERIFIED", "PROOF_INVALID"]
        }
        
        for future in as_completed(consensus_futures):
            try:
                labeled = future.result()
                if labeled.confidence > 0.8:  # High confidence
                    labeled_records.append(labeled)
            except Exception as e:
                logger.error(f"Consensus labeling failed: {e}")
        
        # Method 2: Extended timeout
        logger.info("Applying extended timeout...")
        timeout_futures = {
            executor.submit(label_with_extended_timeout, record, args): record
            for record in sample_records
            if record["outcome"] == "VERIFIER_TIMEOUT"
        }
        
        for future in as_completed(timeout_futures):
            try:
                labeled = future.result()
                if labeled.confidence > 0.8:
                    labeled_records.append(labeled)
            except Exception as e:
                logger.error(f"Extended timeout labeling failed: {e}")
    
    # Method 3: Manual labeling (interactive)
    if args.verbose:
        logger.info("Manual labeling available for remaining records...")
        # TODO: Implement interactive manual labeling UI
    
    logger.info(f"Labeled {len(labeled_records)} records with ground truth")
    
    return labeled_records


def sample_for_labeling(
    telemetry_records: List[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample records for ground truth labeling.
    
    Stratified sampling to ensure coverage of all outcome types.
    """
    import random
    from collections import defaultdict
    
    random.seed(seed)
    
    # Group by outcome
    by_outcome = defaultdict(list)
    for record in telemetry_records:
        by_outcome[record["outcome"]].append(record)
    
    # Sample proportionally from each outcome
    sampled = []
    for outcome, records in by_outcome.items():
        k = min(len(records), n // len(by_outcome))
        sampled.extend(random.sample(records, k))
    
    # Fill remaining slots randomly
    remaining = n - len(sampled)
    if remaining > 0:
        pool = [r for r in telemetry_records if r not in sampled]
        sampled.extend(random.sample(pool, min(remaining, len(pool))))
    
    return sampled


def label_with_consensus(
    record: Dict[str, Any],
    args: argparse.Namespace,
) -> LabeledRecord:
    """Label record using consensus verification."""
    
    # Run verification with multiple verifiers
    outcomes = []
    for verifier in args.consensus_verifiers:
        # TODO: Run verification with different verifier
        # For now, assume only Lean is available
        outcome = record["outcome"]
        outcomes.append(outcome)
    
    # Take majority vote
    from collections import Counter
    vote_counts = Counter(outcomes)
    ground_truth, count = vote_counts.most_common(1)[0]
    confidence = count / len(outcomes)
    
    return LabeledRecord(
        telemetry=record,
        ground_truth=ground_truth,
        labeling_method=GroundTruthMethod.CONSENSUS,
        confidence=confidence,
    )


def label_with_extended_timeout(
    record: Dict[str, Any],
    args: argparse.Namespace,
) -> LabeledRecord:
    """Label timeout record using extended timeout."""
    
    # Re-run verification with extended timeout
    extended_timeout_s = record["timeout_s"] * args.timeout_multiplier
    
    telemetry = run_lean_with_monitoring(
        module_name=record["module_name"],
        tier=VerifierTier[record["tier"]],
        timeout_s=extended_timeout_s,
        context=record["context"] + "_extended",
        master_seed=record["metadata"]["master_seed"],
        noise_config=None,
    )
    
    # If extended verification succeeds, original timeout was spurious
    if telemetry.outcome == VerifierErrorCode.VERIFIED:
        ground_truth = "VERIFIED"
        confidence = 0.9
    elif telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
        # Still times out, likely genuinely hard
        ground_truth = "TIMEOUT"
        confidence = 0.7
    else:
        # Failed with extended timeout, likely invalid
        ground_truth = "INVALID"
        confidence = 0.8
    
    return LabeledRecord(
        telemetry=record,
        ground_truth=ground_truth,
        labeling_method=GroundTruthMethod.EXTENDED_TIMEOUT,
        confidence=confidence,
    )
```

---

## 7. Stage 4: Statistical Fitting

### 7.1 MLE Estimation

```python
from scipy import stats
import numpy as np

@dataclass
class NoiseModelFit:
    """Fitted noise model for a tier."""
    tier: str
    
    # Noise rates
    timeout_rate: float
    timeout_rate_ci: Tuple[float, float]
    
    spurious_fail_rate: float
    spurious_fail_rate_ci: Tuple[float, float]
    
    spurious_pass_rate: float
    spurious_pass_rate_ci: Tuple[float, float]
    
    # Timeout distribution
    timeout_distribution: str
    timeout_distribution_params: Dict[str, float]
    timeout_distribution_gof: Dict[str, float]
    
    # Metadata
    training_samples: int
    calibration_date: str


def fit_noise_models(
    args: argparse.Namespace,
    labeled_records: List[LabeledRecord],
) -> Dict[str, NoiseModelFit]:
    """Fit noise models for all tiers.
    
    Args:
        args: Parsed command-line arguments
        labeled_records: List of labeled records
    
    Returns:
        Dict mapping tier name to fitted noise model
    """
    
    noise_models = {}
    
    for tier_name in args.tiers:
        logger.info(f"Fitting noise model for tier {tier_name}...")
        
        # Filter records for this tier
        tier_records = [
            r for r in labeled_records
            if r.telemetry["tier"] == tier_name
        ]
        
        if not tier_records:
            logger.warning(f"No labeled records for tier {tier_name}, skipping")
            continue
        
        # Fit noise model
        noise_model = fit_noise_model_for_tier(tier_name, tier_records)
        noise_models[tier_name] = noise_model
        
        logger.info(f"  Timeout rate: {noise_model.timeout_rate:.4f} {noise_model.timeout_rate_ci}")
        logger.info(f"  Spurious fail rate: {noise_model.spurious_fail_rate:.4f} {noise_model.spurious_fail_rate_ci}")
        logger.info(f"  Spurious pass rate: {noise_model.spurious_pass_rate:.4f} {noise_model.spurious_pass_rate_ci}")
        logger.info(f"  Timeout distribution: {noise_model.timeout_distribution} {noise_model.timeout_distribution_params}")
    
    return noise_models


def fit_noise_model_for_tier(
    tier_name: str,
    labeled_records: List[LabeledRecord],
) -> NoiseModelFit:
    """Fit noise model for a single tier."""
    
    import datetime
    
    # Extract outcomes and ground truth
    outcomes = [r.telemetry["outcome"] for r in labeled_records]
    ground_truths = [r.ground_truth for r in labeled_records]
    
    # Compute noise rates
    n_total = len(labeled_records)
    n_timeout = sum(1 for o in outcomes if o == "VERIFIER_TIMEOUT")
    timeout_rate = n_timeout / n_total
    timeout_rate_ci = wilson_confidence_interval(n_timeout, n_total)
    
    # Spurious fail rate: P(outcome=FAILED | ground_truth=VERIFIED)
    verified_records = [
        (o, gt) for o, gt in zip(outcomes, ground_truths)
        if gt == "VERIFIED"
    ]
    if verified_records:
        n_verified = len(verified_records)
        n_spurious_fail = sum(1 for o, gt in verified_records if o == "PROOF_INVALID")
        spurious_fail_rate = n_spurious_fail / n_verified
        spurious_fail_rate_ci = wilson_confidence_interval(n_spurious_fail, n_verified)
    else:
        spurious_fail_rate = 0.0
        spurious_fail_rate_ci = (0.0, 0.0)
    
    # Spurious pass rate: P(outcome=VERIFIED | ground_truth=INVALID)
    invalid_records = [
        (o, gt) for o, gt in zip(outcomes, ground_truths)
        if gt == "INVALID"
    ]
    if invalid_records:
        n_invalid = len(invalid_records)
        n_spurious_pass = sum(1 for o, gt in invalid_records if o == "VERIFIED")
        spurious_pass_rate = n_spurious_pass / n_invalid
        spurious_pass_rate_ci = wilson_confidence_interval(n_spurious_pass, n_invalid)
    else:
        spurious_pass_rate = 0.0
        spurious_pass_rate_ci = (0.0, 0.0)
    
    # Fit timeout distribution
    timeout_durations = [
        r.telemetry["duration_ms"] for r in labeled_records
        if r.telemetry["outcome"] == "VERIFIER_TIMEOUT"
    ]
    
    if timeout_durations:
        dist_name, dist_params, gof = fit_timeout_distribution(timeout_durations)
    else:
        dist_name = "exponential"
        dist_params = {"lambda": 1.0}
        gof = {}
    
    # Construct noise model
    noise_model = NoiseModelFit(
        tier=tier_name,
        timeout_rate=timeout_rate,
        timeout_rate_ci=timeout_rate_ci,
        spurious_fail_rate=spurious_fail_rate,
        spurious_fail_rate_ci=spurious_fail_rate_ci,
        spurious_pass_rate=spurious_pass_rate,
        spurious_pass_rate_ci=spurious_pass_rate_ci,
        timeout_distribution=dist_name,
        timeout_distribution_params=dist_params,
        timeout_distribution_gof=gof,
        training_samples=n_total,
        calibration_date=datetime.datetime.now().isoformat(),
    )
    
    return noise_model


def wilson_confidence_interval(
    n_success: int,
    n_total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion."""
    
    if n_total == 0:
        return (0.0, 0.0)
    
    from scipy.stats import norm
    
    p = n_success / n_total
    z = norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
    
    return (max(0.0, center - margin), min(1.0, center + margin))


def fit_timeout_distribution(
    durations: List[float],
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """Fit timeout distribution using MLE and AIC model selection."""
    
    durations_array = np.array(durations)
    
    # Candidate distributions
    distributions = {
        "exponential": stats.expon,
        "gamma": stats.gamma,
        "lognormal": stats.lognorm,
        "weibull": stats.weibull_min,
    }
    
    best_dist = None
    best_aic = np.inf
    best_params = {}
    best_gof = {}
    
    for name, dist in distributions.items():
        try:
            # Fit distribution
            params = dist.fit(durations_array)
            
            # Compute log-likelihood
            log_likelihood = np.sum(dist.logpdf(durations_array, *params))
            
            # Compute AIC
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            
            # Compute KS test
            ks_stat, ks_pvalue = stats.kstest(durations_array, lambda x: dist.cdf(x, *params))
            
            # Check if best
            if aic < best_aic:
                best_dist = name
                best_aic = aic
                best_params = {f"param{i}": p for i, p in enumerate(params)}
                best_gof = {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "aic": aic,
                }
        
        except Exception as e:
            logger.warning(f"Failed to fit {name} distribution: {e}")
    
    return best_dist, best_params, best_gof
```

---

## 8. Stage 5: Cross-Tier Validation

### 8.1 Hypothesis Testing

```python
from scipy.stats import norm

@dataclass
class ValidationReport:
    """Validation report for noise models."""
    summary: str
    tier_monotonicity: Dict[str, Any]
    calibration_error: Dict[str, float]
    coverage: Dict[str, float]
    passed: bool


def validate_noise_models(
    args: argparse.Namespace,
    noise_models: Dict[str, NoiseModelFit],
    labeled_records: List[LabeledRecord],
) -> ValidationReport:
    """Validate noise models using hypothesis testing.
    
    Args:
        args: Parsed command-line arguments
        noise_models: Fitted noise models
        labeled_records: Labeled records
    
    Returns:
        Validation report
    """
    
    # Split data into train/validation
    import random
    random.seed(args.seed)
    random.shuffle(labeled_records)
    
    split_idx = int(len(labeled_records) * (1 - args.validation_split))
    train_records = labeled_records[:split_idx]
    val_records = labeled_records[split_idx:]
    
    logger.info(f"Validation split: {len(train_records)} train, {len(val_records)} validation")
    
    # Test 1: Tier monotonicity
    tier_monotonicity = test_tier_monotonicity(noise_models, args.tiers)
    
    # Test 2: Calibration error
    calibration_error = compute_calibration_error(noise_models, val_records)
    
    # Test 3: Coverage
    coverage = compute_coverage(noise_models, val_records)
    
    # Determine if validation passed
    passed = (
        tier_monotonicity["passed"] and
        all(err < 0.01 for err in calibration_error.values()) and
        all(cov >= 0.95 for cov in coverage.values())
    )
    
    summary = "PASSED" if passed else "FAILED"
    
    report = ValidationReport(
        summary=summary,
        tier_monotonicity=tier_monotonicity,
        calibration_error=calibration_error,
        coverage=coverage,
        passed=passed,
    )
    
    return report


def test_tier_monotonicity(
    noise_models: Dict[str, NoiseModelFit],
    tiers: List[str],
) -> Dict[str, Any]:
    """Test that noise rates are monotonically decreasing across tiers."""
    
    # Order tiers by expected noise (FAST > BALANCED > SLOW)
    tier_order = ["FAST", "BALANCED", "SLOW"]
    ordered_tiers = [t for t in tier_order if t in tiers]
    
    results = []
    passed = True
    
    for i in range(len(ordered_tiers) - 1):
        tier1 = ordered_tiers[i]
        tier2 = ordered_tiers[i + 1]
        
        model1 = noise_models[tier1]
        model2 = noise_models[tier2]
        
        # Test: θ_tier1 >= θ_tier2
        p1 = model1.timeout_rate
        p2 = model2.timeout_rate
        n1 = model1.training_samples
        n2 = model2.training_samples
        
        # Two-proportion z-test
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z = (p1 - p2) / se if se > 0 else 0
        p_value = 1 - norm.cdf(z)  # One-tailed test
        
        test_passed = (z >= 0 and p_value < 0.05)
        passed = passed and test_passed
        
        results.append({
            "tier1": tier1,
            "tier2": tier2,
            "rate1": p1,
            "rate2": p2,
            "z_statistic": z,
            "p_value": p_value,
            "passed": test_passed,
        })
    
    return {
        "passed": passed,
        "tests": results,
    }


def compute_calibration_error(
    noise_models: Dict[str, NoiseModelFit],
    val_records: List[LabeledRecord],
) -> Dict[str, float]:
    """Compute calibration error on validation data."""
    
    errors = {}
    
    for tier_name, model in noise_models.items():
        tier_val_records = [
            r for r in val_records
            if r.telemetry["tier"] == tier_name
        ]
        
        if not tier_val_records:
            errors[tier_name] = 0.0
            continue
        
        # Compute empirical timeout rate
        n_total = len(tier_val_records)
        n_timeout = sum(
            1 for r in tier_val_records
            if r.telemetry["outcome"] == "VERIFIER_TIMEOUT"
        )
        empirical_rate = n_timeout / n_total
        
        # Calibration error = |empirical - predicted|
        error = abs(empirical_rate - model.timeout_rate)
        errors[tier_name] = error
    
    return errors


def compute_coverage(
    noise_models: Dict[str, NoiseModelFit],
    val_records: List[LabeledRecord],
) -> Dict[str, float]:
    """Compute coverage (fraction of validation data within confidence intervals)."""
    
    coverages = {}
    
    for tier_name, model in noise_models.items():
        tier_val_records = [
            r for r in val_records
            if r.telemetry["tier"] == tier_name
        ]
        
        if not tier_val_records:
            coverages[tier_name] = 1.0
            continue
        
        # Compute empirical timeout rate
        n_total = len(tier_val_records)
        n_timeout = sum(
            1 for r in tier_val_records
            if r.telemetry["outcome"] == "VERIFIER_TIMEOUT"
        )
        empirical_rate = n_timeout / n_total
        
        # Check if within confidence interval
        ci_lower, ci_upper = model.timeout_rate_ci
        within_ci = (ci_lower <= empirical_rate <= ci_upper)
        
        coverages[tier_name] = 1.0 if within_ci else 0.0
    
    return coverages
```

---

## 9. Stage 6: Model Export

### 9.1 YAML Export

```python
def export_noise_models(
    args: argparse.Namespace,
    noise_models: Dict[str, NoiseModelFit],
    validation_report: ValidationReport,
) -> None:
    """Export calibrated noise models to YAML."""
    
    # Construct export dict
    export_data = {
        "calibration_metadata": {
            "date": datetime.datetime.now().isoformat(),
            "corpus": args.corpus,
            "total_samples": sum(m.training_samples for m in noise_models.values()),
            "validation_split": args.validation_split,
            "seed": args.seed,
            "validation_passed": validation_report.passed,
        },
        "validation_report": {
            "summary": validation_report.summary,
            "tier_monotonicity": validation_report.tier_monotonicity,
            "calibration_error": validation_report.calibration_error,
            "coverage": validation_report.coverage,
        },
        "noise_models": {},
    }
    
    for tier_name, model in noise_models.items():
        export_data["noise_models"][tier_name.lower()] = {
            "timeout_rate": model.timeout_rate,
            "timeout_rate_ci": list(model.timeout_rate_ci),
            "spurious_fail_rate": model.spurious_fail_rate,
            "spurious_fail_rate_ci": list(model.spurious_fail_rate_ci),
            "spurious_pass_rate": model.spurious_pass_rate,
            "spurious_pass_rate_ci": list(model.spurious_pass_rate_ci),
            "timeout_distribution": {
                "type": model.timeout_distribution,
                "params": model.timeout_distribution_params,
                "goodness_of_fit": model.timeout_distribution_gof,
            },
            "training_samples": model.training_samples,
        }
    
    # Write to YAML
    with open(args.export, "w") as f:
        yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
```

---

## 10. Canonicalization for Telemetry Snapshotting

### 10.1 Telemetry Canonicalization

```python
def canonicalize_telemetry(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize telemetry record for reproducible snapshotting.
    
    Removes non-deterministic fields (timestamps, UUIDs) and sorts keys.
    """
    
    canonical = telemetry.copy()
    
    # Remove non-deterministic fields
    canonical.pop("verification_id", None)
    canonical.pop("timestamp", None)
    
    # Round floating-point fields to fixed precision
    for key in ["duration_ms", "cpu_time_ms", "memory_peak_mb", "memory_final_mb"]:
        if key in canonical and canonical[key] is not None:
            canonical[key] = round(canonical[key], 2)
    
    # Sort dict keys recursively
    canonical = sort_dict_keys(canonical)
    
    return canonical


def sort_dict_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sort dictionary keys."""
    
    sorted_dict = {}
    for key in sorted(d.keys()):
        value = d[key]
        if isinstance(value, dict):
            sorted_dict[key] = sort_dict_keys(value)
        else:
            sorted_dict[key] = value
    
    return sorted_dict


def snapshot_telemetry(
    telemetry_records: List[Dict[str, Any]],
    snapshot_path: Path,
) -> None:
    """Save canonicalized telemetry snapshot for regression testing."""
    
    canonical_records = [canonicalize_telemetry(r) for r in telemetry_records]
    
    with open(snapshot_path, "w") as f:
        json.dump(canonical_records, f, indent=2, sort_keys=True)
```

---

## 11. Deployment Checklist

- [ ] Implement argument parsing with `parse_args()`
- [ ] Implement corpus loaders (Mathlib, curriculum, synthetic, custom)
- [ ] Implement worker pool for parallel telemetry collection
- [ ] Implement ground truth labeling pipeline (consensus, extended timeout, manual)
- [ ] Implement MLE estimation for noise rates
- [ ] Implement Wilson confidence intervals
- [ ] Implement timeout distribution fitting with AIC model selection
- [ ] Implement cross-tier validation with hypothesis testing
- [ ] Implement YAML export for calibrated models
- [ ] Implement telemetry canonicalization for snapshotting
- [ ] Write integration tests for full calibration pipeline
- [ ] Test on real Mathlib corpus (10,000+ proofs)
- [ ] Benchmark calibration runtime and optimize if needed
- [ ] Document CLI interface and usage examples
- [ ] Create example calibration runs with different corpora

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*

**Status**: Implementation Ready  
**Next**: Task C3 (UnifiedNoiseModel Code Skeleton)
