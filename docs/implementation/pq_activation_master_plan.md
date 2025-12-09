# PQ Activation Implementation Master Plan

**Document Version**: 1.0  
**Author**: Manus-H  
**Date**: 2024-12-06  
**Status**: Implementation Blueprint

---

## Executive Summary

This document provides a comprehensive implementation plan for all PQ activation engineering deliverables. It consolidates specifications, implementation status, and next steps for:

1. **Consensus Module** ✅ COMPLETE (1,297 lines implemented)
2. **Benchmark Harness** 🔄 IN PROGRESS (micro + component benchmarks implemented)
3. **Drift Radar Runtime** 📋 SPECIFIED (implementation plan ready)
4. **Governance Orchestration Engine** 📋 SPECIFIED (implementation plan ready)
5. **Node Operator Upgrade Protocol** 📋 SPECIFIED (documentation plan ready)

---

## 1. Consensus Module Implementation ✅ COMPLETE

### Implementation Status

**Total Lines**: 1,297 lines of production code  
**Modules Implemented**: 6 core modules  
**Test Coverage**: Test suite catalogue defined (45+ tests)

### Module Inventory

| Module | Lines | Status | Description |
|:---|---:|:---|:---|
| `rules.py` | 198 | ✅ Complete | Phase-specific consensus rules |
| `epoch.py` | 219 | ✅ Complete | Epoch management and resolution |
| `prev_hash.py` | 154 | ✅ Complete | Prev-hash validation |
| `validation.py` | 193 | ✅ Complete | Block validation logic |
| `violations.py` | 284 | ✅ Complete | Violation detection and classification |
| `reorg.py` | 249 | ✅ Complete | Reorganization handling |

### Key Features Implemented

- **Phase-aware validation**: 5 consensus rule versions (v1-legacy through v3-pq-only)
- **Epoch management**: Registration, resolution, transition detection
- **Dual-chain validation**: Legacy and PQ prev-hash verification
- **Comprehensive violation detection**: 11 violation types with severity classification
- **Reorg handling**: Fork choice, finality enforcement, epoch boundary protection

### Next Steps

1. Implement unit test suite (45+ tests defined in runtime plan)
2. Implement integration test suite (phase transitions, epoch cutover, reorg scenarios)
3. Deploy to testnet for validation
4. Conduct security audit

---

## 2. Benchmark Harness Implementation 🔄 IN PROGRESS

### Implementation Status

**Completed**:
- ✅ Micro-benchmarks (`micro_benchmarks.py`, 291 lines)
- ✅ Component benchmarks (`component_benchmarks.py`, 358 lines)

**Remaining**:
- 📋 Integration benchmarks
- 📋 Network benchmarks
- 📋 5-week orchestration script
- 📋 Acceptance criteria validator

### Micro-Benchmarks (`micro_benchmarks.py`)

**Purpose**: Measure raw hash algorithm performance

**Test Matrix**:
- **Algorithms**: SHA-256, SHA3-256, BLAKE3
- **Input Sizes**: 32B, 64B, 128B, 256B, 512B, 1KB, 4KB, 16KB, 64KB
- **Iterations**: 10,000 per test case
- **Metrics**: Mean, median, P90, P95, P99, throughput (MB/s)

**Key Functions**:
```python
def benchmark_sha256(input_size: int, iterations: int) -> MicroBenchmarkResult
def benchmark_sha3_256(input_size: int, iterations: int) -> MicroBenchmarkResult
def benchmark_blake3(input_size: int, iterations: int) -> MicroBenchmarkResult
def run_micro_benchmarks() -> List[MicroBenchmarkResult]
def compare_algorithms(results: List[MicroBenchmarkResult]) -> None
def export_results_csv(results: List[MicroBenchmarkResult], filename: str) -> None
```

### Component Benchmarks (`component_benchmarks.py`)

**Purpose**: Measure Merkle tree, block sealing, and validation performance

**Test Matrix**:
- **Operations**: Merkle tree construction, dual Merkle tree, dual commitment, block seal, block validation
- **Statement Counts**: 10, 50, 100, 500, 1000
- **Iterations**: 100 per test case
- **Metrics**: Mean, median, P90, P95, overhead percentage

**Key Functions**:
```python
def benchmark_merkle_tree(statements, algorithm_id, algorithm_name, iterations) -> ComponentBenchmarkResult
def benchmark_dual_merkle_tree(statements, iterations) -> ComponentBenchmarkResult
def benchmark_dual_commitment(statements, iterations) -> ComponentBenchmarkResult
def benchmark_block_seal_legacy(statements, prev_hash, block_number, iterations) -> ComponentBenchmarkResult
def benchmark_block_seal_dual(statements, prev_hash, pq_prev_hash, block_number, iterations) -> ComponentBenchmarkResult
def benchmark_block_validation(block, prev_block, iterations) -> ComponentBenchmarkResult
def run_component_benchmarks() -> List[ComponentBenchmarkResult]
```

### Integration Benchmarks (TO IMPLEMENT)

**File**: `backend/benchmarks/integration_benchmarks.py`

**Purpose**: Measure end-to-end block validation and chain synchronization

**Test Scenarios**:
1. **Full Block Validation**: Validate blocks with varying statement counts (10-1000)
2. **Chain Validation**: Validate chains of varying lengths (10-1000 blocks)
3. **Historical Verification**: Verify blocks across epoch boundaries
4. **Epoch Transition**: Validate cutover blocks

**Implementation Outline**:
```python
@dataclass
class IntegrationBenchmarkResult:
    scenario_name: str
    block_count: int
    statement_count_per_block: int
    total_validation_time_ms: float
    mean_block_time_ms: float
    throughput_blocks_per_second: float

def benchmark_full_block_validation(block_count, statements_per_block, iterations) -> IntegrationBenchmarkResult
def benchmark_chain_validation(chain_length, statements_per_block, iterations) -> IntegrationBenchmarkResult
def benchmark_historical_verification(epoch_count, blocks_per_epoch, iterations) -> IntegrationBenchmarkResult
def benchmark_epoch_transition(iterations) -> IntegrationBenchmarkResult
def run_integration_benchmarks() -> List[IntegrationBenchmarkResult]
```

### Network Benchmarks (TO IMPLEMENT)

**File**: `backend/benchmarks/network_benchmarks.py`

**Purpose**: Measure network propagation and multi-node consensus

**Test Scenarios**:
1. **Block Propagation**: Time for block to reach 90% of nodes
2. **Consensus Latency**: Time for network to reach consensus
3. **Bandwidth Usage**: Network traffic for block propagation
4. **Orphan Rate**: Percentage of orphaned blocks

**Implementation Outline**:
```python
@dataclass
class NetworkBenchmarkResult:
    scenario_name: str
    node_count: int
    block_size_bytes: int
    propagation_time_ms: float
    consensus_latency_ms: float
    bandwidth_usage_mbps: float
    orphan_rate: float

def benchmark_block_propagation(node_count, block_size, iterations) -> NetworkBenchmarkResult
def benchmark_consensus_latency(node_count, iterations) -> NetworkBenchmarkResult
def benchmark_bandwidth_usage(node_count, block_size, iterations) -> NetworkBenchmarkResult
def run_network_benchmarks() -> List[NetworkBenchmarkResult]
```

**Note**: Network benchmarks require multi-node testnet deployment.

### 5-Week Orchestration Script (TO IMPLEMENT)

**File**: `backend/benchmarks/orchestrate_benchmarks.py`

**Purpose**: Automate 5-week benchmark execution plan

**Week-by-Week Plan**:

**Week 1: Micro-Benchmarks**
```python
def week1_micro_benchmarks():
    """Run micro-benchmarks and establish baseline."""
    results = run_micro_benchmarks()
    compare_algorithms(results)
    export_results_csv(results, "week1_micro_results.csv")
    generate_report("Week 1: Micro-Benchmark Baseline")
```

**Week 2: Component Benchmarks**
```python
def week2_component_benchmarks():
    """Run component benchmarks and measure overhead."""
    results = run_component_benchmarks()
    analyze_overhead(results)
    export_results_csv(results, "week2_component_results.csv")
    generate_report("Week 2: Component Overhead Analysis")
```

**Week 3: Integration Benchmarks**
```python
def week3_integration_benchmarks():
    """Run integration benchmarks and validate end-to-end performance."""
    results = run_integration_benchmarks()
    validate_acceptance_criteria(results)
    export_results_csv(results, "week3_integration_results.csv")
    generate_report("Week 3: Integration Performance Validation")
```

**Week 4: Network Benchmarks**
```python
def week4_network_benchmarks():
    """Run network benchmarks on testnet."""
    results = run_network_benchmarks()
    analyze_network_impact(results)
    export_results_csv(results, "week4_network_results.csv")
    generate_report("Week 4: Network Impact Assessment")
```

**Week 5: Stress Benchmarks**
```python
def week5_stress_benchmarks():
    """Run stress tests to identify breaking points."""
    results = run_stress_benchmarks()
    identify_bottlenecks(results)
    export_results_csv(results, "week5_stress_results.csv")
    generate_report("Week 5: Stress Testing and Bottleneck Analysis")
```

### Acceptance Criteria Validator (TO IMPLEMENT)

**File**: `backend/benchmarks/acceptance_validator.py`

**Purpose**: Validate benchmark results against acceptance criteria

**Acceptance Criteria**:
```python
ACCEPTANCE_CRITERIA = {
    "block_sealing_overhead": {"max": 300, "unit": "percent"},
    "block_validation_latency": {"max": 100, "unit": "ms"},
    "storage_overhead": {"max": 30, "unit": "percent"},
    "network_propagation": {"max": 2.0, "unit": "multiplier"},
}

def validate_acceptance_criteria(results: List[BenchmarkResult]) -> AcceptanceReport:
    """Validate results against acceptance criteria."""
    report = AcceptanceReport()
    
    for criterion, threshold in ACCEPTANCE_CRITERIA.items():
        actual_value = extract_metric(results, criterion)
        passed = actual_value <= threshold["max"]
        report.add_result(criterion, actual_value, threshold, passed)
    
    return report
```

---

## 3. Drift Radar Runtime Implementation 📋 SPECIFIED

### Implementation Plan

**Directory Structure**:
```
backend/monitoring/
├── __init__.py
├── drift_detector.py          # Core drift detection engine
├── algorithm_monitor.py       # Algorithm mismatch detection
├── consistency_monitor.py     # Cross-algorithm consistency
├── lineage_monitor.py         # Prev-hash lineage tracking
├── commitment_monitor.py      # Dual-commitment verification
├── performance_monitor.py     # Performance drift detection
├── storage_monitor.py         # Storage overhead monitoring
├── network_monitor.py         # Network propagation monitoring
├── alerting.py                # Alert generation and dispatch
└── dashboard.py               # Real-time monitoring dashboard
```

### Detector Class Skeletons

#### Algorithm Monitor (`algorithm_monitor.py`)

```python
@dataclass
class AlgorithmMismatchAlert:
    block_number: int
    block_hash: str
    expected_algorithm: int
    actual_algorithm: int
    severity: str
    message: str

class AlgorithmMonitor:
    """Monitors for algorithm mismatches."""
    
    def __init__(self):
        self.alert_history = []
    
    def detect_algorithm_mismatch(self, block: BlockHeaderPQ) -> Optional[AlgorithmMismatchAlert]:
        """Detect if block uses wrong algorithm for its epoch."""
        epoch = get_epoch_for_block(block.block_number)
        canonical_algorithm = epoch.algorithm_id
        
        if block.pq_algorithm != canonical_algorithm:
            return AlgorithmMismatchAlert(
                block_number=block.block_number,
                block_hash=hash_block_header_historical(block),
                expected_algorithm=canonical_algorithm,
                actual_algorithm=block.pq_algorithm,
                severity="CRITICAL",
                message=f"Algorithm mismatch: expected {canonical_algorithm:02x}, got {block.pq_algorithm:02x}",
            )
        
        return None
    
    def monitor_window(self, blocks: List[BlockHeaderPQ]) -> List[AlgorithmMismatchAlert]:
        """Monitor a window of blocks for algorithm mismatches."""
        alerts = []
        for block in blocks:
            alert = self.detect_algorithm_mismatch(block)
            if alert:
                alerts.append(alert)
        return alerts
```

#### Consistency Monitor (`consistency_monitor.py`)

```python
@dataclass
class ConsistencyAlert:
    block_number: int
    block_hash: str
    legacy_valid: bool
    pq_valid: bool
    expected_legacy_root: str
    actual_legacy_root: str
    expected_pq_root: str
    actual_pq_root: str
    severity: str
    message: str

class ConsistencyMonitor:
    """Monitors cross-algorithm consistency."""
    
    def detect_cross_algorithm_inconsistency(self, block: BlockHeaderPQ) -> Optional[ConsistencyAlert]:
        """Detect inconsistency between legacy and PQ hashes."""
        if not block.has_dual_commitment():
            return None
        
        # Recompute both Merkle roots
        computed_legacy_root = merkle_root_versioned(block.statements, algorithm_id=0x00)
        computed_pq_root = merkle_root_versioned(block.statements, algorithm_id=block.pq_algorithm)
        
        legacy_valid = (computed_legacy_root == block.merkle_root)
        pq_valid = (computed_pq_root == block.pq_merkle_root)
        
        if not legacy_valid or not pq_valid:
            return ConsistencyAlert(
                block_number=block.block_number,
                block_hash=hash_block_header_historical(block),
                legacy_valid=legacy_valid,
                pq_valid=pq_valid,
                expected_legacy_root=computed_legacy_root,
                actual_legacy_root=block.merkle_root,
                expected_pq_root=computed_pq_root,
                actual_pq_root=block.pq_merkle_root,
                severity="CRITICAL",
                message="Merkle root inconsistency detected",
            )
        
        return None
```

### Drift Schema

```python
@dataclass
class DriftEvent:
    """Represents a detected drift event."""
    event_id: str
    timestamp: float
    drift_category: str  # "algorithm_mismatch", "consistency", etc.
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    block_number: int
    block_hash: Optional[str]
    message: str
    details: dict
    resolved: bool
    resolution_timestamp: Optional[float]
```

### Alert Routing Map

```python
ALERT_ROUTING = {
    "CRITICAL": {
        "channels": ["sms", "email", "slack", "network_broadcast"],
        "recipients": {
            "sms": ["+1234567890"],
            "email": ["ops@mathledger.network"],
            "slack": ["#pq-migration-alerts"],
        },
        "throttle_seconds": 0,  # No throttling for critical
    },
    "HIGH": {
        "channels": ["email", "slack"],
        "recipients": {
            "email": ["ops@mathledger.network"],
            "slack": ["#pq-migration-alerts"],
        },
        "throttle_seconds": 300,  # 5 minutes
    },
    "MEDIUM": {
        "channels": ["email"],
        "recipients": {
            "email": ["ops@mathledger.network"],
        },
        "throttle_seconds": 3600,  # 1 hour
    },
    "LOW": {
        "channels": ["dashboard"],
        "recipients": {},
        "throttle_seconds": 86400,  # 24 hours
    },
}
```

### Aggregator API

```python
class DriftAggregator:
    """Aggregates drift events from multiple monitoring agents."""
    
    def __init__(self):
        self.events = []
        self.monitors = {
            "algorithm": AlgorithmMonitor(),
            "consistency": ConsistencyMonitor(),
            "lineage": LineageMonitor(),
            "commitment": CommitmentMonitor(),
            "performance": PerformanceMonitor(),
            "storage": StorageMonitor(),
            "network": NetworkMonitor(),
        }
    
    def collect_events(self, blocks: List[BlockHeaderPQ]) -> List[DriftEvent]:
        """Collect drift events from all monitors."""
        events = []
        
        for monitor_name, monitor in self.monitors.items():
            monitor_events = monitor.detect_drift(blocks)
            events.extend(monitor_events)
        
        return events
    
    def aggregate_and_route(self, events: List[DriftEvent]) -> None:
        """Aggregate events and route alerts."""
        for event in events:
            # Store event
            self.events.append(event)
            
            # Route alert
            route_alert(event)
            
            # Update dashboard
            update_dashboard(event)
```

### Dashboard JSON Schema

```json
{
  "dashboard_data": {
    "timestamp": 1701900000.0,
    "current_block": 1000000,
    "current_epoch": {
      "start_block": 1000000,
      "algorithm_id": 1,
      "algorithm_name": "SHA3-256",
      "rule_version": "v2-dual-required"
    },
    "network_health": "HEALTHY",
    "metrics": {
      "algorithm_monitoring": {
        "mismatch_count_last_1000": 0,
        "expected_algorithm_distribution": {"SHA-256": 0, "SHA3-256": 1000}
      },
      "consistency_monitoring": {
        "consistency_rate": 1.0,
        "validation_success_rate": 0.999
      },
      "lineage_monitoring": {
        "legacy_chain_valid": true,
        "pq_chain_valid": true,
        "drift_alerts": []
      },
      "performance_monitoring": {
        "avg_sealing_time_ms": 75.0,
        "avg_validation_time_ms": 45.0,
        "p95_sealing_time_ms": 120.0
      },
      "storage_monitoring": {
        "avg_block_size_bytes": 2048,
        "storage_overhead_percent": 25.0
      },
      "network_monitoring": {
        "avg_propagation_time_ms": 800,
        "avg_peer_count": 50,
        "orphan_rate": 0.01
      }
    },
    "recent_alerts": [
      {
        "event_id": "alert-001",
        "timestamp": 1701899000.0,
        "severity": "HIGH",
        "message": "Performance drift detected",
        "resolved": false
      }
    ]
  }
}
```

### Drift Injection Test Harness

```python
class DriftInjector:
    """Injects drift scenarios for testing detection."""
    
    def inject_algorithm_mismatch(self, block: BlockHeaderPQ) -> BlockHeaderPQ:
        """Inject algorithm mismatch."""
        block.pq_algorithm = 0xFF  # Invalid algorithm
        return block
    
    def inject_consistency_violation(self, block: BlockHeaderPQ) -> BlockHeaderPQ:
        """Inject Merkle root mismatch."""
        block.pq_merkle_root = "0x" + "FF" * 32  # Invalid root
        return block
    
    def inject_lineage_drift(self, block: BlockHeaderPQ) -> BlockHeaderPQ:
        """Inject prev_hash mismatch."""
        block.pq_prev_hash = "0x" + "EE" * 32  # Invalid prev_hash
        return block
    
    def inject_commitment_inconsistency(self, block: BlockHeaderPQ) -> BlockHeaderPQ:
        """Inject dual commitment mismatch."""
        block.dual_commitment = "0x" + "DD" * 32  # Invalid commitment
        return block

def test_drift_detection():
    """Test drift detection for all scenarios."""
    injector = DriftInjector()
    detector = DriftDetector()
    
    # Test algorithm mismatch
    block = create_test_block()
    block_with_drift = injector.inject_algorithm_mismatch(block)
    alerts = detector.detect_drift([block_with_drift])
    assert len(alerts) > 0
    assert alerts[0].drift_category == "algorithm_mismatch"
    
    # Test consistency violation
    block = create_test_block()
    block_with_drift = injector.inject_consistency_violation(block)
    alerts = detector.detect_drift([block_with_drift])
    assert len(alerts) > 0
    assert alerts[0].drift_category == "consistency"
    
    # ... (similar tests for other drift categories)
```

---

## 4. Governance Orchestration Engine 📋 SPECIFIED

### Implementation Plan

**Directory Structure**:
```
backend/governance/
├── __init__.py
├── proposal.py                # Proposal lifecycle engine
├── reviewer.py                # Reviewer consensus workflow
├── voting.py                  # Vote ledger and tallying
├── attestation.py             # MPC attestation registrar
├── rollback.py                # Emergency rollback executor
└── rpc.py                     # Governance RPC API
```

### Proposal Lifecycle Engine (`proposal.py`)

```python
@dataclass
class ActivationProposal:
    """Represents a PQ activation proposal."""
    proposal_id: str
    proposal_type: str
    title: str
    proposer: str
    submission_timestamp: float
    activation_block: int
    epoch_parameters: dict
    rationale: str
    benchmark_results: dict
    risk_assessment: dict
    implementation_readiness: dict
    endorsements: List[dict]
    status: str  # "DRAFT", "SUBMITTED", "UNDER_REVIEW", "VOTING", "APPROVED", "REJECTED"

class ProposalLifecycleEngine:
    """Manages proposal lifecycle from submission to activation."""
    
    def __init__(self):
        self.proposals = {}
    
    def submit_proposal(self, proposal: ActivationProposal) -> str:
        """Submit a new activation proposal."""
        # Validate proposal structure
        self.validate_proposal_structure(proposal)
        
        # Check proposer eligibility
        if not self.is_eligible_proposer(proposal.proposer):
            raise ValueError("Proposer not eligible")
        
        # Record proposal on-chain
        proposal_hash = self.hash_proposal(proposal)
        self.proposals[proposal_hash] = proposal
        proposal.status = "SUBMITTED"
        
        # Emit event
        self.emit_event("ProposalSubmitted", {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal_hash,
            "proposer": proposal.proposer,
            "activation_block": proposal.activation_block,
        })
        
        return proposal_hash
    
    def advance_to_review(self, proposal_hash: str) -> None:
        """Advance proposal to technical review phase."""
        proposal = self.proposals[proposal_hash]
        proposal.status = "UNDER_REVIEW"
        self.emit_event("ProposalUnderReview", {"proposal_hash": proposal_hash})
    
    def advance_to_voting(self, proposal_hash: str, review_report: dict) -> None:
        """Advance proposal to voting phase."""
        proposal = self.proposals[proposal_hash]
        
        if review_report["recommendation"] != "APPROVE":
            proposal.status = "REJECTED"
            self.emit_event("ProposalRejected", {"proposal_hash": proposal_hash})
            return
        
        proposal.status = "VOTING"
        self.emit_event("VotingStarted", {"proposal_hash": proposal_hash})
    
    def finalize_proposal(self, proposal_hash: str, vote_result: dict) -> None:
        """Finalize proposal based on vote result."""
        proposal = self.proposals[proposal_hash]
        
        if vote_result["outcome"] == "APPROVED":
            proposal.status = "APPROVED"
            self.emit_event("ProposalApproved", {"proposal_hash": proposal_hash})
            
            # Schedule activation
            self.schedule_activation(proposal)
        else:
            proposal.status = "REJECTED"
            self.emit_event("ProposalRejected", {"proposal_hash": proposal_hash})
```

### Reviewer Consensus Workflow (`reviewer.py`)

```python
@dataclass
class TechnicalReviewReport:
    """Technical review report for a proposal."""
    proposal_hash: str
    reviewer_id: str
    review_timestamp: float
    code_review: dict
    test_review: dict
    benchmark_review: dict
    security_review: dict
    docs_review: dict
    recommendation: str  # "APPROVE", "REJECT", "CONDITIONAL"
    reviewer_signature: str

class ReviewerConsensusWorkflow:
    """Manages reviewer consensus for proposals."""
    
    def __init__(self):
        self.reviews = {}
        self.reviewers = []
    
    def submit_review(self, review: TechnicalReviewReport) -> None:
        """Submit a technical review."""
        # Validate reviewer
        if review.reviewer_id not in self.reviewers:
            raise ValueError("Reviewer not authorized")
        
        # Validate signature
        if not self.verify_signature(review):
            raise ValueError("Invalid signature")
        
        # Store review
        if review.proposal_hash not in self.reviews:
            self.reviews[review.proposal_hash] = []
        self.reviews[review.proposal_hash].append(review)
        
        # Emit event
        self.emit_event("ReviewSubmitted", {
            "proposal_hash": review.proposal_hash,
            "reviewer_id": review.reviewer_id,
            "recommendation": review.recommendation,
        })
    
    def check_consensus(self, proposal_hash: str) -> Optional[str]:
        """Check if reviewers have reached consensus."""
        reviews = self.reviews.get(proposal_hash, [])
        
        if len(reviews) < 3:  # Require at least 3 reviews
            return None
        
        approvals = sum(1 for r in reviews if r.recommendation == "APPROVE")
        rejections = sum(1 for r in reviews if r.recommendation == "REJECT")
        
        if approvals >= 2 * len(reviews) / 3:
            return "APPROVE"
        elif rejections > len(reviews) / 3:
            return "REJECT"
        else:
            return None
```

### Vote Ledger (`voting.py`)

```python
@dataclass
class Vote:
    """Represents a vote on a proposal."""
    proposal_hash: str
    voter: str
    vote: str  # "YES", "NO", "ABSTAIN"
    voting_power: float
    timestamp: float
    signature: str

class VoteLedger:
    """Manages voting on proposals."""
    
    def __init__(self):
        self.votes = {}
    
    def cast_vote(self, vote: Vote) -> None:
        """Cast a vote on a proposal."""
        # Validate voter eligibility
        voting_power = self.get_voting_power(vote.voter)
        if voting_power == 0:
            raise ValueError("Voter has no voting power")
        
        # Validate signature
        if not self.verify_signature(vote):
            raise ValueError("Invalid signature")
        
        # Record vote on-chain
        if vote.proposal_hash not in self.votes:
            self.votes[vote.proposal_hash] = []
        self.votes[vote.proposal_hash].append(vote)
        
        # Emit event
        self.emit_event("VoteCast", {
            "proposal_hash": vote.proposal_hash,
            "voter": vote.voter,
            "vote": vote.vote,
            "voting_power": voting_power,
        })
    
    def tally_votes(self, proposal_hash: str) -> dict:
        """Tally votes for a proposal."""
        votes = self.votes.get(proposal_hash, [])
        
        yes_votes = sum(v.voting_power for v in votes if v.vote == "YES")
        no_votes = sum(v.voting_power for v in votes if v.vote == "NO")
        abstain_votes = sum(v.voting_power for v in votes if v.vote == "ABSTAIN")
        total_votes = yes_votes + no_votes + abstain_votes
        
        total_voting_power = self.get_total_voting_power()
        participation_rate = total_votes / total_voting_power
        approval_rate = yes_votes / (yes_votes + no_votes) if (yes_votes + no_votes) > 0 else 0
        
        quorum_met = participation_rate >= 0.40
        approval_met = approval_rate >= 0.667
        
        outcome = "APPROVED" if (quorum_met and approval_met) else "REJECTED"
        
        return {
            "proposal_hash": proposal_hash,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "abstain_votes": abstain_votes,
            "total_votes": total_votes,
            "participation_rate": participation_rate,
            "approval_rate": approval_rate,
            "outcome": outcome,
        }
```

### MPC Attestation Registrar (`attestation.py`)

```python
@dataclass
class Attestation:
    """Multi-party attestation for a migration milestone."""
    attestation_id: str
    milestone: str
    block_number: int
    block_hash: str
    pq_merkle_root: str
    timestamp: float
    attestors: List[dict]
    threshold: str
    status: str  # "PENDING", "VERIFIED", "INSUFFICIENT"

class AttestationRegistrar:
    """Manages multi-party attestations."""
    
    def __init__(self):
        self.attestations = {}
    
    def create_attestation(self, milestone: str, block_number: int) -> str:
        """Create a new attestation for a milestone."""
        block = get_block(block_number)
        
        attestation = Attestation(
            attestation_id=f"ATTEST-{milestone}-{block_number}",
            milestone=milestone,
            block_number=block_number,
            block_hash=hash_block_header_historical(block),
            pq_merkle_root=block.pq_merkle_root,
            timestamp=time.time(),
            attestors=[],
            threshold="5/7",
            status="PENDING",
        )
        
        self.attestations[attestation.attestation_id] = attestation
        return attestation.attestation_id
    
    def submit_attestation_signature(self, attestation_id: str, attestor_id: str, signature: str) -> None:
        """Submit an attestor signature."""
        attestation = self.attestations[attestation_id]
        
        # Verify signature
        if not self.verify_attestor_signature(attestor_id, attestation, signature):
            raise ValueError("Invalid attestor signature")
        
        # Add attestor
        attestation.attestors.append({
            "attestor_id": attestor_id,
            "signature": signature,
            "timestamp": time.time(),
        })
        
        # Check threshold
        required, total = map(int, attestation.threshold.split('/'))
        if len(attestation.attestors) >= required:
            attestation.status = "VERIFIED"
            self.record_attestation_on_chain(attestation)
```

### Emergency Rollback Executor (`rollback.py`)

```python
class EmergencyRollbackExecutor:
    """Executes emergency rollback to legacy algorithm."""
    
    def __init__(self):
        self.council_members = []
        self.rollback_history = []
    
    def declare_emergency(self, trigger: str, evidence: str, council_signatures: List[str]) -> str:
        """Declare migration emergency."""
        # Validate signatures (require 5/7)
        if len(council_signatures) < 5:
            raise ValueError("Insufficient council signatures")
        
        for sig in council_signatures:
            if not self.verify_council_signature(sig):
                raise ValueError("Invalid council signature")
        
        # Record emergency declaration
        emergency_hash = self.hash_emergency_declaration(trigger, evidence)
        
        # Emit event
        self.emit_event("EmergencyDeclared", {
            "emergency_hash": emergency_hash,
            "trigger": trigger,
            "timestamp": time.time(),
        })
        
        # Broadcast to all nodes
        self.broadcast_emergency_alert(emergency_hash, trigger)
        
        return emergency_hash
    
    def execute_rollback(self, emergency_hash: str, rollback_block: int) -> None:
        """Execute emergency rollback."""
        # Verify emergency declaration
        emergency = self.get_emergency_declaration(emergency_hash)
        if not emergency:
            raise ValueError("Invalid emergency declaration")
        
        # Create rollback epoch
        rollback_epoch = HashEpoch(
            start_block=rollback_block,
            end_block=None,
            algorithm_id=0x00,  # SHA-256
            algorithm_name="SHA-256-ROLLBACK",
            rule_version="v1-legacy",
            activation_timestamp=time.time(),
            governance_hash=emergency_hash,
        )
        
        # Register rollback epoch
        register_epoch(rollback_epoch)
        
        # Emit event
        self.emit_event("RollbackExecuted", {
            "emergency_hash": emergency_hash,
            "rollback_block": rollback_block,
            "rollback_algorithm": "SHA-256",
            "timestamp": time.time(),
        })
        
        # Broadcast to all nodes
        self.broadcast_rollback_notice(rollback_block)
```

### Governance RPC API (`rpc.py`)

```python
class GovernanceRPCAPI:
    """RPC API for governance operations."""
    
    def __init__(self):
        self.proposal_engine = ProposalLifecycleEngine()
        self.reviewer_workflow = ReviewerConsensusWorkflow()
        self.vote_ledger = VoteLedger()
        self.attestation_registrar = AttestationRegistrar()
        self.rollback_executor = EmergencyRollbackExecutor()
    
    # Proposal endpoints
    def submit_proposal(self, proposal: dict) -> dict:
        """Submit a new proposal."""
        proposal_obj = ActivationProposal(**proposal)
        proposal_hash = self.proposal_engine.submit_proposal(proposal_obj)
        return {"proposal_hash": proposal_hash, "status": "submitted"}
    
    def get_proposal(self, proposal_hash: str) -> dict:
        """Get proposal details."""
        proposal = self.proposal_engine.proposals.get(proposal_hash)
        if not proposal:
            raise ValueError("Proposal not found")
        return proposal.__dict__
    
    # Review endpoints
    def submit_review(self, review: dict) -> dict:
        """Submit a technical review."""
        review_obj = TechnicalReviewReport(**review)
        self.reviewer_workflow.submit_review(review_obj)
        return {"status": "review_submitted"}
    
    # Voting endpoints
    def cast_vote(self, vote: dict) -> dict:
        """Cast a vote."""
        vote_obj = Vote(**vote)
        self.vote_ledger.cast_vote(vote_obj)
        return {"status": "vote_cast"}
    
    def tally_votes(self, proposal_hash: str) -> dict:
        """Tally votes for a proposal."""
        return self.vote_ledger.tally_votes(proposal_hash)
    
    # Attestation endpoints
    def create_attestation(self, milestone: str, block_number: int) -> dict:
        """Create a new attestation."""
        attestation_id = self.attestation_registrar.create_attestation(milestone, block_number)
        return {"attestation_id": attestation_id}
    
    def submit_attestation_signature(self, attestation_id: str, attestor_id: str, signature: str) -> dict:
        """Submit an attestation signature."""
        self.attestation_registrar.submit_attestation_signature(attestation_id, attestor_id, signature)
        return {"status": "signature_submitted"}
    
    # Rollback endpoints
    def declare_emergency(self, trigger: str, evidence: str, council_signatures: List[str]) -> dict:
        """Declare an emergency."""
        emergency_hash = self.rollback_executor.declare_emergency(trigger, evidence, council_signatures)
        return {"emergency_hash": emergency_hash}
    
    def execute_rollback(self, emergency_hash: str, rollback_block: int) -> dict:
        """Execute emergency rollback."""
        self.rollback_executor.execute_rollback(emergency_hash, rollback_block)
        return {"status": "rollback_executed"}
```

---

## 5. Node Operator Upgrade Protocol 📋 SPECIFIED

### Documentation Structure

```
docs/operators/
├── pq_upgrade_guide.md        # Comprehensive upgrade guide
├── compatibility_tests.md     # Compatibility testing procedures
├── drift_radar_install.md     # Drift radar installation guide
├── activation_verification.md # Activation verification tests
└── rollback_rehearsal.md      # Rollback rehearsal checklist
```

### Upgrade Steps (`pq_upgrade_guide.md`)

**Phase 1: Pre-Upgrade Preparation (1 week before)**

1. **Review Release Notes**
   - Read PQ activation whitepaper addendum
   - Review consensus rule changes
   - Understand epoch transition timeline

2. **Backup Current State**
   - Backup blockchain data directory
   - Backup configuration files
   - Document current node version

3. **Test Environment Setup**
   - Deploy testnet node
   - Sync testnet to current block
   - Verify testnet node health

**Phase 2: Software Upgrade (Activation week)**

1. **Download New Version**
   ```bash
   wget https://releases.mathledger.network/pq-activation-v1.0.0.tar.gz
   sha256sum pq-activation-v1.0.0.tar.gz
   # Verify checksum matches official release
   ```

2. **Stop Current Node**
   ```bash
   systemctl stop mathledger-node
   ```

3. **Install New Version**
   ```bash
   tar -xzf pq-activation-v1.0.0.tar.gz
   cd mathledger-pq-v1.0.0
   ./install.sh
   ```

4. **Update Configuration**
   ```bash
   # Add PQ migration settings to config
   echo "pq_migration_enabled=true" >> /etc/mathledger/config.toml
   echo "epoch_registry_path=/var/lib/mathledger/epochs.db" >> /etc/mathledger/config.toml
   ```

5. **Start Upgraded Node**
   ```bash
   systemctl start mathledger-node
   systemctl status mathledger-node
   ```

**Phase 3: Post-Upgrade Verification (24 hours after)**

1. **Verify Node Sync**
   ```bash
   mathledger-cli node status
   # Check that node is synced to current block
   ```

2. **Verify PQ Support**
   ```bash
   mathledger-cli pq status
   # Should show: "PQ migration support: ENABLED"
   ```

3. **Monitor Logs**
   ```bash
   tail -f /var/log/mathledger/node.log
   # Watch for any errors or warnings
   ```

### Compatibility Tests (`compatibility_tests.md`)

**Test 1: Legacy Block Validation**
```bash
# Verify node can validate pre-PQ blocks
mathledger-cli validate-block --block-number 999999
# Expected: VALID
```

**Test 2: Dual-Commitment Block Validation**
```bash
# Verify node can validate dual-commitment blocks
mathledger-cli validate-block --block-number 1000000
# Expected: VALID (with dual commitment)
```

**Test 3: Epoch Resolution**
```bash
# Verify epoch resolution works correctly
mathledger-cli epoch-info --block-number 1000000
# Expected: Shows SHA3-256 epoch info
```

**Test 4: Historical Verification**
```bash
# Verify historical blocks still validate correctly
mathledger-cli verify-chain --start-block 0 --end-block 1000000
# Expected: All blocks VALID
```

### Drift Radar Installation (`drift_radar_install.md`)

**Installation Steps**:

1. **Install Monitoring Agent**
   ```bash
   pip3 install mathledger-drift-radar
   ```

2. **Configure Agent**
   ```bash
   # Create config file
   cat > /etc/mathledger/drift_radar.yaml <<EOF
   monitoring:
     enabled: true
     interval_seconds: 10
   aggregator:
     host: "aggregator.mathledger.network"
     port: 9090
   alerting:
     email:
       enabled: true
       recipients: ["operator@example.com"]
   EOF
   ```

3. **Start Agent**
   ```bash
   systemctl start mathledger-drift-radar
   systemctl enable mathledger-drift-radar
   ```

4. **Verify Agent Running**
   ```bash
   systemctl status mathledger-drift-radar
   curl http://localhost:9091/health
   # Expected: {"status": "healthy"}
   ```

### Activation Verification Tests (`activation_verification.md`)

**Verification Checklist for Activation Block**:

- [ ] Node synced to activation block
- [ ] First PQ block has valid dual commitment
- [ ] Both legacy and PQ Merkle roots valid
- [ ] Prev-hash linkage correct for both chains
- [ ] Drift radar shows no alerts
- [ ] Block propagated to network successfully

**Verification Commands**:
```bash
# Check activation block
mathledger-cli block-info --block-number 1000000

# Verify dual commitment
mathledger-cli verify-dual-commitment --block-number 1000000

# Check drift radar status
curl http://localhost:9091/api/v1/status
```

### Rollback Rehearsal Checklist (`rollback_rehearsal.md`)

**Rehearsal Scenario**: Simulate emergency rollback on testnet

**Steps**:

1. **Deploy Testnet with PQ Activated**
   - [ ] Testnet running with PQ epoch active
   - [ ] At least 100 PQ blocks sealed
   - [ ] All nodes upgraded and healthy

2. **Trigger Emergency Declaration**
   - [ ] Emergency council convenes
   - [ ] 5/7 council members sign emergency declaration
   - [ ] Emergency broadcast to network

3. **Execute Rollback**
   - [ ] Rollback epoch registered
   - [ ] All nodes switch to legacy algorithm
   - [ ] New blocks sealed with SHA-256 only

4. **Verify Rollback Success**
   - [ ] Chain continuity maintained
   - [ ] Historical blocks still verifiable
   - [ ] No consensus failures
   - [ ] Network health restored

5. **Document Lessons Learned**
   - [ ] Rollback execution time
   - [ ] Communication effectiveness
   - [ ] Technical issues encountered
   - [ ] Improvements for next rehearsal

---

## 6. Implementation Timeline

### Phase 1: Consensus Module (COMPLETE) ✅
- **Duration**: 2 weeks
- **Status**: ✅ Complete (1,297 lines implemented)
- **Deliverables**: 6 core modules, runtime plan document

### Phase 2: Benchmark Harness (IN PROGRESS) 🔄
- **Duration**: 3 weeks
- **Status**: 🔄 40% complete (micro + component benchmarks done)
- **Remaining**: Integration benchmarks, network benchmarks, orchestration script
- **Estimated Completion**: 2 weeks

### Phase 3: Drift Radar Runtime (SPECIFIED) 📋
- **Duration**: 3 weeks
- **Status**: 📋 Specified (implementation plan ready)
- **Deliverables**: 10 monitoring modules, dashboard, alerting system
- **Estimated Completion**: 3 weeks

### Phase 4: Governance Engine (SPECIFIED) 📋
- **Duration**: 3 weeks
- **Status**: 📋 Specified (implementation plan ready)
- **Deliverables**: 6 governance modules, RPC API
- **Estimated Completion**: 3 weeks

### Phase 5: Operator Documentation (SPECIFIED) 📋
- **Duration**: 1 week
- **Status**: 📋 Specified (documentation plan ready)
- **Deliverables**: 5 operator guides
- **Estimated Completion**: 1 week

### Total Timeline: 12 weeks (3 months)

---

## 7. Deployment Roadmap

### Testnet Deployment (Month 1)
- Deploy consensus module to testnet
- Run benchmark harness on testnet
- Deploy drift radar monitoring
- Simulate activation and rollback

### Mainnet Preparation (Month 2)
- Security audit of all modules
- Community review of governance process
- Node operator upgrade campaign
- Final testing and validation

### Mainnet Activation (Month 3)
- Governance proposal submission
- Community voting period
- Activation at approved block
- Grace period monitoring

---

## 8. Success Criteria

### Consensus Module
- [ ] 100% unit test pass rate
- [ ] 100% integration test pass rate
- [ ] Security audit complete with no critical issues
- [ ] 1000+ blocks validated on testnet

### Benchmark Harness
- [ ] All acceptance criteria met
- [ ] Block sealing overhead <300%
- [ ] Validation latency <100ms
- [ ] Storage overhead <30%

### Drift Radar
- [ ] All drift categories detected in testing
- [ ] Alert routing functional
- [ ] Dashboard operational
- [ ] Zero false negatives in drift injection tests

### Governance Engine
- [ ] Proposal lifecycle functional
- [ ] Voting mechanism operational
- [ ] Attestation system functional
- [ ] Rollback procedure tested

### Operator Documentation
- [ ] All guides complete
- [ ] Compatibility tests pass
- [ ] Drift radar installation successful
- [ ] Rollback rehearsal successful

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-06  
**Author**: Manus-H  
**Status**: Master Implementation Plan
