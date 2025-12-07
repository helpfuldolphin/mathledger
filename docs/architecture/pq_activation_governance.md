# Post-Quantum Activation Governance Process

## Document Status

**Version**: 1.0  
**Status**: Governance Specification  
**Author**: Manus-H  
**Date**: 2024-12-06

## Executive Summary

This document specifies the governance process for activating post-quantum (PQ) hash algorithms in MathLedger. The process ensures that the migration is community-driven, transparent, and safe. It defines proposal mechanisms, voting procedures, grace period semantics, emergency rollback protocols, community verification steps, and multi-party attestation schemes.

## Governance Principles

### Principle 1: Community Consensus

All major migration decisions require broad community consensus. No single entity can unilaterally activate or rollback PQ algorithms.

### Principle 2: Transparency

All proposals, votes, and decisions are publicly visible and cryptographically verifiable. The governance process itself is recorded on-chain.

### Principle 3: Safety First

The migration prioritizes network safety over speed. Sufficient testing, verification, and grace periods are mandatory.

### Principle 4: Reversibility

Emergency rollback mechanisms exist but require even higher consensus thresholds than activation.

### Principle 5: Inclusivity

All stakeholders (node operators, developers, users) have voice in the governance process.

## Governance Roles

### Role 1: Proposers

**Who**: Any community member with minimum stake or reputation  
**Responsibilities**: Submit well-formed activation proposals  
**Requirements**: Technical competence, stake threshold, or endorsements

### Role 2: Reviewers

**Who**: Technical committee, security auditors, core developers  
**Responsibilities**: Review proposals for technical soundness and security  
**Requirements**: Demonstrated technical expertise

### Role 3: Voters

**Who**: Stakeholders (token holders, node operators, validators)  
**Responsibilities**: Vote on proposals  
**Requirements**: Minimum stake or validator status

### Role 4: Attestors

**Who**: Independent node operators and validators  
**Responsibilities**: Attest to successful migration milestones  
**Requirements**: Running production nodes, good reputation

### Role 5: Emergency Council

**Who**: Elected representatives with emergency powers  
**Responsibilities**: Execute emergency rollbacks if needed  
**Requirements**: Supermajority election, time-limited mandate

## Activation Proposal Process

### Phase 1: Pre-Proposal (2-4 weeks)

**Objective**: Gather community feedback before formal proposal.

**Steps**:

1. **Draft Proposal**: Proposer creates draft activation proposal
2. **Community Discussion**: Post draft to governance forum for feedback
3. **Technical Review**: Request informal review from technical committee
4. **Revision**: Incorporate feedback and revise draft
5. **Endorsements**: Gather endorsements from community members

**Deliverables**:
- Draft proposal document
- Community feedback summary
- Endorsement signatures

### Phase 2: Formal Proposal (1 week)

**Objective**: Submit formal on-chain proposal.

**Proposal Structure**:

```json
{
  "proposal_id": "PQ-ACTIVATION-001",
  "proposal_type": "epoch_activation",
  "title": "Activate SHA3-256 Post-Quantum Hash Algorithm",
  "proposer": "0x1234...5678",
  "submission_timestamp": 1701900000,
  "activation_block": 1000000,
  "epoch_parameters": {
    "algorithm_id": 1,
    "algorithm_name": "SHA3-256",
    "start_block": 1000000,
    "end_block": null,
    "rule_version": "v2-dual-required",
    "grace_period_blocks": 100000
  },
  "rationale": "SHA3-256 provides quantum resistance with acceptable performance overhead...",
  "benchmark_results": {
    "sealing_overhead": "250%",
    "validation_latency": "75ms",
    "storage_overhead": "25%"
  },
  "risk_assessment": {
    "technical_risks": ["Performance degradation", "Implementation bugs"],
    "mitigation": ["Extensive testing", "Gradual rollout", "Monitoring"]
  },
  "implementation_readiness": {
    "code_complete": true,
    "tests_passing": true,
    "documentation_complete": true,
    "node_compatibility": "95%"
  },
  "endorsements": [
    {"address": "0xabcd...ef01", "signature": "0x9876...5432"},
    {"address": "0x2345...6789", "signature": "0x8765...4321"}
  ]
}
```

**Submission Requirements**:
- Minimum stake: 10,000 tokens (or 10 endorsements from validators)
- Proposal fee: 100 tokens (refunded if proposal passes)
- Complete proposal structure with all required fields
- Cryptographic signature from proposer

**On-Chain Recording**:
```python
def submit_activation_proposal(proposal: ActivationProposal) -> str:
    """
    Submit activation proposal on-chain.
    
    Returns proposal hash for tracking.
    """
    # Validate proposal structure
    validate_proposal_structure(proposal)
    
    # Check proposer eligibility
    if not is_eligible_proposer(proposal.proposer):
        raise ValueError("Proposer does not meet eligibility requirements")
    
    # Record proposal on-chain
    proposal_hash = hash_proposal(proposal)
    proposal_block = seal_governance_block(
        statements=[f"PROPOSAL:{proposal_hash}:{json.dumps(proposal)}"],
        prev_hash=get_latest_block_hash(),
        block_number=get_latest_block_number() + 1,
        timestamp=time.time(),
    )
    
    # Emit proposal event
    emit_event("ProposalSubmitted", {
        "proposal_id": proposal.proposal_id,
        "proposal_hash": proposal_hash,
        "proposer": proposal.proposer,
        "activation_block": proposal.activation_block,
    })
    
    return proposal_hash
```

### Phase 3: Technical Review (2 weeks)

**Objective**: Conduct thorough technical and security review.

**Review Criteria**:

1. **Code Completeness**: All implementation modules complete and tested
2. **Test Coverage**: Comprehensive test suite with >90% coverage
3. **Benchmark Results**: Performance targets met
4. **Security Audit**: Independent security audit completed
5. **Documentation**: Complete documentation for operators and developers
6. **Compatibility**: Backward compatibility maintained
7. **Rollback Plan**: Emergency rollback procedure documented

**Review Process**:

```python
def conduct_technical_review(proposal_hash: str) -> TechnicalReviewReport:
    """
    Conduct technical review of activation proposal.
    """
    proposal = get_proposal(proposal_hash)
    
    # Code review
    code_review = review_implementation_code(proposal.epoch_parameters.algorithm_id)
    
    # Test review
    test_review = review_test_suite(proposal.epoch_parameters.algorithm_id)
    
    # Benchmark review
    benchmark_review = review_benchmark_results(proposal.benchmark_results)
    
    # Security review
    security_review = conduct_security_audit(proposal.epoch_parameters.algorithm_id)
    
    # Documentation review
    docs_review = review_documentation(proposal.epoch_parameters.algorithm_id)
    
    # Generate report
    report = TechnicalReviewReport(
        proposal_hash=proposal_hash,
        code_review=code_review,
        test_review=test_review,
        benchmark_review=benchmark_review,
        security_review=security_review,
        docs_review=docs_review,
        recommendation="APPROVE" if all_checks_pass() else "REJECT",
        reviewer_signatures=[...],
    )
    
    # Record review on-chain
    record_review_report(report)
    
    return report
```

**Review Outcomes**:
- **APPROVE**: Proposal proceeds to voting
- **REJECT**: Proposal rejected, proposer can revise and resubmit
- **CONDITIONAL**: Proposal approved pending specific changes

### Phase 4: Community Voting (2 weeks)

**Objective**: Achieve community consensus on activation.

**Voting Mechanism**:

**Voting Power**:
- Token holders: 1 vote per token
- Validators: Weighted by stake
- Node operators: 1 vote per verified node

**Voting Options**:
- **YES**: Support activation at proposed block
- **NO**: Reject activation
- **ABSTAIN**: No position

**Quorum Requirements**:
- Minimum participation: 40% of total voting power
- Approval threshold: 66.7% (supermajority) of votes cast

**Voting Implementation**:

```python
def cast_vote(proposal_hash: str, voter: str, vote: str, signature: str) -> None:
    """
    Cast vote on activation proposal.
    """
    # Validate voter eligibility
    voting_power = get_voting_power(voter)
    if voting_power == 0:
        raise ValueError("Voter has no voting power")
    
    # Validate signature
    if not verify_signature(voter, f"{proposal_hash}:{vote}", signature):
        raise ValueError("Invalid signature")
    
    # Record vote on-chain
    vote_block = seal_governance_block(
        statements=[f"VOTE:{proposal_hash}:{voter}:{vote}:{voting_power}"],
        prev_hash=get_latest_block_hash(),
        block_number=get_latest_block_number() + 1,
        timestamp=time.time(),
    )
    
    # Emit vote event
    emit_event("VoteCast", {
        "proposal_hash": proposal_hash,
        "voter": voter,
        "vote": vote,
        "voting_power": voting_power,
    })
```

**Vote Tallying**:

```python
def tally_votes(proposal_hash: str) -> VoteTallyResult:
    """
    Tally votes for activation proposal.
    """
    # Retrieve all votes
    votes = get_votes_for_proposal(proposal_hash)
    
    # Calculate totals
    yes_votes = sum(v.voting_power for v in votes if v.vote == "YES")
    no_votes = sum(v.voting_power for v in votes if v.vote == "NO")
    abstain_votes = sum(v.voting_power for v in votes if v.vote == "ABSTAIN")
    total_votes = yes_votes + no_votes + abstain_votes
    
    # Calculate participation
    total_voting_power = get_total_voting_power()
    participation_rate = total_votes / total_voting_power
    
    # Calculate approval rate (excluding abstentions)
    approval_rate = yes_votes / (yes_votes + no_votes) if (yes_votes + no_votes) > 0 else 0
    
    # Determine outcome
    quorum_met = participation_rate >= 0.40
    approval_met = approval_rate >= 0.667
    
    outcome = "APPROVED" if (quorum_met and approval_met) else "REJECTED"
    
    result = VoteTallyResult(
        proposal_hash=proposal_hash,
        yes_votes=yes_votes,
        no_votes=no_votes,
        abstain_votes=abstain_votes,
        total_votes=total_votes,
        participation_rate=participation_rate,
        approval_rate=approval_rate,
        outcome=outcome,
    )
    
    # Record result on-chain
    record_vote_result(result)
    
    return result
```

### Phase 5: Activation Preparation (4-8 weeks)

**Objective**: Prepare network for activation.

**Preparation Steps**:

1. **Node Upgrade Campaign**: Coordinate node operator upgrades
2. **Monitoring Setup**: Deploy monitoring infrastructure
3. **Communication**: Notify all stakeholders of activation timeline
4. **Final Testing**: Conduct final integration tests on testnet
5. **Attestation Setup**: Prepare multi-party attestation infrastructure

**Node Upgrade Tracking**:

```python
def track_node_upgrades(activation_block: int) -> NodeUpgradeStatus:
    """
    Track node upgrade status across network.
    """
    # Query all known nodes
    nodes = get_all_nodes()
    
    # Check upgrade status
    upgraded_nodes = []
    non_upgraded_nodes = []
    
    for node in nodes:
        version = get_node_version(node)
        if supports_pq_migration(version):
            upgraded_nodes.append(node)
        else:
            non_upgraded_nodes.append(node)
    
    # Calculate readiness
    upgrade_rate = len(upgraded_nodes) / len(nodes)
    
    status = NodeUpgradeStatus(
        total_nodes=len(nodes),
        upgraded_nodes=len(upgraded_nodes),
        non_upgraded_nodes=len(non_upgraded_nodes),
        upgrade_rate=upgrade_rate,
        activation_block=activation_block,
        blocks_remaining=activation_block - get_current_block_number(),
    )
    
    return status
```

**Readiness Criteria**:
- Node upgrade rate: >90%
- Testnet validation: 1000+ blocks sealed successfully
- Monitoring infrastructure: Operational
- Emergency procedures: Documented and rehearsed

### Phase 6: Activation (Block N)

**Objective**: Activate PQ algorithm at designated block.

**Activation Procedure**:

```python
def activate_epoch(epoch: HashEpoch) -> None:
    """
    Activate new hash epoch at designated block.
    """
    current_block = get_current_block_number()
    
    # Verify activation block reached
    if current_block < epoch.start_block:
        raise ValueError(f"Activation block {epoch.start_block} not yet reached")
    
    # Verify governance approval
    proposal = get_approved_proposal_for_epoch(epoch)
    if not proposal:
        raise ValueError("No approved proposal for this epoch")
    
    # Register epoch
    register_epoch(epoch)
    
    # Emit activation event
    emit_event("EpochActivated", {
        "epoch_id": epoch.algorithm_id,
        "algorithm_name": epoch.algorithm_name,
        "start_block": epoch.start_block,
        "activation_timestamp": time.time(),
    })
    
    # Log activation
    log_info(f"Epoch {epoch.algorithm_name} activated at block {epoch.start_block}")
```

**First PQ Block Requirements**:
- Must be block number equal to `epoch.start_block`
- Must contain valid dual commitment
- Must use new PQ algorithm
- Must correctly link to previous block

**Activation Monitoring**:
- Real-time monitoring of first PQ block
- Automatic alerts if validation fails
- Multi-party attestation of successful activation

## Grace Period Semantics

### Purpose

The grace period allows the network to operate with dual commitments, providing time for:
- Node operators to complete upgrades
- Developers to fix any discovered issues
- Community to verify migration success
- Ecosystem tools to adapt to PQ hashes

### Duration

**Recommended Duration**: 100,000 blocks (~6-12 months depending on block time)

**Rationale**:
- Sufficient time for ecosystem adaptation
- Long enough to detect and fix issues
- Short enough to avoid indefinite dual overhead

### Grace Period Rules

**During Grace Period**:
1. All blocks MUST contain dual commitments
2. Both legacy and PQ hashes MUST be valid
3. Legacy hashes remain CANONICAL for consensus
4. PQ hashes are VALIDATED but not canonical
5. Node operators MAY use either hash for verification

**Grace Period Monitoring**:

```python
def monitor_grace_period(epoch: HashEpoch) -> GracePeriodStatus:
    """
    Monitor grace period progress and health.
    """
    current_block = get_current_block_number()
    grace_start = epoch.start_block
    grace_end = grace_start + epoch.grace_period_blocks
    
    # Calculate progress
    blocks_elapsed = current_block - grace_start
    blocks_remaining = grace_end - current_block
    progress_percent = (blocks_elapsed / epoch.grace_period_blocks) * 100
    
    # Check health metrics
    recent_blocks = get_recent_blocks(count=100)
    dual_commitment_rate = sum(1 for b in recent_blocks if b.header.has_dual_commitment()) / len(recent_blocks)
    validation_success_rate = sum(1 for b in recent_blocks if validate_dual_commitment(b)) / len(recent_blocks)
    
    status = GracePeriodStatus(
        grace_start=grace_start,
        grace_end=grace_end,
        current_block=current_block,
        blocks_elapsed=blocks_elapsed,
        blocks_remaining=blocks_remaining,
        progress_percent=progress_percent,
        dual_commitment_rate=dual_commitment_rate,
        validation_success_rate=validation_success_rate,
        health="HEALTHY" if validation_success_rate > 0.99 else "DEGRADED",
    )
    
    return status
```

### Grace Period Extension

If issues are discovered during the grace period, the community can vote to extend it.

**Extension Proposal**:
- Requires same governance process as activation
- Must include rationale for extension
- Must specify new grace period end block

**Extension Criteria**:
- Critical bugs discovered
- Insufficient node upgrade rate
- Performance issues requiring optimization
- Ecosystem tools not ready

## Emergency Rollback Protocol

### Rollback Triggers

**Automatic Triggers** (immediate rollback without vote):
1. **Consensus Failure**: Network cannot reach consensus on PQ blocks
2. **Critical Security Vulnerability**: Severe vulnerability discovered in PQ algorithm
3. **Network Partition**: Network splits due to PQ migration

**Manual Triggers** (require emergency council vote):
1. **Performance Degradation**: Unacceptable performance impact
2. **Implementation Bugs**: Critical bugs in PQ implementation
3. **Ecosystem Disruption**: Major ecosystem tools broken

### Rollback Authority

**Emergency Council**:
- 7 elected members
- 5/7 supermajority required for rollback
- Time-limited mandate (6 months)
- Subject to community oversight

**Emergency Council Election**:
- Held during activation preparation phase
- Token-weighted voting
- Candidates must be experienced node operators or developers

### Rollback Procedure

**Phase 1: Emergency Declaration (1 hour)**

```python
def declare_emergency(trigger: str, evidence: str, council_signatures: List[str]) -> str:
    """
    Declare migration emergency.
    
    Requires 5/7 emergency council signatures.
    """
    # Validate signatures
    if len(council_signatures) < 5:
        raise ValueError("Insufficient council signatures")
    
    for sig in council_signatures:
        if not verify_council_signature(sig):
            raise ValueError("Invalid council signature")
    
    # Record emergency declaration
    emergency_hash = hash_emergency_declaration(trigger, evidence)
    emergency_block = seal_governance_block(
        statements=[f"EMERGENCY:{emergency_hash}:{trigger}:{evidence}"],
        prev_hash=get_latest_block_hash(),
        block_number=get_latest_block_number() + 1,
        timestamp=time.time(),
    )
    
    # Emit emergency event
    emit_event("EmergencyDeclared", {
        "emergency_hash": emergency_hash,
        "trigger": trigger,
        "timestamp": time.time(),
    })
    
    # Broadcast to all nodes
    broadcast_emergency_alert(emergency_hash, trigger)
    
    return emergency_hash
```

**Phase 2: Rollback Execution (24 hours)**

```python
def execute_rollback(emergency_hash: str, rollback_block: int) -> None:
    """
    Execute emergency rollback to legacy algorithm.
    """
    # Verify emergency declaration
    emergency = get_emergency_declaration(emergency_hash)
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
    
    # Emit rollback event
    emit_event("RollbackExecuted", {
        "emergency_hash": emergency_hash,
        "rollback_block": rollback_block,
        "rollback_algorithm": "SHA-256",
        "timestamp": time.time(),
    })
    
    # Broadcast to all nodes
    broadcast_rollback_notice(rollback_block)
```

**Phase 3: Post-Rollback Analysis (1 week)**

After rollback, conduct thorough analysis:
1. Root cause analysis of failure
2. Fix implementation issues
3. Additional testing
4. Revised activation proposal

### Rollback Constraints

**Temporal Constraints**:
- Rollback must occur within grace period
- Cannot rollback after grace period ends (too late)
- Rollback must have at least 1000 block notice

**Safety Constraints**:
- Cannot rollback past finalized blocks
- Must maintain chain continuity
- Historical blocks remain verifiable

## Community Verification Steps

### Verification Responsibilities

All stakeholders participate in verification:

**Node Operators**:
- Verify blocks seal correctly with PQ hashes
- Monitor performance metrics
- Report anomalies

**Developers**:
- Review implementation code
- Run test suites
- Verify benchmark results

**Validators**:
- Validate blocks with both algorithms
- Attest to successful validation
- Participate in multi-party attestation

**Users**:
- Monitor network health dashboards
- Verify transaction inclusion
- Report issues

### Verification Checkpoints

**Checkpoint 1: First PQ Block (Block N)**
- Verify dual commitment valid
- Verify both Merkle roots valid
- Verify prev_hash linkage correct
- Multi-party attestation required

**Checkpoint 2: 100 Blocks After Activation (Block N+100)**
- Verify 100 consecutive valid PQ blocks
- Verify no consensus failures
- Verify performance within targets
- Community attestation required

**Checkpoint 3: 1000 Blocks After Activation (Block N+1000)**
- Verify sustained operation
- Verify historical verification works
- Verify ecosystem tools adapted
- Validator attestation required

**Checkpoint 4: Grace Period Midpoint**
- Verify grace period health
- Assess readiness for PQ canonicalization
- Community feedback collection

**Checkpoint 5: Grace Period End**
- Verify readiness to deprecate legacy fields
- Final community attestation
- Prepare for Phase 4 transition

### Verification Tools

**Dashboard**: Real-time network health monitoring  
**CLI Tools**: Command-line verification tools for operators  
**APIs**: Programmatic access to verification data  
**Reports**: Automated verification reports

## Multi-Party Attestation Scheme

### Purpose

Multi-party attestation provides cryptographic proof that multiple independent parties have verified migration milestones.

### Attestation Structure

```json
{
  "attestation_id": "ATTEST-001-FIRST-PQ-BLOCK",
  "milestone": "first_pq_block",
  "block_number": 1000000,
  "block_hash": "0xabcd...ef01",
  "pq_merkle_root": "0x1234...5678",
  "timestamp": 1701900000,
  "attestors": [
    {
      "node_id": "node-001",
      "operator": "Alice",
      "signature": "0x9876...5432",
      "verification_method": "full_validation"
    },
    {
      "node_id": "node-002",
      "operator": "Bob",
      "signature": "0x8765...4321",
      "verification_method": "merkle_proof"
    }
  ],
  "threshold": "5/7",
  "status": "VERIFIED"
}
```

### Attestation Process

**Step 1: Milestone Reached**
- Network reaches verification milestone (e.g., first PQ block)
- Attestation coordinator broadcasts attestation request

**Step 2: Independent Verification**
- Each attestor independently verifies the milestone
- Attestors run verification scripts on their own nodes
- Verification includes all consensus rules

**Step 3: Signature Collection**

```python
def sign_attestation(milestone: str, block_number: int, node_id: str, private_key: str) -> str:
    """
    Sign attestation for migration milestone.
    """
    # Retrieve block
    block = get_block(block_number)
    
    # Verify block
    if not validate_block_full(block, get_block(block_number - 1)):
        raise ValueError("Block validation failed")
    
    # Create attestation message
    message = f"{milestone}:{block_number}:{block.header.merkle_root}:{block.header.pq_merkle_root}"
    
    # Sign message
    signature = sign_message(message, private_key)
    
    # Submit attestation
    submit_attestation(milestone, block_number, node_id, signature)
    
    return signature
```

**Step 4: Threshold Verification**

```python
def verify_attestation_threshold(attestation_id: str, threshold: str) -> bool:
    """
    Verify attestation meets threshold requirement.
    
    Args:
        attestation_id: Attestation identifier
        threshold: Required threshold (e.g., "5/7")
        
    Returns:
        True if threshold met, False otherwise
    """
    attestation = get_attestation(attestation_id)
    
    # Parse threshold
    required, total = map(int, threshold.split('/'))
    
    # Verify signatures
    valid_signatures = 0
    for attestor in attestation.attestors:
        if verify_attestor_signature(attestor):
            valid_signatures += 1
    
    # Check threshold
    threshold_met = valid_signatures >= required
    
    # Record result
    attestation.status = "VERIFIED" if threshold_met else "INSUFFICIENT"
    record_attestation_result(attestation)
    
    return threshold_met
```

**Step 5: On-Chain Recording**

```python
def record_attestation_on_chain(attestation: Attestation) -> None:
    """
    Record attestation on-chain for permanent verification.
    """
    attestation_hash = hash_attestation(attestation)
    
    attestation_block = seal_governance_block(
        statements=[f"ATTESTATION:{attestation_hash}:{json.dumps(attestation)}"],
        prev_hash=get_latest_block_hash(),
        block_number=get_latest_block_number() + 1,
        timestamp=time.time(),
    )
    
    emit_event("AttestationRecorded", {
        "attestation_id": attestation.attestation_id,
        "attestation_hash": attestation_hash,
        "milestone": attestation.milestone,
        "status": attestation.status,
    })
```

### Attestation Milestones

| Milestone | Threshold | Description |
|-----------|-----------|-------------|
| **First PQ Block** | 5/7 | First block with PQ hash validates correctly |
| **100 Block Stability** | 5/7 | 100 consecutive PQ blocks validate correctly |
| **1000 Block Stability** | 5/7 | 1000 consecutive PQ blocks validate correctly |
| **Grace Period Midpoint** | 5/7 | Grace period health check |
| **Grace Period End** | 5/7 | Ready for legacy deprecation |

### Attestor Selection

**Criteria**:
- Geographic diversity (different regions)
- Organizational diversity (different entities)
- Technical diversity (different implementations)
- Reputation (proven track record)

**Selection Process**:
- Community nominates candidates
- Token-weighted vote
- 7 attestors selected
- 6-month term

## Governance Timeline Example

### Realistic Timeline for SHA3-256 Activation

**Month 0: Pre-Proposal**
- Week 1-2: Draft proposal, gather feedback
- Week 3-4: Technical review, revisions

**Month 1: Formal Proposal**
- Week 1: Submit on-chain proposal
- Week 2-3: Technical committee review
- Week 4: Review report published

**Month 2: Voting**
- Week 1-2: Community voting period
- Week 3: Vote tallying and result announcement
- Week 4: Activation preparation begins

**Month 3-4: Preparation**
- Node upgrade campaign
- Monitoring infrastructure deployment
- Final testing on testnet
- Emergency council election
- Attestor selection

**Month 5: Activation**
- Block 1,000,000: SHA3-256 epoch activates
- First PQ block sealed and attested
- Grace period begins (100,000 blocks)

**Month 6-16: Grace Period**
- Dual commitments maintained
- Continuous monitoring
- Ecosystem adaptation
- Verification checkpoints

**Month 17: Grace Period End**
- Final attestation
- Prepare for legacy deprecation

**Month 18: Legacy Deprecation**
- Legacy fields become optional
- PQ hashes fully canonical
- Migration complete

## Governance Security

### Attack Vectors

**Attack 1: Vote Buying**
- Attacker buys votes to force activation
- Mitigation: High approval threshold (66.7%), reputation-weighted voting

**Attack 2: Sybil Attack**
- Attacker creates fake nodes to inflate attestations
- Mitigation: Stake-weighted attestation, reputation requirements

**Attack 3: Emergency Council Capture**
- Attacker compromises emergency council
- Mitigation: Supermajority requirement (5/7), time-limited mandate, community oversight

**Attack 4: Rollback Abuse**
- Malicious rollback to disrupt network
- Mitigation: High threshold for rollback, evidence requirement, post-rollback analysis

### Security Measures

1. **Cryptographic Verification**: All votes and attestations cryptographically signed
2. **On-Chain Recording**: All governance actions recorded on-chain
3. **Transparency**: All proposals and votes publicly visible
4. **Auditability**: Complete audit trail of governance process
5. **Time Locks**: Sufficient notice periods for all major changes

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-06  
**Author**: Manus-H (Quantum-Migration Engineer)  
**Status**: Governance Specification - Pending Community Approval
