# Appendix: Curriculum Stability Requirements for P3/P4 Validation

**Document ID**: `CUR-STABILITY-APPENDIX-V1`  
**Author**: MANUS-E, Curriculum Integrity Engineer  
**Status**: DRAFT  
**Date**: 2025-12-10

---

## 1. Preamble: The Scientific Imperative for a Stable Curriculum

The entire premise of the MathLedger RFL (Reflexive Formal Learning) system rests on the principles of controlled experimentation. We aim to measure the impact of changes to our learning algorithms, proof strategies, and teacher models. To do this, we must treat the **curriculum as a controlled variable**. An unstable or drifting curriculum introduces confounding variables that make it impossible to isolate the true effect of our interventions, rendering our experimental results scientifically invalid.

This appendix outlines why **identity-stable curriculum**—a curriculum that is bit-for-bit identical and verifiable via a deterministic fingerprint—is a non-negotiable requirement for all P3 (Performance, Progress, and Proficiency) and P4 (Policy, Planning, and Prediction) validation runs.

---

## 2. Why Curriculum Drift Invalidates Synthetic P3 Runs

P3 runs are designed to measure the performance of a specific, versioned agent against a static set of problems defined by the curriculum. The goal is to answer the question: "How capable is Agent X at solving problems of type Y?"

Curriculum drift fundamentally breaks this model in several ways:

### 2.1. Altered Problem Space

A change in curriculum parameters (e.g., `atoms`, `depth_max`, `predicate_arity_max`) directly alters the problem space. 

- **Example**: Increasing `atoms` from 5 to 6 in a slice exponentially increases the complexity of the logical formulas being generated. An agent that achieved 90% coverage on the `atoms=5` slice may only achieve 60% on the `atoms=6` slice. This is not a regression in the agent; it is a change in the task. Without a stable curriculum, we cannot distinguish between an agent getting worse and the problems getting harder.

### 2.2. Shifting Goalposts

A change in gate thresholds (e.g., `ci_lower_min`, `min_pph`) is equivalent to changing the definition of success. 

- **Example**: Lowering the `ci_lower_min` for the coverage gate from 0.85 to 0.80 makes it easier for a slice to be considered "passed." An agent might appear to be making faster progress through the curriculum, but this is an illusion created by lowering the bar. This compromises all P3 metrics related to velocity and proficiency.

### 2.3. Invalidation of Baselines

P3 metrics are only meaningful when compared to a baseline. We measure progress by comparing the performance of a new agent against a previous one on the *same* curriculum. 

- **Scenario**: Agent A achieves a proof velocity of 100 PPH on curriculum `C1`. We then modify the agent to create Agent B. In the process, we accidentally modify the curriculum to `C2` (e.g., by changing a parameter). Agent B achieves 120 PPH on `C2`.
- **Invalid Conclusion**: We cannot claim that Agent B is 20% faster. The underlying task has changed. The 20 PPH uplift could be due to the agent improvement, the curriculum change, or a combination of both. The experiment is confounded and the results are meaningless.

> **Governance Rule**: All P3 runs claiming to measure agent performance over time MUST be conducted against a curriculum with an identical, unchanging fingerprint. Any change in the curriculum fingerprint requires the establishment of a new baseline.

---

## 3. Why Curriculum Drift Compromises P4 Comparisons

P4 runs involve comparing two or more different policies or agents (e.g., A/B testing) to determine which one is superior. The fundamental assumption of any A/B test is that both groups are subjected to the exact same conditions, with the only difference being the intervention being tested.

In our context, the curriculum *is* the condition. It defines the environment in which the agents operate.

### 3.1. Violation of the Ceteris Paribus Principle

*Ceteris paribus* ("all other things being equal") is the bedrock of A/B testing. If we are comparing Policy A and Policy B, we must ensure they are both run against the exact same curriculum.

- **Scenario**: We run Policy A against curriculum `C` (fingerprint `F1`). We then run Policy B, but an accidental change is made, and it runs against curriculum `C'` (fingerprint `F2`).
- **Invalid Comparison**: Even if Policy B outperforms Policy A, we cannot attribute this to the policy itself. The environment changed. It is impossible to disentangle the effect of the policy from the effect of the curriculum.

### 3.2. Introduction of Systematic Bias

If the curriculum changes between the A and B runs, we introduce a systematic bias that favors one group over the other.

- **Example**: Suppose we are testing a new proof search strategy (Policy B) against the current production strategy (Policy A). Policy A is run on the standard curriculum. Before running Policy B, a developer "helpfully" simplifies a difficult slice in the curriculum, believing it to be a minor fix. Policy B now runs on an easier curriculum and shows a 15% uplift in proof success rate.
- **False Positive**: This 15% uplift is not real. It is an artifact of the easier curriculum. If we were to deploy Policy B based on this result, we would likely see no improvement in production, or even a regression. The experiment has produced a dangerous false positive.

> **Governance Rule**: Any P4 run that involves a comparison between two or more agents or policies MUST ensure that all agents in the comparison were run against curricula with identical fingerprints. If the fingerprints do not match, the results of the comparison are invalid and MUST be discarded.

---

## 4. The Requirement for Identity-Stable Curriculum

To address these issues, we enforce the concept of an **identity-stable curriculum**. This is a curriculum that is not just conceptually similar, but bit-for-bit identical, verifiable through a cryptographic hash.

### 4.1. What is an Identity-Stable Curriculum?

An identity-stable curriculum is defined by its **fingerprint**. The `CurriculumSystem.fingerprint()` method produces a SHA-256 hash of the curriculum's functional content, including:
- All slice names and parameters
- All gate thresholds and specifications
- All system-level invariants

This fingerprint is deterministic and stable. Two curriculum files with the same functional content will always produce the same fingerprint, regardless of formatting, comments, or slice ordering.

### 4.2. How It Enables Valid Uplift Claims

By enforcing curriculum stability, we can make valid claims about agent improvement.

- **Valid P3 Claim**: "Agent B shows a 10% improvement in proof velocity over Agent A" is a valid claim **only if** both agents were run on a curriculum with fingerprint `F1`.

- **Valid P4 Claim**: "Policy B provides a 5% uplift in coverage over Policy A with 95% confidence" is a valid claim **only if** the A/B test was conducted with both policies running against a curriculum with fingerprint `F1`.

**The curriculum fingerprint is the formal link that makes these claims scientifically defensible.** Without it, any claims of uplift are anecdotal at best and dangerously misleading at worst.

### 4.3. Enforcement

This requirement is not a guideline; it is an enforced property of the system.

1.  The `CurriculumDriftSentinel` **BLOCKS** any run where the curriculum fingerprint changes unexpectedly.
2.  The `RunLedgerEntry` for every run is permanently stamped with the curriculum fingerprint, creating an immutable audit trail.

This ensures that every result in our system can be traced back to the exact experimental conditions under which it was generated, preserving the integrity of our research and the trustworthiness of our results.
