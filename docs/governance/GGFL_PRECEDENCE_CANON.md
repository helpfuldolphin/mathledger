# GGFL Canon: Identity Precedence Over Risk

This document formalizes the principle that cryptographic identity failures, characterized by a `HARD_BLOCK` signal, consistently override any risk assessment outputs within the Global Governance Fusion Layer. This invariant prioritizes foundational system integrity over probabilistic risk metrics, a design choice validated by automated integration tests documented in `tests/test_ggfl_fusion_integration.py`.

***In short, a definitive cryptographic integrity failure will always halt the system, regardless of any statistical risk assessment suggesting otherwise.***

A foundational principle of the Global Governance Fusion Layer (GGFL) is the non-negotiable precedence of cryptographic verification over statistical risk assessment. This document codifies the "why" behind this critical safety invariant.

## The Invariant

**An `identity` signal issuing a `HARD_BLOCK` will always result in a final `HARD_BLOCK` decision, regardless of the state of the `risk` signal or any other lower-precedence signal.**

This is not merely a configurable setting; it is a doctrinal aspect of the system's architecture.

## Rationale

The `identity` signal (SIG-IDN) represents the system's cryptographic ground truth. Its checks include:

*   Block hash validation
*   Chain continuity
*   Dual-root attestation consistency
*   Signature validity

A failure in any of these areas indicates a potential compromise of the system's fundamental integrity or a security-critical event. The signal is deterministic and binary: the chain is either valid, or it is not.

The `risk` signal (SIG-RSK), in contrast, is the output of a statistical model. It provides a probabilistic assessment of future system behavior based on a collection of weighted metrics. While it is a vital tool for forward-looking governance, it operates on a foundation of uncertainty.

Therefore, allowing a probabilistic warning (`risk`) to override a deterministic failure of ground truth (`identity`) would violate the core safety premise of the entire system. The GGFL is designed to trust cryptographic proof over statistical prediction, without exception.

## Proof of Enforcement

This precedence invariant is explicitly and automatically enforced by our continuous integration tests. The following test file contains the "reality lock" for this rule:

*   [`tests/test_ggfl_fusion_integration.py`](../tests/test_ggfl_fusion_integration.py)

The `test_precedence_invariant_with_hard_block` case specifically asserts that an `identity` hard block is final, even in the presence of a nominal `risk` signal.
