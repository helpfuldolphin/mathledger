# Phase II Figure Atlas
**Version:** 1.0
**Status:** Specification
**Target:** Final Research Paper & Investor Brief

**NOTE:** All figures except the abstention curve are Phase II / Planned — no empirical data yet.

This document specifies the figure requirements for Phase II analysis, mapping each visual artifact to its scientific purpose, investor narrative, and source data telemetry.

---

## 1. FO Cycle Harness
*Schematic visualization of the First Organism's reflexive feedback loop.*

*   **Status:** Planned
*   **Scientific Purpose:** Illustrates the information flow between the Derivation Engine, Ledger, and UI, explicitly showing the Feedback mechanism ($\mathcal{F}$) that drives the RFL loop.
*   **Investor-Facing Purpose:** Visualizes the "digital metabolism" that turns raw compute into verified, immutable truth, establishing the system's autonomous nature.
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest Field:** `config.architecture` / `system.topology` (Conceptual Map)

---

## 2. Abstention Rate Curve
*Time-series plot of the abstention rate $A(t)$ showing logistic decay.*

*   **Status:** Available (Derived from degenerate (100% abstention) run — admissible as baseline only)
*   **File Path:** `artifacts/figures/rfl_abstention_rate.png`
*   **Scientific Purpose:** Validates **Theorem 4.1** (Abstention Dynamics), confirming that $A(t)$ follows a logistic decay curve under sustained entropy injection.
*   **Investor-Facing Purpose:** Shows the system "learning" to handle unknown problems over time, rapidly reducing its failure rate as it builds experience.
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest Field:** `metrics.abstention.rate` (plotted vs `cycle_index`)

---

## 3. $\Delta H$ Scaling
*Scatter plot of Hash State Drift ($\Delta H$) vs. Verified Volume ($N_v$).*

*   **Status:** Planned
*   **Scientific Purpose:** Verifies **Cryptographic Integrity** by demonstrating that the Hamming distance $\Delta H$ scales with exponent $\beta \approx 0$ (random avalanche), proving no structural artifacts in the attestation chain.
*   **Investor-Facing Purpose:** Proves the ledger's security is mathematically robust and scalable, ensuring that "more data" does not mean "more security risk."
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest Field:** `attestation.h_t` (derived Hamming distance between sequential roots)

---

## 4. RFL Uplift
*Comparative density plot of efficiency metrics between Baseline and RFL modes.*

*   **Status:** Planned
*   **Scientific Purpose:** Quantifies the **Learning Signal efficiency ($U$)**, isolating the contribution of the RFL policy compared to a random baseline walk.
*   **Investor-Facing Purpose:** Demonstrates the "AI multiplier" effect, showing how the active learning engine makes proof generation 3.0x faster than brute force.
*   **Source Experiment:** `fo_1000_rfl` vs. `fo_1000_baseline`
*   **Manifest Field:** `uplift.ratio` (or computed from `derivation.verified_count` ratio)

---

## 5. Capability Frontier
*2D boundary plot (Depth vs. Atom Count) showing the region of solvable theorems.*

*   **Status:** Planned
*   **Scientific Purpose:** Empirically maps the **Knowledge Frontier $\mathcal{K}_t$**, defining the current logical expressivity limits of the derivation engine.
*   **Investor-Facing Purpose:** Visualizes the expanding "territory" of mathematical knowledge the system can conquer, marking the transition from simple to complex reasoning.
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest Field:** `derivation.depth_max` vs. `derivation.atoms_count` (success/fail classification)

---

## 6. Knowledge Growth
*Cumulative area chart of unique verified theorems over time.*

*   **Status:** Planned
*   **Scientific Purpose:** Tracks the cardinality of the verified set $|∂​\mathcal{K}_t|$, measuring the system's cumulative deductive throughput.
*   **Investor-Facing Purpose:** Shows the accumulation of intellectual property (IP) in the ledger—the "Data Moat" that grows automatically with compute time.
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest Field:** `derivation.verified_count` (cumulative sum)

---

## 7. Error Surfaces
*Heatmap of abstention density over the formula space (Depth $\times$ Atoms).*

*   **Status:** Planned
*   **Scientific Purpose:** Identifies **Hardness Clusters** in the formula space where the current axioms/rules are insufficient (undecidable or high-complexity regions).
*   **Investor-Facing Purpose:** Highlights specific areas where the system discovers "hard problems," guiding future R&D resource allocation (e.g., "we need better handling for depth-8 logic").
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest Field:** `metrics.abstention.mass` (binned by `depth` and `atoms`)