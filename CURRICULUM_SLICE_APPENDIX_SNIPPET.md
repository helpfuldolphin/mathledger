# Appendix: Curriculum Slice Summary

## Curriculum Slicing Strategy

The Reflexive Formal Learning (RFL) curriculum is segmented into distinct "slices" to systematically probe the behavior and robustness of the First Organism (FO) prover. These slices are defined by progressively increasing complexity parameters (e.g., number of atoms, maximum derivation depth and breadth), allowing for targeted analysis of the prover's performance, abstention patterns, and attestation integrity across varied cognitive loads.

```
+------------------------------------------------------------------+
|  Slice  |  Complexity Parameters  |  Expected Abstention  |  Purpose                   |
+------------------------------------------------------------------+
|   Easy  |  Atoms=3, Depth=3       |  ~0%                  |  Harness Sanity Check      |
|  Medium |  Atoms=5, Depth=7       |  5-15%                |  Zone of Proximal Dev.     |
|   Hard  |  Atoms=7, Depth=12      |  >25%                 |  Stress Test (FO Dyno Exp.)|
+------------------------------------------------------------------+
```

*Note: Only the Easy slice (atoms3-depth3) and the FO slice (first-organism-pl) were actually executed in Phase I. The Medium and Hard slices are retained here as conceptual strata for future experimentation.*

The "Easy" slice serves as a foundational sanity check, ensuring the underlying experimental harness and basic prover logic are fully operational and yielding near-zero abstention rates. Conversely, the "Hard" slice pushes the prover into a high-abstention regime (exceeding 25%), specifically designed to stress-test the system's resilience and resource management, as extensively utilized in the FO Dyno experiments for benchmarking extreme conditions. The "Medium" slice targets the optimal learning zone, where the RFL system encounters a balanced mix of solvable and challenging problems, driving meaningful epistemic growth and attestation over its zone of proximal development.
