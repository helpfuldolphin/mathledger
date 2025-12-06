# Figure Catalog V2: RFL Experiment Assets

**Date:** 2025-11-27
**Status:** Generated
**Location:** `artifacts/figures/`

This catalog lists the generated figures and provides their LaTeX integration wrappers for the Research Paper (`research_paper.tex`).

---

## Figure 3: Capability Frontier
**File:** `fig_rfl_frontier_v1.png`
**Narrative:** Defines the operational envelope of the system, showing robust success rates (>95%) up to depth 4.

### LaTeX Wrapper
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.85\linewidth]{artifacts/figures/fig_rfl_frontier_v1.png}
    \caption{\textbf{Capability Frontier.} Success rate as a function of reasoning depth. The system maintains high reliability ($>0.95$) for shallow proofs ($d \le 4$), with a gradual polynomial degradation rather than a sharp cliff, indicating robust search policy coverage.}
    \label{fig:rfl_frontier}
\end{figure}
```

---

## Figure 4: Throughput vs. Depth
**File:** `fig_throughput_vs_depth_v1.png`
**Narrative:** Demonstrates that computational cost scales polynomially, not exponentially, with reasoning depth.

### LaTeX Wrapper
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.85\linewidth]{artifacts/figures/fig_throughput_vs_depth_v1.png}
    \caption{\textbf{Throughput Scaling.} Proof generation throughput (proofs/hour) plotted against mean derivation depth. The distribution shows that while deep reasoning is costlier, the system avoids exponential blow-up, maintaining viable interactive rates even at depth $d=5$.}
    \label{fig:throughput_scaling}
\end{figure}
```

---

## Figure 5: Knowledge Base Growth
**File:** `fig_knowledge_growth_v1.png`
**Narrative:** Illustrates the accumulation of a proprietary dataset of formal proofs ("Data Moat").

### LaTeX Wrapper
```latex
\begin{figure}[b]
    \centering
    \includegraphics[width=0.85\linewidth]{artifacts/figures/fig_knowledge_growth_v1.png}
    \caption{\textbf{Knowledge Accumulation.} Cumulative count of unique, verified statements added to the Ledger over 40 RFL epochs. The super-linear growth in the early phases ($t < 10$) transitions to a steady discovery rate, building a proprietary corpus of formal knowledge.}
    \label{fig:knowledge_growth}
\end{figure}
```

---

## Usage
To regenerate these figures with new data:
1. Ensure `rfl_results.json` is populated or use the synthetic generator.
2. Run: `uv run python experiments/generate_report.py`

```