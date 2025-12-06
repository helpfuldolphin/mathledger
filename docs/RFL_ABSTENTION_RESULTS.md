# RFL Abstention Results

**Analyst:** GEMINI-D  
**Date:** 2025-11-27  
**Source Data:** `results/fo_baseline.jsonl`, `results/fo_rfl.jsonl`  
**Status:** Verified (Prototype Data)

## Executive Summary

Recursive Feedback Learning (RFL) demonstrates a significant reduction in abstention rates compared to the baseline. After a burn-in period of 200 cycles, the RFL agent stabilizes at a lower abstention rate, indicating improved policy coverage.

### Statistical Summary (Post-Burn-in)

| Metric | Baseline | RFL | Delta ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Mean Abstention Rate ($\bar{A}$)** | 0.232 | 0.061 | **-0.172** |
| **Sample Size ($n$)** | 99 cycles | 99 cycles | - |

*Note: Stats computed on cycles > 200.*

## Figures for Research Paper

### Figure 1: Rolling Abstention Rate

\begin{figure}[h]
\centering
\includegraphics[width=0.9\linewidth]{artifacts/figures/rfl_abstention_rate.png}
\caption{Rolling abstention rate ($W=100$). RFL reduces abstention systematically after burn-in.}
\label{fig:rfl_abstention_rate}
\end{figure}

### Figure 2: Cumulative Abstentions

\begin{figure}[h]
\centering
\includegraphics[width=0.9\linewidth]{artifacts/figures/rfl_cumulative_abstentions.png}
\caption{Cumulative abstentions over time. The diverging slopes highlight the long-term efficiency gain of RFL.}
\label{fig:rfl_cumulative}
\end{figure}

## Method Distribution

Overall usage of verification methods across the full run (300 cycles):

**Baseline:**
- `auto`: 78.0%
- `lean-disabled`: 22.0% (Abstentions)

**RFL:**
- `auto`: 92.0%
- `lean-disabled`: 8.0%

## Conclusion
The hypothesis is supported: $A_{RFL}(t) < A_{Baseline}(t)$ for $t > t_{burn\_in}$.
