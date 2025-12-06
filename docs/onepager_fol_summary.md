# PL2 and FOL One-Pagers Summary

## Newly Added FOL Assets
- `docs/onepager_fol.tex` populated with metrics from `artifacts/wpv5/fol_ab.csv` and `fol_stats.json`.
- `docs/onepager_fol.pdf` generated via `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=docs`.
- Macros used:
  - `\PolicyHashFOL = f483821397526ae5361625c1530689f9f0df9ea06a3b73e7452c698ceee728d7`
  - `\BlockRootBLFOL = f0c2a2b9d8c54c7d56d7a5f5e8ddaa98c1df42cb8bc87db69af24aef3fb26bbd`
  - `\BlockRootGFOL = 49e8e636d603b88328b91a729064d7d303ea6959a24bb4b112c82b7d173e14a7`
  - `\BaselineMeanFOL = 44.0`, `\GuidedMeanFOL = 132.0`, `\UpliftXFOL = 3.0`

## Existing PL2 Assets
- `docs/onepager_pl2.pdf` previously compiled; no changes required this run.
- `docs/onepager_pl2.tex` retains macros sourced from `pl2_ab.csv`/`pl2_stats.json`.

## Notes
- Ensure the generated PDFs (`docs/onepager_pl2.pdf`, `docs/onepager_fol.pdf`) are reviewed for overfull hboxes (long hashes) prior to distribution.
- MiKTeX warnings about administrative updates persist but do not affect output.
