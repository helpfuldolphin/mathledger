# Whitepaper Changelog

## 2025-12-10 - LaTeX Structure Correction
- **FIX:** Corrected an issue where `\appendix` was mistakenly inserted into `docs/whitepaper/sections/section3_substrate_and_governance.tex`. The content was restored to its original section format, and the artifact grounding table, along with the smoke-test checklist, was integrated as a new subsection.
- **Reasoning:** Inserting `\appendix` within a sub-document that is included in a larger `main.tex` file fundamentally disrupts the LaTeX document structure. It can lead to incorrect section numbering, unexpected formatting, and compilation errors, rendering the formal document invalid or unreadable. Appendices should only be declared once in the main document, signifying the start of supplementary material. This fix ensures proper LaTeX document hierarchy and maintainability.

## LaTeX Document Structure Invariant
- **INVARIANT:** Appendix declarations (`\appendix` command) are allowed only in root LaTeX documents (e.g., `main.tex`), not within included section files.
- **POLICY LINK:** Refer to the [Whitepaper Build Process Checksum Governance Policy](docs/whitepaper/README.md#checksum-governance-policy) for related document integrity invariants.
