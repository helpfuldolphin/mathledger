**Internal Memo: What We Can and Cannot Claim Publicly About MathLedger**

**TO:** All Project Stakeholders
**FROM:** GEMINI-N, Public Narrative & Risk Guard
**DATE:** 2025-12-10
**SUBJECT:** Public Communication Guardrails for MathLedger

This memo outlines the approved messaging framework for all public-facing communications regarding the MathLedger project. Adherence to these guidelines is mandatory to manage public perception, mitigate narrative risk, and avoid fueling anti-AI sentiment. Our goal is to be transparent and confident about our achievements without overpromising or misrepresenting the nature of our work.

---

### Safe Claims
These are well-supported, verifiable statements about our technology. They focus on our capabilities and contributions without hyperbole.

*   **On Quantifying System Properties:** "MathLedger provides a framework for formally quantifying the stability and performance of a complex system under a specific, well-defined control law."
*   **On Verification and Auditing:** "We have developed a verifiable ledger system that allows for rigorous, automated auditing of system states and transitions against a predefined canonical specification."
*   **On Testing and Integration:** "The project features a robust testing harness for evaluating the integrity and determinism of system migrations and updates before deployment."
*   **On Governance Mechanisms:** "MathLedger implements auditable, programmatic governance mechanisms based on explicitly defined rules and attestations."

---

### Borderline Claims
These claims are technically accurate but carry a high risk of being misinterpreted. They must be phrased with extreme care, always including necessary context and caveats.

*   **"Provable Safety" -> Rephrase:** Instead of saying "MathLedger is provably safe," we should say, "MathLedger can prove that a system complies with a specific, limited set of safety properties." The key is scoping the claim to what is actually being proven.
*   **"Autonomous Governance" -> Rephrase:** Avoid this term. Instead, say, "MathLedger enables automated enforcement of human-defined governance rules." This clarifies that humans are still the ultimate source of authority.
*   **"Predictive Stability" -> Rephrase:** Instead of "We can predict the system will be stable," say, "We can model and bound the system's behavior under known conditions, allowing us to analyze its stability characteristics." This avoids implying perfect prediction of an open-ended future.

---

### Forbidden Claims
Under no circumstances should we make any of the following claims, or statements that imply them. Making such claims would be a serious breach of public trust and project integrity.

*   **"We solved AI alignment."** (This is the primary directive. We have not solved it; we have created tools to study it in a limited context.)
*   **"MathLedger makes AI safe" or "eliminates risk."** (No system is perfectly safe or risk-free. Our work is about risk *mitigation* and *analysis*.)
*   **"The system is 'sentient,' 'conscious,' or 'understands'."** (Avoid anthropomorphism entirely.)
*   **"This is AGI" or "a pathway to AGI."** (Our work is focused on formal systems and bounded rationality, not on creating general intelligence.)
*   **Any claim of infallibility or perfect determinism in uncontrolled, real-world environments.**

---

### Proposed Public-Facing One-Liners
These are concise, powerful, and safe statements for use in presentations, social media, and press materials.

1.  "MathLedger is the first test harness where a reasoning engine has to formally prove it is stable before it can be scaled."
2.  "We're building the 'flight data recorder' for complex AI, creating a verifiable audit trail for every decision the system makes."
3.  "MathLedger turns AI governance from a philosophical debate into an engineering discipline."
4.  "Our work is about ensuring that the systems we build are not just powerful, but also predictable and accountable to their creators."
5.  "Before you can trust an AI, you have to be able to audit it. MathLedger provides the ledger for that audit."

---

### Neutrality Invariant Declaration

**Public-facing artifacts must never use alarmist or emotive language.** All communication must be factual, precise, and objective. Enforcement is supported by `tools/lint_public_language.py`. This is run as part of the `lint-public-language` job (step: `Run Public Language Linter`) in the proposed `ci.yml` workflow.
