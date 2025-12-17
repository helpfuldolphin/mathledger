# NL to Formal Translation Layer Specification

This document outlines the architectural specifications for translating Natural Language (NL) inputs into formal structures verifiable by MathLedger. It defines the architectural components, intermediate representations, failure modes, and a roadmap for integrating NL semantic claims.

## 1. Formal Architecture Diagram for NL-to-Formal Translation (Text-Based)

```
+-------------------+
|   Natural Language|
|   Input (e.g.,    |
|   "Contract A     |
|   stipulates...") |
+---------+---------+
          |
          v
+---------+---------+
|   NL Processor    |  (e.g., NLU, Semantic Parsing,
|   (Application    |        Coreference Resolution, NER)
|   Layer)          |
+---------+---------+
          |  (Generates Abstract Syntax Tree, Logical Forms,
          |   or other structured semantic representations)
          v
+---------+---------+
|   Intermediate    |  (e.g., Predicate Logic, RDF Triples,
|   Representation  |        Domain-Specific Abstract Syntax Tree (AST))
|   Normalizer/     |  (Canonicalization, Type Checking, Disambiguation)
|   Validator       |
+---------+---------+
          |  (Ensures adherence to a predefined schema/ontology)
          v
+---------+---------+
|   Formal Language |  (Translates intermediate representation to
|   Translator      |        MathLedger's native formal language)
|   (e.g., to Coq,  |
|   Isabelle,       |
|   or MathLedger's |
|   internal DSL)   |
+---------+---------+
          |  (Outputs formal proofs, assertions, or propositions)
          v
+---------+---------+
|   MathLedger      |  (Verification Engine, Theorem Prover,
|   Substrate       |        State Machine)
|   (Formal Truth   |
|   Verification)   |
+-------------------+
```

## 2. Contract: Intermediate Representations Required

Before MathLedger can accept NL-derived structures, the NL Processor and Normalizer/Validator layers must collectively produce an intermediate representation that adheres to the following contract. This ensures semantic clarity, formal tractability, and minimizes ambiguity before formal translation.

**Core Requirements:**

*   **Formal Type System:** All entities, predicates, and functions derived from NL must be explicitly typed according to a predefined, extensible ontology. This includes basic types (e.g., `Person`, `Asset`, `Quantity`, `Date`) and potentially higher-order types.
*   **Unique Referents:** All named entities and pronouns must resolve to unique, unambiguous identifiers within the domain context. Coreference resolution is critical here.
*   **Quantifier Scope Resolved:** Ambiguities arising from quantifiers (e.g., "every", "some", "all") must be explicitly resolved into their logical scope.
*   **Temporal and Spatial Grounding:** All temporal and spatial references must be resolved to precise, absolute or relative, machine-interpretable values (e.g., UTC timestamps, specific coordinates, or references to predefined spatial entities).
*   **Modality and Deontic Logic:** Modal verbs (e.g., "must", "can", "may") and deontic operators (e.g., "obligatory", "permitted", "forbidden") must be translated into explicit formal logical operators within the intermediate representation.
*   **Event Semantics:** Actions and events described in NL must be represented as structured events with clear actors, patients, instruments, and temporal/spatial extents.
*   **Logical Connectives:** NL conjunctions, disjunctions, implications, and negations must map directly and unambiguously to their formal logical counterparts.
*   **Domain-Specific Constraints:** The intermediate representation must respect and incorporate domain-specific constraints defined in the ontology (e.g., "a contract cannot contradict canon law").

**Example Intermediate Representation (Conceptual):**

A predicate logic-like structure, potentially serialized as JSON or a graph format:

```json
{
  "statement_id": "NL_001",
  "source_nl": "If a user commits fraud, their account must be frozen immediately.",
  "logical_form": {
    "type": "IMPLICATION",
    "antecedent": {
      "type": "PREDICATE",
      "name": "commits_fraud",
      "args": [
        {"type": "ENTITY", "id": "user_X", "name": "User X"}
      ]
    },
    "consequent": {
      "type": "OBLIGATION",
      "operator": "MUST",
      "action": {
        "type": "EVENT",
        "name": "freeze_account",
        "actor": {"type": "SYSTEM_AGENT", "id": "system_lawkeeper"},
        "patient": {"type": "ENTITY", "id": "user_X", "name": "User X", "property": "account"},
        "timing": {"type": "TEMPORAL", "value": "IMMEDIATELY"}
      }
    }
  },
  "grounding_context": {
    "user_X": {"nl_reference": ["a user"], "schema_ref": "Person"},
    "commits_fraud": {"nl_reference": ["commits fraud"], "schema_ref": "ActionType.Fraudulent"},
    "freeze_account": {"nl_reference": ["frozen"], "schema_ref": "SystemAction.AccountFreeze"}
  },
  "confidence_score": 0.98,
  "disambiguation_notes": []
}
```

## 3. Failure-Analysis Table for NL→Formal Mapping

| Failure Mode                       | Description                                                                                             | Example NL Input                                                | Substrate Governance / Mitigation                                                                                                          |
| :--------------------------------- | :------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| **Ambiguity (Lexical/Structural)** | A word or sentence structure has multiple plausible interpretations.                                  | "The agent saw the man with the telescope."                     | **Rejection:** MathLedger refuses formalization without explicit disambiguation from an oracle (human/expert system). <br>**Confidence Threshold:** If confidence in a single interpretation is below threshold, reject.<br>**Annotation Requirement:** Force explicit tagging/annotation of intended meaning. |
| **Underspecification**             | NL input lacks sufficient detail for a complete formal representation.                                | "Perform the action."                                           | **Rejection:** Demand more specific input. <br>**Default Protocol:** Allow fallback to predefined, safe defaults *only if explicitly configured and audited* for the specific underspecified element. |
| **Contradiction (Internal)**       | The NL input, once formalized, leads to a logical inconsistency within itself.                       | "The door is open and the door is closed."                      | **Formal Proof of Contradiction:** MathLedger’s verification engine detects `False` and rejects the input, flagging the specific contradiction points.                                  |
| **Ontological Mismatch**           | NL concepts cannot be mapped to the predefined formal ontology/schema.                                | "The fairy flew over the rainbow."                              | **Schema Extension Request:** Flag unknown concepts. Require ontology extension or re-evaluation of the NL input. <br>**Type Error:** Formal translator raises a type error.             |
| **Vagueness / Gradability**        | Concepts with continuous or subjective interpretations (e.g., "tall," "soon").                        | "The payment should be made soon."                              | **Rejection:** MathLedger cannot operate on vague terms without formal thresholds. <br>**Parameterization:** Require explicit parameterization (e.g., "soon = within 24 hours"). |
| **Implicit Knowledge Dependency**  | NL relies on common-sense or background knowledge not present in the formal system.                    | "Avoid violating the spirit of the law."                        | **Rejection:** Cannot formalize "spirit" without explicit definition. <br>**Formalization of Principles:** Require abstract principles to be formalized as axioms if they are to be verifiable. |
| **Hallucination (NLU)**            | The NLU layer misinterprets or fabricates semantic content not present in the original NL.             | NL says "X is Y," NLU outputs "X is Z."                         | **Redundancy/Cross-Verification:** Multiple independent NLU models/algorithms. <br>**Human-in-the-Loop:** Periodic audit/review of NLU output. <br>**Formal Constraints:** Leverage formal constraints to detect nonsensical NLU outputs (e.g., type violations). |
| **Scope Misinterpretation**        | Incorrect assignment of quantifier scope or logical precedence.                                       | "Every person does not have a car." (Could mean `¬∀x.Car(x)` or `∀x.¬Car(x)`) | **Explicit Scope Markers:** Require the NL processing layer to use explicit scope delimiters or logical bracketing in the intermediate representation. <br>**Heuristic Validation:** Apply domain-specific heuristics to prefer one scope interpretation if context allows. |

## 4. Roadmap for Integrating NL Semantic Claims into Evidence Packs without Hallucination Risk

The integration of NL semantic claims into MathLedger's evidence packs requires a multi-layered, strictly controlled approach to mitigate hallucination risks. The core principle is that MathLedger always verifies formal truth; NL serves as an *input source* that must be rigorously transformed and audited.

**Phase 1: NL-Annotated Formal Claims (Low Risk, Current-Adjacent)**
*   **Description:** Start by linking NL text directly to *already formally verified* claims. The NL serves as a human-readable annotation or explanation of a formal assertion, but the formal assertion's truth is independently established.
*   **Mechanism:** Evidence packs include a `NL_Annotation` field alongside formal proofs. This field stores the relevant NL snippet and a pointer to the formal claim it describes.
*   **Risk Mitigation:** Hallucination risk is external to MathLedger; the formal claim stands alone. The NL annotation is for human context, not formal verification.
*   **Example:** A formal proof for `(x > 0)` in an evidence pack is accompanied by the NL: "The value of `x` must be positive."

**Phase 2: Human-Curated NL-to-Formal Mappings (Medium Risk, Controlled Expansion)**
*   **Description:** Introduce a process where specific, pre-approved NL patterns or statements are manually mapped by experts to their corresponding formal representations. These mappings are stored in a highly governed registry.
*   **Mechanism:** Evidence packs can include `NL_Derived_Claim` entries. Each entry references a formal claim `F` and the originating NL `N`. Critically, `F` must be derivable from `N` via a *registered and human-verified* translation rule `T`. The evidence pack implicitly verifies that `T(N) = F`.
*   **Risk Mitigation:** No NLU/semantic parsing is automated yet. Human experts ensure translation fidelity. Hallucination is prevented by direct human supervision of the translation rules.
*   **Example:** A registered rule: "If NL is 'A is greater than B', then formal is `GreaterThan(A, B)`". An evidence pack uses this rule to derive `GreaterThan(Value(Account1), Value(Account2))` from "Account 1 is greater than Account 2".

**Phase 3: Assisted NL-to-Formal Translation (Higher Risk, Human-in-the-Loop)**
*   **Description:** Integrate AI-powered NL processors (NLU, semantic parsers) to *suggest* formal representations from NL inputs. These suggestions are never directly accepted by MathLedger without explicit human review and approval.
*   **Mechanism:** An intermediate "NL Claim Proposal" artifact is generated, containing the raw NL, the AI's proposed formalization, confidence scores, and identified ambiguities. A human validator reviews, corrects, and approves this proposal. Only the *approved formal claim* is then ingested by MathLedger and linked to an evidence pack, along with an audit trail of the NL source and human approval.
*   **Risk Mitigation:** Hallucination by the AI is mitigated by the mandatory human-in-the-loop validation step. The formal claim's integrity is maintained by human oversight. The audit trail provides transparency.
*   **Dependencies:** Robust UI/UX for human validation, version control for proposed claims.

**Phase 4: Trust-Anchored NL-to-Formal Gateways (Automated, High Assurance)**
*   **Description:** For extremely well-defined, bounded domains with highly stable ontologies, develop formally verified "translation gateways." These gateways are themselves proven correct in their translation from a constrained subset of NL to specific formal language constructs.
*   **Mechanism:** These gateways act as trusted components. NL input within their scope is automatically translated to a formal claim. The evidence pack now includes a reference to the *verified gateway* used for translation, in addition to the NL source and the derived formal claim. MathLedger implicitly verifies the correctness of the gateway's operation.
*   **Risk Mitigation:** Hallucination is mitigated by the formal verification of the translation gateway itself. This is the highest level of automation, limited to contexts where the NL-to-formal mapping is provably unambiguous and correct.
*   **Dependencies:** Formal verification of translation logic, highly constrained NL input domain, robust change management for gateways.

Throughout all phases, MathLedger's substrate governance will enforce that any NL-derived claim *must always resolve to a precisely defined formal statement* that is then subject to the same rigorous verification as any other natively formal claim. The role of NL is to inform the creation of formal claims, not to replace the need for them.
