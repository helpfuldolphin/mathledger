# backend/nl/nl_ir_failure_codes.py

"""
Defines standardized error codes for failures detected during the validation
of Natural Language Intermediate Representations (NL IR).

These codes are used to map specific validation failures to broader
governance responses, ensuring consistent handling of ambiguity and
potential hallucinations.
"""

# --- Schema and Structural Errors ---
SCHEMA_REQUIRED_FAILED = "SCHEMA_REQUIRED_FAILED"
SCHEMA_TYPE_FAILED = "SCHEMA_TYPE_FAILED"
SCHEMA_NOT_FOUND = "SCHEMA_NOT_FOUND"
DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
UNEXPECTED_VALIDATION_ERROR = "UNEXPECTED_VALIDATION_ERROR"

# --- Semantic and Content Errors ---

AMBIGUITY_DETECTED = "AMBIGUITY_DETECTED"
"""
A general code indicating that the 'disambiguation_notes' field is not empty,
signaling that some form of ambiguity was noted by the NLU processor. This
serves as a catch-all for issues like referent and scope ambiguity when a more
specific code isn't used.
"""

AMBIGUOUS_REFERENT = "AMBIGUOUS_REFERENT"
"""
Specifically denotes an ambiguity in resolving a pronoun or entity mention
to a unique identifier in the grounding context.
Example: "He saw it." - Who is 'He'? What is 'it'?
"""

SCOPE_AMBIGUITY = "SCOPE_AMBIGUITY"
"""
Denotes an ambiguity in the scope of a quantifier (e.g., 'all', 'some')
or a logical operator (e.g., negation).
Example: "All members are not approved." -> (NOT (ALL members approved)) OR (ALL members (NOT approved))
"""

ONTOLOGY_MISMATCH = "ONTOLOGY_MISMATCH"
"""
The 'schema_ref' in the grounding_context for an entity or concept does not
exist in the MathLedger's known ontology.
Example: A 'schema_ref' of 'mythical:Dragon' when the ontology only covers 'animal:*'.
"""
