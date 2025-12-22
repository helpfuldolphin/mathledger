from .canon import normalize, normalize_pretty, are_equivalent, get_atomic_propositions, canonical_bytes
from .taut import truth_table_is_tautology

# Backwards compatibility alias (deprecated - use truth_table_is_tautology directly)
# Note: truthtab.is_tautology is deprecated; this re-exports from taut for migration
from .taut import truth_table_is_tautology as is_tautology
