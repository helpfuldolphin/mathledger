"""Logic substrate facade.

Provides stable import anchors for propositional canonicalization and support
utilities required by derivation and normalization pipelines.

Note: truthtab is deprecated in favor of taut. It is kept for backwards
compatibility but will be removed in v1.0.
"""

from normalization import canon
from normalization import taut

# Deprecated: truthtab is kept for backwards compatibility
# Use taut.truth_table_is_tautology() instead of truthtab.is_tautology()
from normalization import truthtab  # noqa: F401 (deprecated)

__all__ = ["canon", "taut", "truthtab"]
