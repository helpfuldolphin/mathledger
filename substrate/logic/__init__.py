"""Logic substrate facade.

Provides stable import anchors for propositional canonicalization and support
utilities required by derivation and normalization pipelines.
"""

from normalization import canon
from normalization import taut
from normalization import truthtab

__all__ = ["canon", "taut", "truthtab"]
