"""Test-facing rules helpers for propositional MP: Statement + ModusPonens with strict semantics."""
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple
from datetime import datetime
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Text, Integer, Boolean, DateTime
from functools import lru_cache

Base = declarative_base()

@lru_cache(maxsize=1000)
def _cached_normalize(text: str) -> str:
    """Cached normalization to avoid redundant calls."""
    try:
        from normalization.canon import normalize
        return normalize(text)
    except Exception:
        return _strip_outer(text).replace(" ", "")

@lru_cache(maxsize=4096)
def _strip_outer(s: str) -> str:
    s = s.strip()
    if not s: return s
    if s[0] != "(" or s[-1] != ")": return s
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                return s
    return s[1:-1].strip()

@lru_cache(maxsize=4096)
def _split_top_impl(s: str) -> Tuple[Optional[str], Optional[str]]:
    s_stripped = s.strip()
    if len(s_stripped) < 3:
        return (None, None)
    
    depth = 0
    for i in range(len(s_stripped) - 1):
        ch = s_stripped[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and s_stripped[i:i+2] == "->":
            a = s_stripped[:i].strip()
            b = s_stripped[i+2:].strip()
            return (a, b) if a and b else (None, None)
    return (None, None)

@lru_cache(maxsize=4096)
def _is_implication(stmt: str) -> bool:
    a, b = _split_top_impl(stmt); return a is not None and b is not None

@lru_cache(maxsize=4096)
def _parse_implication(stmt: str) -> Tuple[Optional[str], Optional[str]]:
    a, b = _split_top_impl(stmt)
    if a is None or b is None: return (None, None)
    return (_strip_outer(a), _strip_outer(b))

class Statement(Base):
    __tablename__ = "statements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    system_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # canonicalized content
    normalized_text: Mapped[Optional[str]] = mapped_column(Text)
    # pretty/original (some schemas use 'text'—keep both if needed)
    text: Mapped[Optional[str]] = mapped_column(Text)
    # sha256 of normalized_text
    hash: Mapped[Optional[str]] = mapped_column(String(64))
    # optional metadata the tests sometimes read
    content_norm: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    derivation_rule: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_axiom: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    derivation_depth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    # For backward compatibility with tests
    parent_statements: List[str] = []
    keep_text_content: bool = False

    def __init__(self, text: str = None, is_axiom: bool = False, derivation_rule: Optional[str] = None,
                 derivation_depth: Optional[int] = None, parent_statements: List[str] = None,
                 keep_text_content: bool = False, system_id: int = 1, normalized_text: str = None,
                 hash: str = None, content_norm: str = None, created_at: datetime = None):
        self.text = text
        self.is_axiom = is_axiom
        self.derivation_rule = derivation_rule
        self.derivation_depth = derivation_depth
        self.parent_statements = parent_statements or []
        self.keep_text_content = keep_text_content
        self.system_id = system_id
        self.normalized_text = normalized_text
        self.hash = hash
        self.content_norm = content_norm
        from backend.repro.determinism import deterministic_timestamp
        self.created_at = created_at or deterministic_timestamp(0)

    @property
    def content(self) -> str:
        if self.keep_text_content:
            return self.text
        try:
            from normalization.canon import normalize
            return normalize(self.text)
        except Exception:
            return _strip_outer(self.text).replace(" ", "")

class ModusPonens:
    _parse_implication = staticmethod(_parse_implication)
    _is_implication = staticmethod(_is_implication)

    @staticmethod
    def can_apply(premises: List[Statement]) -> bool:
        if len(premises) != 2: return False
        x, y = premises[0].text, premises[1].text
        if _is_implication(x):
            a,_ = _parse_implication(x)
            return a is not None and _cached_normalize(a) == _cached_normalize(y)
        if _is_implication(y):
            a,_ = _parse_implication(y)
            return a is not None and _cached_normalize(a) == _cached_normalize(x)
        return False

    @staticmethod
    def apply(premises: List[Statement]) -> List[Statement]:
        if len(premises) != 2: return []
        out: List[Statement] = []
        p, q = premises
        for imp, ante in ((p.text, q.text), (q.text, p.text)):
            if _is_implication(imp):
                a, c = _parse_implication(imp)
                if a and c and _cached_normalize(ante) == _cached_normalize(a):
                    out.append(
                        Statement(
                            text=_cached_normalize(c),
                            is_axiom=False,
                            derivation_rule="MP",
                            parent_statements=[ante, imp],
                        )
                    )
        seen: Set[str] = set(); uniq: List[Statement] = []
        for s in out:
            if s.content not in seen:
                seen.add(s.content); uniq.append(s)
        return uniq

@lru_cache(maxsize=2048)
def _apply_modus_ponens_cached(statements_frozen: frozenset) -> frozenset:
    """
    Cached implementation of Modus Ponens.
    Takes frozenset for hashability, returns frozenset for caching.
    """
    derived: Set[str] = set()
    implications_by_antecedent = {}
    normalized_statements = set()

    for stmt in statements_frozen:
        if _is_implication(stmt):
            a, c = _parse_implication(stmt)
            if a and c:
                norm_a = _cached_normalize(a)
                norm_c = _cached_normalize(c)
                if norm_a not in implications_by_antecedent:
                    implications_by_antecedent[norm_a] = []
                implications_by_antecedent[norm_a].append(norm_c)
        else:
            norm_stmt = _cached_normalize(stmt)
            normalized_statements.add(norm_stmt)

    for norm_stmt in normalized_statements:
        if norm_stmt in implications_by_antecedent:
            for consequent in implications_by_antecedent[norm_stmt]:
                if consequent not in normalized_statements:
                    derived.add(consequent)

    return frozenset(derived)

def apply_modus_ponens(statements: Set[str]) -> Set[str]:
    """
    Optimized Modus Ponens application using antecedent indexing and proof caching.
    Reduces complexity from O(n²) to O(n) by indexing implications by antecedent.
    
    Performance optimizations:
    - Single-pass normalization with caching
    - Antecedent indexing for O(1) lookup
    - Precomputed normalized statement set
    - Proof caching for repeated derivations
    """
    statements_frozen = frozenset(statements)
    derived_frozen = _apply_modus_ponens_cached(statements_frozen)
    return set(derived_frozen)
