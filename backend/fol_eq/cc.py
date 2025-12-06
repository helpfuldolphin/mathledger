from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Term:
    sym: str
    args: Tuple["Term", ...] = ()

def const(a: str) -> Term:
    return Term(a, ())

def fun(f: str, *args: Term) -> Term:
    return Term(f, tuple(args))

class UnionFind:
    def __init__(self) -> None:
        self.p: Dict[int, int] = {}
        self.r: Dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.p; r = self.r
        if x not in p:
            p[x] = x
            r[x] = 0
            return x
        # path halving
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        pa, pb = self.find(a), self.find(b)
        if pa == pb:
            return False
        ra, rb = self.r[pa], self.r[pb]
        if ra < rb:
            self.p[pa] = pb
        elif rb < ra:
            self.p[pb] = pa
        else:
            self.p[pb] = pa
            self.r[pa] = ra + 1
        return True

class CC:
    """Minimal ground congruence closure (EUF) with simple certificate edges."""
    def __init__(self) -> None:
        self.uf = UnionFind()
        self.parents: Dict[Term, List[Term]] = {}
        self.sig: Dict[Tuple[str, Tuple[int, ...]], Term] = {}
        self.ids: Dict[Term, int] = {}
        self._next_id = 0

    def _tid(self, t: Term) -> int:
        if t not in self.ids:
            self.ids[t] = self._next_id
            self._next_id += 1
        return self.ids[t]

    def add_term(self, t: Term) -> None:
        for a in t.args:
            self.add_term(a)
        self._tid(t)
        for a in t.args:
            self.parents.setdefault(a, []).append(t)

    def _classes(self, t: Term) -> Tuple[int, ...]:
        return tuple(self.uf.find(self._tid(a)) for a in t.args)

    def _canon(self, t: Term) -> Tuple[str, Tuple[int, ...]]:
        return (t.sym, self._classes(t))

    def _enqueue_congruence(self, t: Term, work: List[Tuple[int, int]]) -> None:
        if not t.args:
            return
        key = self._canon(t)
        rep = self.sig.get(key)
        cid = self.uf.find(self._tid(t))
        if rep is None:
            self.sig[key] = t
        else:
            rid = self.uf.find(self._tid(rep))
            if cid != rid:
                work.append((cid, rid))

    def assert_eqs(self, eqs: List[Tuple[Term, Term]]) -> List[Tuple[str, Tuple[int, int]]]:
        """Assert equations; return simple proof edges ('axiom' or 'cong', (u,v))."""
        for (s, t) in eqs:
            self.add_term(s); self.add_term(t)

        proof: List[Tuple[str, Tuple[int, int]]] = []
        work: List[Tuple[int, int]] = []

        # unify axioms
        for (s, t) in eqs:
            a, b = self._tid(s), self._tid(t)
            if self.uf.union(a, b):
                proof.append(("axiom", (a, b)))
                # enqueue congruence on parents
                for parent in set(self.parents.get(s, []) + self.parents.get(t, [])):
                    self._enqueue_congruence(parent, work)

        # process congruence worklist
        while work:
            a, b = work.pop()
            if self.uf.union(a, b):
                proof.append(("cong", (a, b)))
                # compact: full impl would walk parents of both classes
        return proof

    def equal(self, s: Term, t: Term) -> bool:
        if s not in self.ids: self.add_term(s)
        if t not in self.ids: self.add_term(t)
        return self.uf.find(self._tid(s)) == self.uf.find(self._tid(t))
