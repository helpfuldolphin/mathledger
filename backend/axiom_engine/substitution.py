import re
from typing import List, Tuple, Optional, Dict

OP_IMP = "->"
OP_AND = "/\\"
OP_OR  = "\\/"

class SubstitutionRule:
    def __init__(self, max_depth: int = 1, atoms: List[str] | None = None):
        self.max_depth = max_depth
        self.atoms = atoms or ["p", "q", "r", "s"]
        self.size_budget: int | None = None  # optional global cap

    # ---------- parsing helpers ----------
    def _strip_outer_parens(self, s: str) -> str:
        s = s.strip()
        if not s or s[0] != "(" or s[-1] != ")": return s
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(s)-1: return s
        return s[1:-1].strip()

    def _split_top(self, s: str, op: str) -> Tuple[Optional[str], Optional[str]]:
        s = s.strip()
        depth = 0; w = len(op); i = 0
        while i <= len(s)-w:
            ch = s[i]
            if ch == "(": depth += 1; i += 1; continue
            if ch == ")": depth -= 1; i += 1; continue
            if depth == 0 and s[i:i+w] == op:
                return s[:i].strip(), s[i+w:].strip()
            i += 1
        return None, None

    def _formula_depth(self, s: str) -> int:
        t = re.sub(r"\s+", "", self._strip_outer_parens(s))
        # implication adds 1 + max(children)
        a,b = self._split_top(t, OP_IMP)
        if a is not None and b is not None:
            return 1 + max(self._formula_depth(a), self._formula_depth(b))
        # conjunction counts as 1 + max(children)
        a,b = self._split_top(t, OP_AND)
        if a is not None and b is not None:
            return 1 + max(self._formula_depth(a), self._formula_depth(b))
        # disjunction treated as max(children) (doesn't increase depth)
        a,b = self._split_top(t, OP_OR)
        if a is not None and b is not None:
            return max(self._formula_depth(a), self._formula_depth(b))
        return 0

    # ---------- substitution API ----------
    def _extract_metavariables(self, axiom: str) -> List[str]:
        # single-letter lowercase identifiers become metavars
        return sorted(set(re.findall(r"\b([a-z])\b", axiom)))

    def substitute_axiom(self, axiom: str, subst: Dict[str,str]) -> str:
        out = axiom
        for k,v in subst.items():
            out = re.sub(rf"\b{k}\b", v.replace(" ", ""), out)
        out = re.sub(r"\s+", "", out)
        return out

    def generate_instances(self, axiom: str, max_instances: int = 10) -> List[Tuple[str, Dict[str,str]]]:
        """Return list of (instance, substitution) tuples."""
        metas = self._extract_metavariables(axiom)
        if not metas:
            return [(re.sub(r"\s+","",axiom), {})]
        out: List[Tuple[str, Dict[str,str]]] = []
        def backtrack(i: int, cur: Dict[str,str]):
            if self.size_budget is not None and len(out) >= self.size_budget:
                return
            if i == len(metas):
                inst = self.substitute_axiom(axiom, cur)
                out.append((inst, cur.copy()))
                return
            for a in self.atoms:
                cur[metas[i]] = a
                backtrack(i+1, cur)
                if len(out) >= max_instances:
                    break
        backtrack(0, {})
        return out[:max_instances]

    def apply_to_axioms(self, axioms: List[str], max_instances_per_axiom: int = 10) -> List[Tuple[str, str, Dict[str,str]]]:
        """Return list of (original_axiom_with_spaces, instance, substitution)."""
        results: List[Tuple[str, str, Dict[str,str]]] = []
        for ax in axioms:
            for inst, sub in self.generate_instances(ax, max_instances=max_instances_per_axiom):
                results.append((ax, inst, sub))
                if self.size_budget is not None and len(results) >= self.size_budget:
                    return results[:self.size_budget]
        return results

    def generate_formulas(self, depth: int) -> List[str]:
        """Budgeted generation up to depth. Ensures 'p->q' appears for depth>=1."""
        budget = self.size_budget
        def add(lst: List[str], s: str):
            if budget is None or len(lst) < budget:
                lst.append(s)

        if depth <= 0:
            base: List[str] = []
            for a in self.atoms:
                add(base, a)
            return base

        out: List[str] = []
        # atoms
        for a in self.atoms:
            add(out, a)
        # simple implications
        for a in self.atoms:
            for b in self.atoms:
                add(out, f"{a}{OP_IMP}{b}")

        if depth == 1:
            return out

        # depth >= 2: add conjunctions & small imp-chains, respect budget
        for i, a in enumerate(self.atoms):
            for b in self.atoms[i:]:
                add(out, f"({a}{OP_AND}{b})")
                add(out, f"({b}{OP_AND}{a})")
        # small sample of disjunctions
        for a in self.atoms:
            add(out, f"({a}{OP_OR}{self.atoms[0]})")
        # imp-chains
        for a in self.atoms:
            add(out, f"{a}{OP_IMP}({self.atoms[0]}{OP_IMP}{self.atoms[1]})")

        return out
