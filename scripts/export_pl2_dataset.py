import os, csv, itertools, random
from normalization.canon import normalize
from backend.logic.truthtab import is_tautology
from backend.axiom_engine.features import featurize

random.seed(1234)
os.makedirs("artifacts/policy", exist_ok=True)
OUT = "artifacts/policy/dataset_pl2.csv"

atoms = ["p","q","r","s"]
def implies(a,b): return f"({a}->{b})"
def conj(a,b):    return f"({a}/\\{b})"
def disj(a,b):    return f"({a}\\/{b})"

# Candidate pool (depth ≤ ~5 by simple templates)
pool=set()
# base
for a in atoms:
    pool.add(a)
for a,b in itertools.permutations(atoms,2):
    pool.add(implies(a,b))
    pool.add(conj(a,b)); pool.add(disj(a,b))
# shallow expansions
base=list(pool)
for x in base:
    for y in base:
        if len(pool)>8000: break
        pool.add(implies(x,y))
        pool.add(conj(x,y)); pool.add(disj(x,y))

# sample to ~6000 unique
S = 6000
if len(pool)>S:
    pool = set(random.sample(list(pool), S))

with open(OUT,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["normalized","label"]+[f"f{i}" for i in range(128)])
    pos=neg=0
    for s in pool:
        n = normalize(s)
        lab = 1 if is_tautology(n) else 0
        if   lab==1 and pos>3500:  continue
        elif lab==0 and neg>3500:  continue
        fv = featurize(n)
        w.writerow([n, lab] + list(fv))
        pos += (lab==1); neg += (lab==0)
print(f"Wrote {OUT}  pos={pos} neg={neg}")
