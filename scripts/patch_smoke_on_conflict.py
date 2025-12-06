import io, os, re
p = r"scripts/run_fol_smoke.py"
s = open(p, "r", encoding="utf8").read()
# Add ON CONFLICT DO NOTHING to *each* proof_parents insert
s_new = re.sub(
    r'(INSERT INTO proof_parents\(child_hash,parent_hash\) VALUES \(%s,%s\))',
    r'\1 ON CONFLICT DO NOTHING',
    s
)
if s_new != s:
    open(p, "w", encoding="utf8").write(s_new)
    print("patched proof_parents inserts with ON CONFLICT DO NOTHING")
else:
    print("no changes made (already patched?)")
