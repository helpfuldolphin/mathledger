import os, re, json, csv, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque

# ---------- helpers ----------
def iso_date(ts):
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        s = str(ts).strip()
        # try parse ISO-ish
        for fmt in ("%Y-%m-%d","%Y-%m-%dT%H:%M:%SZ","%Y-%m-%d %H:%M:%S"):
            try: return datetime.strptime(s, fmt).date().isoformat()
            except: pass
        # dateutil fallback if available
        try:
            from dateutil import parser
            return parser.parse(s).date().isoformat()
        except:
            return None
    except:
        return None

def add_point(day, acc, key, delta):
    if day is None: return
    acc[day][key] = acc[day].get(key, 0) + max(0, int(delta))

def add_value(day, acc, key, value):
    if day is None: return
    acc[day][key] = value

def git_days():
    # returns list of YYYY-MM-DD with activity, oldest->newest
    try:
        out = subprocess.check_output(["git","log","--date=short","--pretty=%ad"]).decode()
        days = sorted(set([ln.strip() for ln in out.splitlines() if ln.strip()]))
        return days
    except:
        return []

def git_estimate_proofs_per_day():
    # heuristic: count commits touching known proof/engine paths
    paths = [
        "backend/axiom_engine", "backend/logic", "backend/ledger",
        "tests/test_mp.py", "tests/test_taut.py", "tests/test_derive.py",
        "migrations", "artifacts"
    ]
    days = defaultdict(int)
    try:
        out = subprocess.check_output(["git","log","--name-only","--date=short","--pretty=%ad"]).decode()
        cur_day = None
        for ln in out.splitlines():
            if re.match(r"^\d{4}-\d{2}-\d{2}$", ln.strip()):
                cur_day = ln.strip(); continue
            p = ln.strip()
            if cur_day and any(p.startswith(q) for q in paths):
                days[cur_day] += 1
        return days
    except:
        return {}

# ---------- harvest ----------
repo = Path(".")
series = defaultdict(dict)  # day -> metric dict
found_any = False

# 1) agent_ledger.jsonl (preferred)
ledger = repo / "docs" / "progress" / "agent_ledger.jsonl"
if ledger.exists():
    with ledger.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except:
                continue
            day = iso_date(rec.get("timestamp") or rec.get("time") or rec.get("ts"))
            if day is None: continue
            # proofs_added, coverage, success_rate
            if "proofs_added" in rec:
                add_point(day, series, "proofs_added", rec["proofs_added"])
                found_any = True
            if "coverage" in rec:
                try: add_value(day, series, "coverage", float(rec["coverage"]))
                except: pass
            if "success_rate" in rec:
                try: add_value(day, series, "success_rate", float(rec["success_rate"]))
                except: pass

# 2) progress docs snapshots
for md in [repo/"docs"/"progress.md", repo/"progress.md"]:
    if md.exists():
        cur_day = None
        # naive day context from headings like ## 2025-09-27 or dates in lines
        with md.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                m = re.search(r"(\d{4}-\d{2}-\d{2})", ln)
                if m: cur_day = m.group(1)
                m2 = re.search(r"(proofs_total|proofs|inserted_proofs)\s*:\s*(\d+)", ln, re.I)
                if m2 and cur_day:
                    add_value(cur_day, series, "proofs_total", int(m2.group(2))); found_any = True
                m3 = re.search(r"(coverage)\s*:\s*([0-9.]+)", ln, re.I)
                if m3 and cur_day:
                    try: add_value(cur_day, series, "coverage", float(m3.group(2)))
                    except: pass
                m4 = re.search(r"(success_rate)\s*:\s*([0-9.]+)", ln, re.I)
                if m4 and cur_day:
                    try: add_value(cur_day, series, "success_rate", float(m4.group(2)))
                    except: pass

# 3) artifacts JSONL with inserted_proofs / metrics
arts = list((repo/"artifacts").glob("*.jsonl")) if (repo/"artifacts").exists() else []
for jf in arts:
    try:
        with jf.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                try: rec = json.loads(ln)
                except: continue
                day = iso_date(rec.get("timestamp") or rec.get("date") or rec.get("ts"))
                if day is None: continue
                if "inserted_proofs" in rec:
                    add_point(day, series, "proofs_added", rec["inserted_proofs"]); found_any = True
                # if v1 metrics record has success flag, etc., infer success_rate crudely
                if "successes" in rec and "attempts" in rec:
                    try:
                        sr = float(rec["successes"])/max(1.0,float(rec["attempts"]))
                        add_value(day, series, "success_rate", sr)
                    except: pass
    except: pass

# 4) git heuristic fallback (if no direct counts)
if not found_any:
    est = git_estimate_proofs_per_day()
    for d, c in est.items():
        add_point(d, series, "proofs_added", c)
    if est: found_any = True

# Normalize into ordered days
days = sorted(series.keys())
# derive cumulative proofs: prefer proofs_total snapshots; otherwise cum-sum proofs_added
cum = 0
rows = []
for d in days:
    s = series[d]
    if "proofs_total" in s:
        cum = int(s["proofs_total"])
    else:
        cum += int(s.get("proofs_added", 0))
    row = {
        "date": d,
        "proofs_added": int(s.get("proofs_added", 0)),
        "proofs_total": int(cum),
        "success_rate": (s.get("success_rate", "")),
        "coverage": (s.get("coverage",""))
    }
    rows.append(row)

# compute 7-day rolling proofs
dq = deque()
for r in rows:
    dq.append(r["proofs_added"])
    if len(dq)>7: dq.popleft()
    r["proofs_7d"] = sum(dq)

# write CSV
out = Path("artifacts") / "flywheel_timeseries.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["date","proofs_added","proofs_7d","proofs_total","success_rate","coverage"])
    w.writeheader()
    for r in rows: w.writerow(r)

print(f"[OK] wrote {out} with {len(rows)} rows")
