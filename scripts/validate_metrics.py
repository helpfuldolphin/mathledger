#!/usr/bin/env python3
import argparse, csv, os, sys, statistics as stats

def load_csv(path):
    rows=[]
    if not os.path.exists(path): return rows
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def floatget(d,k,default=None):
    v=d.get(k)
    if v is None or v=="":
        return default
    try: return float(v)
    except: return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--csv", default="artifacts/wpv5", help="dir with baseline_runs.csv and guided_runs.csv")
    ap.add_argument("--guidance-gate", action="store_true", help="check guided >= 1.25x baseline on PL-2 if present, else overall")
    ap.add_argument("--target-pps", type=float, default=0.005, help="ScaleA target proofs/sec (default ~18/h)")
    args = ap.parse_args()

    bl = load_csv(os.path.join(args.csv, "baseline_runs.csv"))
    gd = load_csv(os.path.join(args.csv, "guided_runs.csv"))
    if not bl:
        print("FAIL: no baseline_runs.csv rows", file=sys.stderr); sys.exit(2)
    if not gd:
        print("FAIL: no guided_runs.csv rows", file=sys.stderr); sys.exit(2)

    def filt(rows, slice_name=None):
        out=[]
        for r in rows:
            if slice_name and r.get("slice","").strip().upper()!=slice_name:
                continue
            pph = floatget(r, "proofs_per_hour", None)
            if pph is None:
                # fallback: per-sec field
                pps = floatget(r, "proofs_per_sec", None)
                if pps is not None: pph = pps*3600.0
            if pph is not None: out.append(pph)
        return out

    def gate_ratio(slice_name=None, thresh=1.25):
        b = filt(bl, slice_name); g = filt(gd, slice_name)
        if not b or not g: return None
        rb = stats.mean(b); rg = stats.mean(g)
        ratio = rg/max(rb,1e-9)
        ok = ratio >= thresh
        label = slice_name or "ALL"
        print(f"Guidance Gate [{label}]: guided={rg:.2f}/h baseline={rb:.2f}/h ratio={ratio:.3f} -> {'PASS' if ok else 'FAIL'} (threshold {thresh}x)")
        return ok

    rc = 0
    if args.guidance_gate:
        ok_pl2 = gate_ratio("PL-2", 1.25)
        if ok_pl2 is None:
            # fallback overall
            ok_all = gate_ratio(None, 1.25)
            rc |= 0 if ok_all else 1
        else:
            rc |= 0 if ok_pl2 else 1

    # Optional ScaleA from CSVs (treat mean per-hour as proxy)
    all_bl = filt(bl); all_gd = filt(gd)
    if all_bl and all_gd:
        mean_pph = stats.mean(all_gd)  # optimistic check against target
        mean_pps = mean_pph/3600.0
        ok_scaleA = (mean_pps >= args.target_pps)
        print(f"ScaleA: mean_pps={mean_pps:.6f} (target {args.target_pps}) -> {'PASS' if ok_scaleA else 'FAIL'}")
        rc |= 0 if ok_scaleA else 1

    sys.exit(rc)

if __name__ == "__main__":
    main()
