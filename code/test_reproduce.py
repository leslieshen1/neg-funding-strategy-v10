#!/usr/bin/env python3
"""Reproducibility check: re-runs the headline and compares with the stored reference (results/reference_headline.json).

  python test_reproduce.py                              # public equal-cap headline (needs only the downloaded public data)
  python test_reproduce.py --holdings /path/holdings.xlsx   # additionally rebuilds the PRIVATE inventory-cap headline

The holdings file is read locally, converted to a temporary cap file and deleted afterwards; no quantity is written anywhere.
v8: besides ann / net the reference pins the EPISODE COUNT, the FILL COUNT and the SET OF TRADED COINS (equal cap: the public coin list; private cap:
only the count + a sha256 of the sorted set, no names), so a changed data set / pool / engine cannot slip through on a similar return number.
Also re-runs the all-flags-off legacy engine and compares it with the 20d9522 (v7) numbers.
Exit code 0 = reproduced (ann on S: 0.05 pp; net: 1%; episodes / fills / coin set: exact).  Re-downloading changes nothing in closed history, but
the data window end (config.DATA_END) must be the same as when the reference was written.
"""
import argparse, hashlib, json, os, sys, tempfile, subprocess
import numpy as np, pandas as pd
import config as C
import run_backtest as R

TOL_ANN, TOL_NET = 0.05, 0.01


LEGACY_REF = {"1y": dict(ann_on_S_pct=17.0096, net_wan=1190.6751, episodes=1977), "6m": dict(ann_on_S_pct=11.8405, net_wan=408.7392, episodes=693)}   # 20d9522, equal cap


def coin_set(D, r):
    ep = r["ep"]; idx = sorted(set(int(i) for i in ep[:, 0])) if len(ep) else []
    return [D.fut[i] for i in idx]


def headline(D, cap_mode, cap_file=None, S=7e7, legacy=False):
    ok = R.load_ok(D); cap = R.make_cap(D, ok, S, cap_mode, cap_file); out = {}
    for w in ("1y", "6m"):
        r = D.run(w, S, ok, cap, R.base_params(legacy=legacy), oi_frac=0.10, oi_mode="hour"); m = R.metrics(r); cs = coin_set(D, r)
        out[w] = dict(ann_on_S_pct=m["ann_on_S_pct"], net_wan=m["net_wan"], ann_on_used_pct=m["ann_on_used_pct"], episodes=int(r["episodes"]), fills=int(r["fills"]),
                      n_coins=len(cs), coins_hash=hashlib.sha256(",".join(cs).encode()).hexdigest()[:16])
        if cap_mode == "equal": out[w]["coins"] = cs
    return out


def causality(D, ddir):
    """Structural checks (no reference file): volume basis at t excludes hour t; hourly OI at t is the last snapshot of hour t-1."""
    sq, fq = D.arr[2], D.arr[5]; bad = 0; i, t = 5, 500
    exp = min(np.nansum(sq[t - 24:t, i]), np.nansum(fq[t - 24:t, i])); ok1 = abs(D.dv[t, i] - exp) < 1e-6 * max(1.0, exp)
    f = D.fut[i]; p = ddir / "oi_hourly" / f"{f}.csv"; ok2 = True
    if p.exists():
        d = pd.read_csv(p); d["h"] = pd.to_datetime(d.hour, utc=True); T0 = pd.Timestamp(C.DATA_START, tz="UTC")
        for tt in (100, 2000, 6000):
            hh = T0 + pd.Timedelta(hours=tt - 1); r = d[d.h == hh]
            if len(r): ok2 &= abs(D.oih[tt, i] - r.oi_usdt.iloc[0]) < 1e-6 * max(1.0, r.oi_usdt.iloc[0])
    print(f"[{'OK' if ok1 else 'FAIL'}] volume basis at t = hours t-24..t-1 (min spot/perp)"); print(f"[{'OK' if ok2 else 'FAIL'}] OI at t = last 5-min snapshot of hour t-1")
    return int(not ok1) + int(not ok2)


def check(name, got, ref):
    bad = 0
    for w in ("1y", "6m"):
        g, f = got[w], ref[w]
        da = abs(g["ann_on_S_pct"] - f["ann_on_S_pct"]); dn = abs(g["net_wan"] / f["net_wan"] - 1)
        exact = [k for k in ("episodes", "fills", "n_coins", "coins_hash") if k in f and g.get(k) != f[k]]
        if "coins" in f and g.get("coins") != f["coins"]: exact.append("coin set")
        ok = da <= TOL_ANN and dn <= TOL_NET and not exact; bad += (not ok)
        print(f"[{'OK' if ok else 'FAIL'}] {name} {w}: ann on S {g['ann_on_S_pct']:.2f}% (ref {f['ann_on_S_pct']:.2f}%)  net {g['net_wan']:.1f}万 (ref {f['net_wan']:.1f}万)  "
              f"episodes {g.get('episodes')} (ref {f.get('episodes')}) fills {g.get('fills')} (ref {f.get('fills')}) coins {g.get('n_coins')} (ref {f.get('n_coins')})" + (f"  MISMATCH: {exact}" if exact else ""))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None); ap.add_argument("--holdings", default=None)
    ap.add_argument("--write-reference", action="store_true", help="maintainer: (over)write results/reference_headline.json")
    a = ap.parse_args()
    D = R.Data(C.data_dir(a.data_dir)); ref_path = C.REPO / "results" / "reference_headline.json"
    bad_c = causality(D, C.data_dir(a.data_dir))
    res = {"equal": headline(D, "equal")}
    leg = headline(D, "equal", legacy=True)
    if a.holdings:
        with tempfile.TemporaryDirectory() as td:
            cf = os.path.join(td, "cap.csv")
            subprocess.check_call([sys.executable, str(C.REPO / "make_holdings_cap.py"), "--holdings", a.holdings, "--out", cf,
                                   "--data-dir", str(C.data_dir(a.data_dir))], stdout=subprocess.DEVNULL)
            res["holdings"] = headline(D, "holdings", cf)
    if a.write_reference:
        ref = json.load(open(ref_path)) if ref_path.exists() else {}
        ref.update({k: {w: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for w, v in x.items()} for k, x in res.items()})
        json.dump(ref, open(ref_path, "w"), indent=1); print("wrote", ref_path); return
    ref = json.load(open(ref_path)); bad = bad_c
    bad += check("legacy engine (flags off) vs 20d9522", leg, LEGACY_REF)
    for k, x in res.items(): bad += check(k, x, ref[k])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
