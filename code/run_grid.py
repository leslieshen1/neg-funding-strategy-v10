#!/usr/bin/env python3
"""v8 full grid: cap (A=private inventory, B=equal S/267) x window (1y, 6m) x OI lag (0, 1h) x own-impact k (0, 4, 8 bp per 1% OI) x cost (10, 20, 30, 50bp).
Per run: headline + bootstrap + concentration (net of fees) + quarterly blocks; ex-top5/top10 (ranked by NET pnl, whole portfolio re-simulated) for the
base cost (20bp) rows and for the combined realistic case (OI lag 1h + 30bp + k=4).
Writes results/grid_<cap>.csv, results/grid_quarterly_<cap>.csv, results/grid_extop_<cap>.csv (aggregate only; no per-coin numbers).
  python run_grid.py --cap-mode equal|holdings [--holdings-cap CSV] [--jobs 8]
"""
import argparse, itertools, multiprocessing as mp
import numpy as np, pandas as pd
import config as C
import run_backtest as R

G = {}
COMBINED = (1, 30, 4.0)       # OI lag 1h, 30bp, k=4 : the "combined realistic case"


def one(task):
    w, lag, k, imp = task; D, ok, cap, S = G["D"], G["ok"], G["cap"], G["S"]
    P = R.base_params(imp, k_imp=float(k)); r = D.run(w, S, ok, cap, P, oi_extra_lag_h=lag)
    m = R.metrics(r); m.update(R.bootstrap(r)); m.update(R.concentration(r, D.N)); m.pop("window", None)
    m = dict(window=w, oi_lag_h=lag, k_bp_per_1pct_oi=k, impact_bp=imp, combined_realistic=int((lag, imp, k) == COMBINED), **m)
    q = R.quarterly(r); q.insert(1, "oi_lag_h", lag); q.insert(2, "k_bp_per_1pct_oi", k); q.insert(3, "impact_bp", imp)
    ex = []
    if imp == 20 or (lag, imp, k) == COMBINED:
        pnl = R.coin_pnl(r, D.N, net=True); order = np.argsort(-pnl)
        for kk in (5, 10):
            ok2 = ok.copy(); ok2[order[:kk]] = 0.0
            m2 = R.metrics(D.run(w, S, ok2, cap, P, oi_extra_lag_h=lag))
            ex.append(dict(window=w, oi_lag_h=lag, k_bp_per_1pct_oi=k, impact_bp=imp, test=f"ex_top{kk}_net", net_wan=m2["net_wan"], ann_on_S_pct=m2["ann_on_S_pct"],
                           avg_used_wan=m2["avg_used_wan"], ann_on_used_pct=m2["ann_on_used_pct"], base_ann_on_S_pct=m["ann_on_S_pct"]))
    return m, q, ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-mode", choices=["equal", "holdings"], default="equal"); ap.add_argument("--holdings-cap", default=None)
    ap.add_argument("--S", type=float, default=7e7); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--out-dir", default=str(C.REPO / "results"))
    a = ap.parse_args()
    D = R.Data(C.data_dir(None)); ok = R.load_ok(D); G.update(D=D, ok=ok, cap=R.make_cap(D, ok, a.S, a.cap_mode, a.holdings_cap), S=a.S)
    one(("6m", 0, 0, 20))     # compile / warm up before forking
    tasks = list(itertools.product(("1y", "6m"), (0, 1), (0, 4, 8), (10, 20, 30, 50)))
    with mp.get_context("fork").Pool(a.jobs) as pool: out = pool.map(one, tasks, chunksize=1)
    H = pd.DataFrame([o[0] for o in out]); Q = pd.concat([o[1] for o in out]); X = pd.DataFrame([e for o in out for e in o[2]])
    H.insert(0, "cap_mode", a.cap_mode); Q.insert(0, "cap_mode", a.cap_mode); X.insert(0, "cap_mode", a.cap_mode)
    od = __import__("pathlib").Path(a.out_dir); od.mkdir(exist_ok=True)
    H.to_csv(od / f"grid_{a.cap_mode}.csv", index=False); Q.to_csv(od / f"grid_quarterly_{a.cap_mode}.csv", index=False); X.to_csv(od / f"grid_extop_{a.cap_mode}.csv", index=False)
    print(H[["window", "oi_lag_h", "k_bp_per_1pct_oi", "impact_bp", "ann_on_S_pct", "net_wan", "avg_used_wan", "episodes"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
