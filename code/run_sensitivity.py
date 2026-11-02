#!/usr/bin/env python3
"""Sensitivity + bridge tables (1y and 6m).  Writes results/sensitivity_<cap>.csv and results/bridge_<cap>.csv.

bridge  : starts from the OLD rule set (replicated on this repo's public data pipeline) and applies each remediation one
          at a time, so every change has a measured cost.  Order matters (sequential, cumulative).
sens    : one-at-a-time variations around the new base case (20bp impact, no basis filter, 10% vol, 10% OI of the previous completed hour).
Nothing here is tuned: the base case is fixed beforehand and the variations are shown, not selected.
"""
import argparse, sys
import numpy as np, pandas as pd
import config as C
import run_backtest as R


def row(D, w, S, ok, cap, params, label, group, hair=1.0, **kw):
    r = D.run(w, S, ok, cap, params, **kw); m = R.metrics(r, hair)
    return dict(group=group, case=label, window=w, net_wan=round(m["net_wan"], 1), funding_wan=round(m["funding_wan"], 1),
                basis_gross_wan=round(m["basis_gross_wan"], 1), spread_impact_wan=round(m["spread_impact_wan"], 1), fees_wan=round(m["fees_wan"], 1),
                ann_on_S_pct=round(m["ann_on_S_pct"], 2), ann_on_used_pct=round(m["ann_on_used_pct"], 1),
                avg_used_wan=round(m["avg_used_wan"], 0), episodes=m["episodes"],
                sens_idle5_ann_on_S_pct=round(m["sens_idle5_ann_on_S_pct"], 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None); ap.add_argument("--S", type=float, default=7e7)
    ap.add_argument("--cap-mode", choices=["equal", "holdings"], default="equal"); ap.add_argument("--holdings-cap", default=None)
    ap.add_argument("--out-dir", default=str(C.REPO / "results")); ap.add_argument("--no-bridge", action="store_true")
    a = ap.parse_args()
    ddir = C.data_dir(a.data_dir)
    D = R.Data(ddir, "dataset"); ok = R.load_ok(D); S = a.S
    cap = R.make_cap(D, ok, S, a.cap_mode, a.holdings_cap)
    B = R.base_params()
    rows = []
    for w in ("1y", "6m"):
        rows.append(row(D, w, S, ok, cap, B, "BASE: impact20 / no basis filter / vol10% (t-1) / OI10% (prev hour) / de-risk on", "base"))
        for bp in (10, 20, 30):
            rows.append(row(D, w, S, ok, cap, R.base_params(bp), f"impact {bp}bp", "impact"))
        for b in (None, 0, -20, -38, -60):
            rows.append(row(D, w, S, ok, cap, R.base_params(basis_bp=b), "no basis filter" if b is None else f"entry basis <= {b}bp", "basis_filter"))
        for f in (0, 0.02, 0.05, 0.10, 0.20):
            rows.append(row(D, w, S, ok, cap, B, "OI limit off" if f == 0 else f"OI limit {int(f*100)}% of prev-hour OI", "oi_limit", oi_frac=f))
        rows.append(row(D, w, S, ok, cap, B, "OI limit 10% of previous-UTC-day OI (v4 daily, for comparison)", "oi_granularity", oi_mode="lag"))
        for k in (6, 12, 24):
            rows.append(row(D, w, S, ok, cap, B, f"hourly OI made {k}h staler (diagnostic: OI of hour t-1-{k})", "oi_staleness", oi_extra_lag_h=k))
        for f in (0, 0.05, 0.10, 0.20):
            rows.append(row(D, w, S, ok, cap, R.base_params(vol_frac=f), "volume limit off" if f == 0 else f"volume limit {int(f*100)}% of trailing-24h", "vol_limit"))
        rows.append(row(D, w, S, ok, cap, R.base_params(derisk=0), "no de-risk while holding", "derisk"))
        for e in (0.0005, 0.001, 0.002, 0.005):
            rows.append(row(D, w, S, ok, cap, R.base_params(entry_fav_n=e), f"entry 8h-rate magnitude > {e*1e4:.0f}bp", "entry_threshold"))
        for wm in (0, 24, 168):
            rows.append(row(D, w, S, ok, cap, R.base_params(wait_max_n=wm), f"max wait after rate turns {wm}h", "exit_wait"))
        for h in (1.0, 0.8, 0.6):
            rows.append(row(D, w, S, ok, cap, B, f"funding x{h} (illustration only: no evidence for any haircut)", "haircut_illustration", hair=h))
    S_df = pd.DataFrame(rows)
    out = __import__("pathlib").Path(a.out_dir); out.mkdir(exist_ok=True)
    S_df.to_csv(out / f"sensitivity_{a.cap_mode}.csv", index=False)
    print(S_df[S_df.window == "1y"][["group", "case", "ann_on_S_pct", "ann_on_used_pct", "avg_used_wan", "net_wan"]].to_string(index=False))
    if a.no_bridge: return
    # ---------------- bridge (1y) ----------------
    Dm = R.Data(ddir, "dataset_median")
    L = R.base_params(10, -38, legacy=True, derisk=0)                      # old rule set
    steps = []
    def add(label, r, hair, note):
        m = R.metrics(r, hair); steps.append(dict(step=label, note=note, ann_no_idle_pct=round(m["ann_on_S_pct"], 2),
                  ann_idle5_pct=round(m["sens_idle5_ann_on_S_pct"], 2), net_wan=round(m["net_wan"], 1), avg_used_wan=round(m["avg_used_wan"], 0)))
    w = "1y"
    add("0 old rules replicated", Dm.run(w, S, ok, cap, L, oi_frac=0.10, oi_mode="static", dv=Dm.dv_sameday()), 0.8,
        "same-day volume, full-sample median OI, no de-risk, 10bp, -38bp basis filter, funding x0.8, median funding interval")
    add("1 per-event funding interval", D.run(w, S, ok, cap, L, oi_frac=0.10, oi_mode="static", dv=D.dv_sameday()), 0.8, "interval per event instead of one median")
    add("2 remove funding x0.8", D.run(w, S, ok, cap, L, oi_frac=0.10, oi_mode="static", dv=D.dv_sameday()), 1.0, "raw realised funding")
    add("3 causal volume (t-1 trailing 24h), dv==0 = no room", D.run(w, S, ok, cap, L, oi_frac=0.10, oi_mode="static"), 1.0, "entry/add only, as before")
    add("4 + de-risk while holding (volume)", D.run(w, S, ok, cap, R.base_params(10, -38, legacy=True), oi_frac=0.10, oi_mode="static"), 1.0, "sell down when position > 10% of trailing volume")
    add("5 OI: full-sample median -> previous-day OI, enforced while holding", D.run(w, S, ok, cap, R.base_params(10, -38, legacy=True), oi_frac=0.10, oi_mode="lag"), 1.0, "")
    add("6 impact 10bp -> 20bp", D.run(w, S, ok, cap, R.base_params(20, -38, legacy=True), oi_frac=0.10, oi_mode="lag"), 1.0, "")
    add("7 drop the -38bp basis filter  (= v4 base, daily OI)", D.run(w, S, ok, cap, R.base_params(20, None, legacy=True), oi_frac=0.10, oi_mode="lag"), 1.0, "idle yield column is sensitivity only")
    add("8 OI: previous-UTC-day -> previous completed hour  (= v7 base, 20d9522)", D.run(w, S, ok, cap, R.base_params(20, None, legacy=True), oi_frac=0.10, oi_mode="hour"), 1.0, "hourly 5-min-snapshot OI, last snapshot of hour t-1, checked every hour at entry and while holding")
    add("9 v8 engine fixes (see waterfall_*.csv for the split)  (= NEW BASE)", D.run(w, S, ok, cap, R.base_params(20, None), oi_frac=0.10, oi_mode="hour"), 1.0, "exit lag, markPrice funding, shared prior-hour budget, hard caps, exit POV, exit state; k=0")
    br = pd.DataFrame(steps); br["delta_no_idle_pp"] = br.ann_no_idle_pct.diff().round(2)
    br.to_csv(out / f"bridge_{a.cap_mode}.csv", index=False)
    print(br.to_string(index=False))


if __name__ == "__main__":
    main()
