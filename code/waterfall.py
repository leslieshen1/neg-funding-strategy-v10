#!/usr/bin/env python3
"""Marginal effect of every v8 engine fix, cumulative, starting from the 20d9522 (v7) engine (all flags off = reproduces v7 exactly).
Order (each row adds one fix to all the previous ones):  see run_backtest.FIX_ORDER.  OI lag 0 (hourly OI of hour t-1), 20bp, k=0, S=7000万.
Writes results/waterfall_<cap>.csv (aggregate numbers only).   python waterfall.py --cap-mode equal|holdings [--holdings-cap CSV]
"""
import argparse
import numpy as np, pandas as pd
import config as C
import run_backtest as R

LABEL = dict(f_exitlag="1 exit/derisk decision uses previous-hour close (no same-hour close look-ahead)", f_markfund="2 funding valued at markPrice at settlement",
             f_budget="3 ONE shared participation budget per coin-hour (entry+add+exit+derisk)",
             f_prior="4 budget & depth from the PRIOR hour's volume", f_hardcap="5 10% volume / 10% OI as hard caps (OI on perp notional)",
             f_exitpov="6 window-end / timeout exits obey participation; missing bar defers",
             f_exitstate="7 exit PnL state after partial derisk (whole episode)", f_mtmcap="8 capital use reported at mark-to-market (no PnL effect)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-mode", choices=["equal", "holdings"], default="equal"); ap.add_argument("--holdings-cap", default=None)
    ap.add_argument("--S", type=float, default=7e7); ap.add_argument("--out-dir", default=str(C.REPO / "results"))
    a = ap.parse_args()
    D = R.Data(C.data_dir(None)); ok = R.load_ok(D); cap = R.make_cap(D, ok, a.S, a.cap_mode, a.holdings_cap)
    rows = []; v7 = {}
    for w in ("1y", "6m"):
        p = R.base_params(legacy=True); prev = None; pass
        for i in range(-1, len(R.FIX_ORDER)):
            if i >= 0: p[R.FIX_ORDER[i]] = 1
            r = D.run(w, a.S, ok, cap, p); m = R.metrics(r)
            rows.append(dict(window=w, step="0 20d9522 (v7 engine)" if i < 0 else LABEL[R.FIX_ORDER[i]], ann_on_S_pct=m["ann_on_S_pct"],
                             delta_pp=np.nan if prev is None else m["ann_on_S_pct"] - prev, net_wan=m["net_wan"], avg_used_wan=m["avg_used_wan"], episodes=m["episodes"], fills=m["fills"],
                             coin_hours_over_vol_cap=m["hours_over_vol_cap"], coin_hours_over_oi_cap=m["hours_over_oi_cap"], coin_hours_held=m["coin_hours_held"],
                             fills_over_prior_pov=m["fills_over_prior_pov"], residual_end_wan=m["residual_end_wan"], funding_events_no_mark=m["funding_events_no_mark"]))
            prev = m["ann_on_S_pct"]
            if i < 0: v7[w] = prev
        hl = R.base_params(); base = R.metrics(D.run(w, a.S, ok, cap, hl))["ann_on_S_pct"]       # headline stack
        for f in R.FIX_ORDER:                                    # (b) each fix ALONE on the v7 engine, (c) headline stack WITHOUT that fix
            q = R.base_params(legacy=True); q[f] = 1; ma = R.metrics(D.run(w, a.S, ok, cap, q))["ann_on_S_pct"]
            q2 = dict(hl); q2[f] = 0; mo = R.metrics(D.run(w, a.S, ok, cap, q2))["ann_on_S_pct"]
            rows.append(dict(window=w, step="[alone] " + LABEL[f], ann_on_S_pct=ma, delta_pp=ma - v7[w]))
            rows.append(dict(window=w, step="[leave-one-out: headline minus this fix = fix contribution] " + LABEL[f], ann_on_S_pct=mo, delta_pp=base - mo))
    df = pd.DataFrame(rows).round(2); out = __import__("pathlib").Path(a.out_dir); out.mkdir(exist_ok=True)
    df.to_csv(out / f"waterfall_{a.cap_mode}.csv", index=False); print(df[["window", "step", "ann_on_S_pct", "delta_pp", "net_wan", "avg_used_wan", "episodes", "fills"]].to_string(index=False))


if __name__ == "__main__":
    main()
