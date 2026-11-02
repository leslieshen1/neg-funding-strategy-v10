#!/usr/bin/env python3
"""Capacity curve: re-run the (unchanged) v8 engine for total capital S in a grid, scenarios A (private inventory cap, min(cap,S) per coin) and
B (public equal cap S/267), windows 1y/6m, cases optimistic (OI lag0, 20bp, k=0) and realistic (OI lag1h, 30bp, k=4).
Aggregate only: no per-coin inventory or cap values are written.   python capacity_curve.py [--holdings-cap CSV] [--jobs 8]
Writes results/capacity_curve.csv and results/capacity_curve.png."""
import argparse, itertools, multiprocessing as mp
import numpy as np, pandas as pd
import config as C
import run_backtest as R

SGRID = [1e6, 2e6, 3e6, 5e6, 7.5e6, 1e7, 1.5e7, 2e7, 3e7, 4e7, 5e7, 7e7]
CASES = {"optimistic": dict(lag=0, imp=20, k=0.0), "realistic": dict(lag=1, imp=30, k=4.0)}
G = {}


def one(t):
    scen, S, w, case = t; D, ok = G["D"], G["ok"]; c = CASES[case]
    cap = R.make_cap(D, ok, S, "equal" if scen == "B" else "holdings", None if scen == "B" else G["hc"])
    r = D.run(w, S, ok, cap, R.base_params(c["imp"], k_imp=c["k"], **G.get("mtm", {})), oi_extra_lag_h=c["lag"]); m = R.metrics(r)
    y = m["years"]
    return dict(scenario=scen, case=case, window=w, S_wan=S / 1e4, net_wan=m["net_wan"], net_usdt_per_yr=m["net_wan"] * 1e4 / y,
                ann_on_S_pct=m["ann_on_S_pct"], ann_on_avg_mtm_pct=m["net_wan"] * 1e4 / y / (m["avg_used_mtm_wan"] * 1e4) * 100,
                ann_on_avg_cost_used_pct=m["ann_on_used_pct"], avg_used_mtm_wan=m["avg_used_mtm_wan"], peak_used_mtm_wan=m["peak_used_mtm_wan"],
                avg_used_mtm_pct_of_S=m["avg_used_mtm_wan"] / (S / 1e4) * 100, peak_used_mtm_pct_of_S=m["peak_used_mtm_pct_of_S"],
                avg_used_cost_wan=m["avg_used_cost_wan"], peak_used_cost_wan=m["peak_used_cost_wan"], episodes=m["episodes"],
                peak_used_cost_pct_of_S=m["peak_used_cost_wan"] / (S / 1e4) * 100, peak_mtm_over_S=m["peak_used_mtm_pct_of_S"], derisk_traded_wan=m["derisk_traded_wan"], derisk_fee_cost_wan=m["derisk_fee_cost_wan"],
                hours_mtm_over_S=m["hours_mtm_over_S"], hours_derisk_active=m["hours_derisk_active"], hours_trim_short=m["hours_trim_short"], max_trim_short_ratio=m["max_trim_short_ratio"], hours_over_vol_cap=m["hours_over_vol_cap"], hours_over_oi_cap=m["hours_over_oi_cap"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--holdings-cap", default=str(C.data_dir(None) / "holdings_cap.csv")); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--mtmderisk", action="store_true", help="f_mtmderisk=1: enforce MTM usage <= S (pro-rata derisk + no new entries/adds) -> results/capacity_curve_mtmderisk.*")
    ap.add_argument("--mtmcap", action="store_true", help="f_mtmcap=2: capital room for entries/adds measured at market value -> results/capacity_curve_mtmcap.*")
    a = ap.parse_args(); G["mtm"] = {"f_mtmcap": 2} if a.mtmcap else ({"f_mtmderisk": 1} if a.mtmderisk else {}); sfx = "_mtmcap" if a.mtmcap else ("_mtmderisk" if a.mtmderisk else "")
    D = R.Data(C.data_dir(None)); G.update(D=D, ok=R.load_ok(D), hc=a.holdings_cap)
    one(("B", 1e6, "6m", "optimistic"))
    tasks = list(itertools.product(("A", "B"), SGRID, ("1y", "6m"), CASES))
    with mp.get_context("fork").Pool(a.jobs) as pool: rows = pool.map(one, tasks, chunksize=1)
    df = pd.DataFrame(rows).sort_values(["scenario", "case", "window", "S_wan"]); df.to_csv(C.REPO / "results" / f"capacity_curve{sfx}.csv", index=False)
    print(df[["scenario", "case", "window", "S_wan", "ann_on_S_pct", "ann_on_avg_mtm_pct", "net_usdt_per_yr", "avg_used_mtm_wan", "peak_used_mtm_wan"]].round(2).to_string(index=False))
    sat = []
    for (sc, cs, w), g in df.groupby(["scenario", "case", "window"]):
        g = g.sort_values("S_wan"); p70 = g[g.S_wan == 7000].net_usdt_per_yr.iloc[0]; base = g.ann_on_S_pct.iloc[0]
        sat.append(dict(scenario=sc, case=cs, window=w, S_for_90pct_of_70M_pnl_M=g[g.net_usdt_per_yr >= .9 * p70].S_wan.min() / 100,
                        S_for_95pct_M=g[g.net_usdt_per_yr >= .95 * p70].S_wan.min() / 100, S_at_max_pnl_M=g.loc[g.net_usdt_per_yr.idxmax(), "S_wan"] / 100,
                        maxpnl_over_pnl70=g.net_usdt_per_yr.max() / p70, ann_flat_within_1pp_up_to_S_M=g[(base - g.ann_on_S_pct) <= 1].S_wan.max() / 100, ann_at_1M=base))
    pd.DataFrame(sat).to_csv(C.REPO / "results" / f"capacity_curve_saturation{sfx}.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for j, w in enumerate(("1y", "6m")):
        for scen, cs, ls in itertools.product(("A", "B"), CASES, ("-",)):
            d = df[(df.scenario == scen) & (df.case == cs) & (df.window == w)]
            kw = dict(marker="o", ms=4, color={"A": "tab:blue", "B": "tab:orange"}[scen], ls="-" if cs == "optimistic" else "--", label=f"{scen} {cs}")
            ax[0, j].plot(d.S_wan / 100, d.ann_on_S_pct, **kw); ax[1, j].plot(d.S_wan / 100, d.net_usdt_per_yr / 1e4, **kw)
        ax[0, j].set_title(f"{w}: annualized return on S (%)"); ax[1, j].set_title(f"{w}: annualized net PnL (10k USDT/yr)")
        ax[1, j].set_xlabel("total capital S (million USDT)")
        for i in (0, 1): ax[i, j].set_xscale("log"); ax[i, j].grid(alpha=.3); ax[i, j].set_xticks([1, 2, 3, 5, 10, 20, 40, 70]); ax[i, j].set_xticklabels([1, 2, 3, 5, 10, 20, 40, 70])
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(C.REPO / "results" / f"capacity_curve{sfx}.png", dpi=130)


if __name__ == "__main__":
    main()
