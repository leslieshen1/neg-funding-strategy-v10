"""v10 r2 (after two independent reviews): negative-funding leg. Headline = v9.1 + A5 (pov=0.03) + A3 (same-hour de-risk).
Re-runs v9.1 / A3 / A5 / A5+A3 / old bundle A1+A3+A5 over the whole capital grid (A and B cap scenarios, realistic & optimistic) with tune/test/6m slices,
mechanism diagnostics (turnover, fills, episodes, cost, funding), monthly PnL (incl. MTM) and the end-of-window forced-exit cost.
Also: sqrt-impact sensitivity and the OI=0 glitch-fix sensitivity (A/B scenarios)."""
import multiprocessing as mp, sys
from v10lib import *
CFG = {"v9.1": {}, "A3 同小时砍回": dict(derisk_force=1), "A5 拆单(pov=0.03)": dict(pov=0.03), "A5+A3": dict(pov=0.03, derisk_force=1),
       "旧捆绑 A1+A3+A5": dict(n_streak=3, big_thr=2e7, derisk_force=1, pov=0.03)}
G = {}
MB = [str(x)[:10] for x in pd.date_range("2025-11-01", "2026-10-01", freq="MS")]      # month starts after the window start
def month_nets(cum):
    a, _ = C.WINDOWS["1y"]; h0 = R.hidx(a); H = len(cum)
    edges = [0] + [R.hidx(m) - h0 for m in MB] + [H]; out = {}
    names = ["2025-10"] + [m[:7] for m in MB]
    for i, nm in enumerate(names): out[f"mo_{nm}"] = (cum[edges[i + 1] - 1] - (cum[edges[i] - 1] if edges[i] > 0 else 0.0)) / 1e4
    return out
def one(t):
    scen, case, S, name = t[:4]; extra = t[4] if len(t) > 4 else {}
    D, ok = G["D"], G["ok"]; c = CASES[case]; kw = dict(CFG[name]); kw.update(extra.get("params", {}))
    cap = R.make_cap(D, ok, S, "equal") if scen == "B" else R.make_cap(D, ok, S, "holdings", G["hc"])
    p = R.base_params(c["imp"], k_imp=c["k"], **kw)
    r = D.run("1y", S, ok, cap, p, oi_extra_lag_h=c["lag"]); m = R.metrics(r); sl = slice_metrics(r, S)
    r6 = D.run("6m", S, ok, cap, p, oi_extra_lag_h=c["lag"]); m6 = R.metrics(r6)
    row = dict(scen=scen, case=case, S_wan=S / 1e4, cfg=name, tag=extra.get("tag", ""), net_wan=m["net_wan"], ann_pct=m["ann_on_S_pct"], avg_used_mtm_wan=m["avg_used_mtm_wan"],
               peak_pct_of_S=m["peak_used_mtm_pct_of_S"], episodes=r["episodes"], fills=r["fills"], traded_x_S=r["traded"] / S, funding_wan=m["funding_wan"],
               cost_wan=m["spread_impact_wan"], fees_wan=m["fees_wan"], resid_end_wan=m["residual_end_wan"], resid_liq_cost_wan=m["residual_liq_cost_wan"],
               **{k: sl[k] for k in sl}, m6_ann_pct=m6["ann_on_S_pct"], m6_net_wan=m6["net_wan"])
    row.update(month_nets(r["cum"])); return row
if __name__ == "__main__":
    G["D"] = R.Data(C.data_dir(None)); G["ok"] = R.load_ok(G["D"]); G["hc"] = os.environ["NFB_HOLDINGS_CAP"]
    one(("B", "realistic", 1e6, "v9.1"))
    tasks = [(s, c, S, n) for s in ("A", "B") for c in CASES for S in SGRID for n in CFG]
    with mp.get_context("fork").Pool(8) as p: rows = p.map(one, tasks, chunksize=4)
    df = pd.DataFrame(rows); df.to_csv("../out/a_final.csv", index=False)
    x = df[(df.S_wan == 7000)]; print(x[["scen", "case", "cfg", "ann_pct", "tune_ann_pct", "test_ann_pct", "m6_ann_pct", "net_wan", "episodes", "fills", "traded_x_S", "cost_wan", "funding_wan"]].round(2).to_string(index=False))
    # --- sensitivity 1: sqrt impact (replaces the fixed-bp impact; lag1, k=4 as realistic), coefficient NOT calibrated
    srows = []
    ts = [("A", "realistic", 7e7, n, dict(params=dict(sqrt_y=y), tag=f"sqrt_y={y}")) for n in ("v9.1", "A5+A3") for y in (0.5, 1.0)]
    with mp.get_context("fork").Pool(4) as p: srows = p.map(one, ts)
    pd.DataFrame(srows).to_csv("../out/a_sqrt_sens.csv", index=False)
    print(pd.DataFrame(srows)[["scen", "cfg", "tag", "ann_pct", "tune_ann_pct", "test_ann_pct"]].round(2).to_string(index=False))
    # --- sensitivity 2: OI=0 glitch fix applied to the A/B pool (70M, 10M)
    n = fix_oi_zero(G["D"]); print("OI zero coin-hours filled:", n)
    fr = [one((s, "realistic", S, nm, dict(tag="oifix"))) for s in ("A", "B") for S in (7e7, 1e7) for nm in ("v9.1", "A5+A3")]
    fr += [one((s, "optimistic", 7e7, nm, dict(tag="oifix"))) for s in ("A", "B") for nm in ("v9.1", "A5+A3")]
    pd.DataFrame(fr).to_csv("../out/a_oifix_sens.csv", index=False)
    print(pd.DataFrame(fr)[["scen", "case", "S_wan", "cfg", "ann_pct", "tune_ann_pct", "test_ann_pct"]].round(2).to_string(index=False))
