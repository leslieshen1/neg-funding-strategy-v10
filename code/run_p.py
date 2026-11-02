"""P = large-coin positive-funding strategy (renamed from 'B' so it is not confused with the public per-coin-cap scenario B). v10 r2: isolated OI=0 (feed glitch, <=3h) forward-filled,
then the whole thing (rule grid on the TUNE window, main table, cost sensitivity) is re-run. No k_imp (own-funding impact only exists for the negative leg)."""
import os, itertools, multiprocessing as mp
os.environ["NFB_DATA_DIR"] = "/workspace/nfb-data-big"
from v10lib import *
D = R.Data(C.data_dir(None)); N_FILLED = fix_oi_zero(D)
SET4 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]; SETS = (("Set4", SET4), ("Set10", list(D.fut)))
SG = [1e6, 2e6, 5e6, 1e7, 2e7, 3e7, 5e7, 7e7]
def setmask(names): return np.array([1.0 if f in names else 0.0 for f in D.fut])
def runP(S, names, case, entry=-1.0, exit_=-1.0, wait=0, imp=None):
    om = setmask(names); okn = np.zeros(D.N); c = CASES[case]
    p = R.base_params(c["imp"] if imp is None else imp, k_imp=0.0, mirror_on=1, pov_m=0.20, xmax_m=0.02, dfrac_m=0.75, entry_fav_m=entry, exit_fav_m=exit_, wait_max_m=wait)
    return run_cfg(D, S, okn, R.CapSpec(np.zeros(D.N)), p, c["lag"], ok_m=om, cap_m=om * S / om.sum())
def _g(t):
    nm, case, ef, xf, wm = t; names = dict(SETS)[nm]; r = runP(1e7, names, case, ef, xf, wm); a = r["1y"]
    return dict(set=nm, case=case, entry=ef, exit=xf, wait=wm, tune=a["tune_ann_pct"], test=a["test_ann_pct"], y1=a["ann_pct"], m6=r["6m"]["ann_pct"], net_wan=a["net_wan"], ep=a["episodes"],
                avg_used=a["avg_used_mtm_wan"], fund=a["funding_wan"], cost=a["cost_wan"])
def _m(t):
    nm, case, S, rule = t; r = runP(S, dict(SETS)[nm], case, *rule); a = r["1y"]
    return dict(part="P", set=nm, case=case, S=S / 1e6, entry=rule[0], exit=rule[1], wait=rule[2], tune_ann=a["tune_ann_pct"], test_ann=a["test_ann_pct"], y1_ann=a["ann_pct"], m6_ann=r["6m"]["ann_pct"],
                net_wan=a["net_wan"], tune_net_wan=a["tune_net_wan"], test_net_wan=a["test_net_wan"], m6_net_wan=r["6m"]["net_wan"], peak_pct_of_S=a["peak_pct_of_S"], avg_used_wan=a["avg_used_mtm_wan"],
                fund_wan=a["funding_wan"], basis_wan=a["basis_wan"], cost_wan=a["cost_wan"], fees_wan=a["fees_wan"])
def _s(t):
    nm, imp = t; r = runP(1e7, dict(SETS)[nm], "realistic", *(RULE[nm] or (-1.0, -1.0, 0)), imp=imp); a = r["1y"]
    return dict(set=nm, impact_bp=imp, y1_ann=a["ann_pct"], tune_ann=a["tune_ann_pct"], test_ann=a["test_ann_pct"], m6=r["6m"]["ann_pct"], net_wan=a["net_wan"])
RULE = {}
if __name__ == "__main__":
    print("OI=0 coin-hours filled in big set:", N_FILLED)
    ef_ = (-1, 0.0, 0.00002, 0.00003, 0.00005, 0.00007, 0.0001, 0.0002); xf_ = (-1, -0.00005, -0.00002, 0.0, 0.00002, 0.00005)
    tasks = [(nm, case, ef, xf, wm) for nm, _ in SETS for case in ("realistic", "optimistic") for ef, xf, wm in itertools.product(ef_, xf_, (0, 72, 168)) if not (xf >= ef and ef >= 0)]
    with mp.get_context("fork").Pool(8) as p: rows = p.map(_g, tasks, chunksize=4)
    g = pd.DataFrame(rows); g.to_csv("../out/p_grid.csv", index=False)
    for nm, _ in SETS:
        x = g[(g.set == nm) & (g.case == "realistic")].sort_values("tune", ascending=False); print(nm, "tune-best realistic:"); print(x.head(5).round(3).to_string(index=False))
        b = x.iloc[0]; RULE[nm] = (float(b.entry), float(b.exit), int(b.wait)) if b.tune > 0 else None
        print(nm, "selected rule (None = no trade, tune-best <= 0):", RULE[nm])
    tasks = [(nm, case, S, RULE[nm] if RULE[nm] else (-1.0, -1.0, 0)) for nm, _ in SETS for case in ("realistic", "optimistic") for S in SG]
    with mp.get_context("fork").Pool(8) as p: rows = p.map(_m, tasks)
    m = pd.DataFrame(rows); m.to_csv("../out/p_main.csv", index=False); print(m[["set", "case", "S", "entry", "exit", "wait", "tune_ann", "test_ann", "y1_ann", "m6_ann", "net_wan"]].round(3).to_string(index=False))
    with mp.get_context("fork").Pool(8) as p: s = pd.DataFrame(p.map(_s, [(nm, imp) for nm, _ in SETS for imp in (30, 20, 10, 5, 2, 0)]))
    s.to_csv("../out/p_sens.csv", index=False); print(s.round(3).to_string(index=False))
