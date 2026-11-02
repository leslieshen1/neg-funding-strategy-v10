"""A4: fix the peak-position and binding-constraint columns of weekly_2026-11-01/coin_table_full.csv & summary_per_coin.csv.
Bug 1: the old table printed the peak at COST basis (price at entry); the caps (10% OI, 10% volume) bind on the position's MARKET value, so a coin that doubled in price looked ~20% smaller
       (LSK: 116万 cost vs 143万 at market, realistic).  Fix: peak = peak_mtm (market value).
Bug 2: 'bound by' compared that peak with the AVERAGE cap over the whole holding span.  Fix: compare with the cap IN FORCE IN THE HOUR OF THE PEAK (10% x lagged OI, 10% x trailing-24h volume).
Engine and data unchanged (weekly_2026-11-01/data, engine c864f47)."""
import sys, os
W = "/workspace/funding-strategy/weekly_2026-11-01"
sys.path.insert(0, W + "/repo"); os.chdir(W + "/repo"); os.environ["NFB_DATA_DIR"] = W + "/data"
import numpy as np, pandas as pd
import config as C, run_backtest as R
ddir = C.data_dir(None); SC = {"realistic": dict(lag=1, imp=30, k=4.0), "optimistic": dict(lag=0, imp=20, k=0.0)}
old = pd.read_csv(W + "/coin_table_full.csv"); rows = []
for name in ("dataset", "dataset_proxy"):
    D = R.Data(ddir, name); a, b = C.WINDOWS["wk"]; h0 = R.hidx(a)
    for i, f in enumerate(D.fut):
        row = dict(fut=f, spot_real=(name == "dataset"))
        for scn, p in SC.items():
            ok = np.zeros(D.N); ok[i] = 1
            r = D.run("wk", 1e9, ok, R.CapSpec(np.where(ok > 0, 1e9, 0.0)), R.base_params(p["imp"], k_imp=p["k"], wind_down_h=1), oi_extra_lag_h=p["lag"])
            u = r["used"]; d = r["diag"]; pk = float(u.max())
            row[f"{scn}_peak_cost_basis"] = d[12]; row[f"{scn}_peak_mtm"] = pk
            if pk > 0:
                t = h0 + int(u.argmax()); oicap = 0.1 * D.oiv_for("hour", p["lag"])[t, i]; volcap = 0.1 * D.dv[t, i]
                row[f"{scn}_peak_hour_utc"] = str((R.T0 + pd.Timedelta(hours=t)))[:16]; row[f"{scn}_oi_cap_at_peak"] = oicap; row[f"{scn}_vol_cap_at_peak"] = volcap
                row[f"{scn}_peak_over_oicap"] = pk / oicap if oicap > 0 else np.nan; row[f"{scn}_peak_over_volcap"] = pk / volcap if volcap > 0 else np.nan
        rows.append(row)
T = pd.DataFrame(rows)
def bind(r, scn):
    pk = r[f"{scn}_peak_mtm"]
    if not pk > 0: return "未入场"
    o, v = r[f"{scn}_peak_over_oicap"], r[f"{scn}_peak_over_volcap"]
    if max(o, v) >= 0.90: return "OI上限(10%OI)" if o >= v else "成交量上限(10%×24h量)"
    if scn == "realistic" and r["optimistic_peak_mtm"] > 0 and pk < 0.85 * r["optimistic_peak_mtm"]: return "自冲击k(费率被推高)"
    return "信号短/参与率爬坡(未触上限)"
T["bind_opt"] = [bind(r, "optimistic") for _, r in T.iterrows()]; T["bind_real"] = [bind(r, "realistic") for _, r in T.iterrows()]
cap_eq = 7e7 / 267
T = T.merge(old[["fut", "spot_real", "optimistic_peak_equalcap", "realistic_peak_equalcap", "optimistic_peak_pos", "realistic_peak_pos"]].rename(columns={"optimistic_peak_pos": "old_opt_peak(cost)", "realistic_peak_pos": "old_real_peak(cost)"}), on=["fut", "spot_real"], how="left")
T["bind_equal"] = np.where((T.optimistic_peak_equalcap >= 0.99 * cap_eq) & (T.optimistic_peak_mtm > cap_eq), "每币上限(S/267=26.2万U,公开口径)", "")
T.to_csv("/workspace/funding-strategy/v10/out/weekly_fix/coin_table_peak_bind_fixed.csv", index=False)
x = T[T.fut.isin(["LSKUSDT", "BLURUSDT", "ERAUSDT", "1000PEPEUSDT", "TRXUSDT"])]
print(x[["fut", "spot_real", "realistic_peak_cost_basis", "realistic_peak_mtm", "optimistic_peak_mtm", "realistic_peak_hour_utc", "realistic_oi_cap_at_peak", "realistic_peak_over_oicap", "realistic_peak_over_volcap", "bind_real", "bind_opt"]].round(2).to_string())
print(T.bind_real.value_counts().to_dict()); print(T.bind_opt.value_counts().to_dict())
