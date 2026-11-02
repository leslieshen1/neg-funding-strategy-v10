#!/usr/bin/env python3
"""v8 sensitivities that are NOT part of the headline stack (headline = run_backtest.FIXES), OI lag 0, 20bp unless stated, k=0 unless stated.
 add_once        one top-up per episode (initial + 1 add); headline keeps unlimited re-adds
 coin_ledger     inventory cap as a token ledger (A: qty at the window end; B: S/N converted with the t-1 close each hour)
 mtm_cap         the capital limit itself measured at market value (headline: only reported at MTM)
 reviewer_equiv  min(prior hour, current hour) volume/depth and SOFT volume/OI targets with OI on perp notional (= reviewer's "full stack", reconciliation only)
 no_hardcap      f_hardcap=0 (legacy soft limit on spot notional)
 sqrt_Y          square-root impact instead of the fixed bp: per leg  Y * sigma_day * sqrt(Q / ADV24h), Q = fill notional, ADV24h = trailing-24h min(spot, perp) quote volume (t-1),
                 sigma_day = std of 24 hourly spot log-returns (t-24..t-1) * sqrt(24); Y = 0.25 / 0.5 / 1 / 2 (Y=1 is the textbook order of magnitude; NOT calibrated to this book).
                 Fees and half-spread unchanged; the fixed 20bp is NOT added on top.
Also reports the combined realistic case (OI lag 1h + k=4) under sqrt impact.   Writes results/sens_v8_<cap>.csv.
"""
import argparse
import numpy as np, pandas as pd
import config as C
import run_backtest as R

CASES = [("headline", {}, {}), ("add_once", dict(f_addonce=1), {}), ("coin_ledger", dict(f_coin=1), {}), ("mtm_cap", dict(f_mtmcap=2), {}),
         ("reviewer_equiv", dict(f_prior=2, f_hardcap=2), {}), ("no_hardcap", dict(f_hardcap=0), {}),
         ("reviewer_equiv k4", dict(f_prior=2, f_hardcap=2, k_imp=4.0), {}), ("reviewer_equiv 30bp", dict(f_prior=2, f_hardcap=2, impact_bp=30.0), {}),
         ("sqrt_Y0.25", dict(sqrt_y=0.25), {}), ("sqrt_Y0.5", dict(sqrt_y=0.5), {}), ("sqrt_Y1", dict(sqrt_y=1.0), {}), ("sqrt_Y2", dict(sqrt_y=2.0), {}),
         ("sqrt_Y1 + OI lag1 + k4", dict(sqrt_y=1.0, k_imp=4.0), dict(oi_extra_lag_h=1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-mode", choices=["equal", "holdings"], default="equal"); ap.add_argument("--holdings-cap", default=None)
    ap.add_argument("--S", type=float, default=7e7); ap.add_argument("--out-dir", default=str(C.REPO / "results"))
    a = ap.parse_args()
    D = R.Data(C.data_dir(None)); ok = R.load_ok(D); cap = R.make_cap(D, ok, a.S, a.cap_mode, a.holdings_cap)
    rows = []
    for w in ("1y", "6m"):
        for name, pk, rk in CASES:
            r = D.run(w, a.S, ok, cap, R.base_params(**pk), **rk); m = R.metrics(r)
            rows.append(dict(window=w, case=name, ann_on_S_pct=m["ann_on_S_pct"], net_wan=m["net_wan"], avg_used_wan=m["avg_used_wan"], ann_on_used_pct=m["ann_on_used_pct"],
                             peak_used_mtm_wan=m["peak_used_mtm_wan"], episodes=m["episodes"], fills=m["fills"], spread_impact_wan=m["spread_impact_wan"],
                             hours_over_vol_cap=m["hours_over_vol_cap"], hours_over_oi_cap=m["hours_over_oi_cap"]))
    df = pd.DataFrame(rows).round(2); od = __import__("pathlib").Path(a.out_dir); od.mkdir(exist_ok=True)
    df.to_csv(od / f"sens_v8_{a.cap_mode}.csv", index=False); print(df.to_string(index=False))


if __name__ == "__main__":
    main()
