#!/usr/bin/env python3
"""Aggregate coverage / comparison of the hourly OI (no per-coin data): writes results/oi_hourly_coverage_summary.csv."""
import argparse
import numpy as np, pandas as pd
import config as C

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=None); a = ap.parse_args()
ddir = C.data_dir(a.data_dir)
cov = pd.read_csv(ddir / "oi_hourly_coverage.csv", parse_dates=["first", "last"])
z = np.load(ddir / "dataset.npz"); oih, oi = z["oih"], z["oi"]
from run_backtest import hidx
rows = []
def add(k, v): rows.append(dict(item=k, value=v))
add("symbols_in_pool", len(cov)); add("symbols_with_hourly_file", int((cov.hours > 0).sum()))
for w, (s, e) in C.WINDOWS.items():
    a0, a1 = hidx(s), hidx(e); x = oih[a0:a1]; d = oi[a0:a1]
    add(f"{w}_symbol_hours_total", int(x.size)); add(f"{w}_symbol_hours_with_hourly_oi_pct", round(float(np.isfinite(x).mean() * 100), 3))
    add(f"{w}_symbol_hours_with_daily_oi_pct", round(float(np.isfinite(d).mean() * 100), 3))
    have = np.isfinite(x).any(0); add(f"{w}_symbols_with_100pct_hours", int(np.isfinite(x).all(0).sum()))
    add(f"{w}_symbols_with_any_missing_hour", int((~np.isfinite(x).all(0)).sum()))
    # internal gaps: NaN after the first valid hour (cannot happen with ffill unless NaN persists) -> report hours NaN after first valid value
    first = np.argmax(np.isfinite(x), axis=0); internal = sum(int((~np.isfinite(x[first[i]:, i])).sum()) if have[i] else 0 for i in range(x.shape[1]))
    add(f"{w}_missing_hours_after_first_valid_value", internal)
    both = np.isfinite(x) & np.isfinite(d) & (d > 0)
    r = x[both] / d[both]
    add(f"{w}_hourly_over_prev_day_oi_ratio_p10", round(float(np.percentile(r, 10)), 3)); add(f"{w}_hourly_over_prev_day_oi_ratio_median", round(float(np.median(r)), 3))
    add(f"{w}_hourly_over_prev_day_oi_ratio_p90", round(float(np.percentile(r, 90)), 3))
    add(f"{w}_share_of_hours_hourly_limit_looser_than_daily_pct", round(float((r > 1).mean() * 100), 2))
pd.DataFrame(rows).to_csv(C.REPO / "results" / "oi_hourly_coverage_summary.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
