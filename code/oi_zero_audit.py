"""Audit of the isolated OI=0 forward-fill (v10lib.fix_oi_zero): how many coin-hours are filled and WHEN (UTC), for the 267-coin pool and the big-coin set.
Writes ../out/oi_zero_fill_summary.csv (one row per pool x month, plus top single hours). No inventory data is read."""
import os, sys
import numpy as np, pandas as pd
import config as C
import run_backtest as R
from v10lib import fix_oi_zero
rows = []
for name, d in (("pool267", "/workspace/nfb-data"), ("big_set", "/workspace/nfb-data-big")):
    os.environ["NFB_DATA_DIR"] = d
    D = R.Data(C.data_dir(None)); before = D.oih.copy(); n = fix_oi_zero(D)
    filled = (before == 0) & (D.oih != 0)
    hrs = filled.sum(1); t = R.T0 + pd.to_timedelta(np.arange(len(hrs)), unit="h")
    s = pd.Series(hrs, index=t)
    print(name, "coin-hours filled:", n, "(check", int(filled.sum()), ") coins:", D.N)
    for mth, v in s.groupby(s.index.strftime("%Y-%m")).sum().items(): rows.append(dict(set=name, kind="month", key=mth, coin_hours=int(v)))
    for h, v in s[s > 0].sort_values(ascending=False).head(10).items(): rows.append(dict(set=name, kind="top_hour_utc", key=str(h), coin_hours=int(v)))
    print("distinct hours with a fill:", int((s > 0).sum()), "; distinct days:", int(s[s > 0].index.normalize().nunique()))
    print(s[s > 0].sort_values(ascending=False).head(8))
    print(s.groupby(s.index.strftime("%Y-%m")).sum().to_string())
pd.DataFrame(rows).to_csv("../out/oi_zero_fill_summary.csv", index=False)
