"""Builds ../out/oos_monthly.csv from ../out/a_final.csv (written by run_final_a.py): test-period monthly net PnL (万U, incl. mark-to-market, 70M) per config,
and the difference vs v9.1. Months: 2026-07, 08, 09 and 10 (10 = 1-27 Oct incl. the forced end-of-window exit). oos_total = sum of the four months (= test_net_wan)."""
import pandas as pd
a = pd.read_csv("../out/a_final.csv")
a = a[(a.S_wan == 7000) & (a.tag.isna() | (a.tag == ""))]
MO = ["mo_2026-07", "mo_2026-08", "mo_2026-09", "mo_2026-10"]
rows = []
for scen in ("A", "B"):
    for case in ("realistic", "optimistic"):
        x = a[(a.scen == scen) & (a.case == case)]; base = x[x.cfg == "v9.1"].iloc[0]
        for _, r in x.iterrows():
            d = dict(scen=scen, case=case, cfg=r.cfg)
            for m in MO: d[m] = round(r[m], 2); d["diff_" + m] = round(r[m] - base[m], 2)
            d["oos_total"] = round(sum(r[m] for m in MO), 2); d["oos_total_diff"] = round(sum(r[m] - base[m] for m in MO), 2)
            d["test_net_wan_check"] = round(r.test_net_wan, 2)
            rows.append(d)
pd.DataFrame(rows).to_csv("../out/oos_monthly.csv", index=False); print(len(rows))
