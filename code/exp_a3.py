from v10lib import *
D = R.Data(C.data_dir(None)); ok = R.load_ok(D); hc=os.environ["NFB_HOLDINGS_CAP"]
rows=[]; S=7e7
for scen in ("A","B"):
  cap = R.make_cap(D, ok, S, "equal" if scen=="B" else "holdings", None if scen=="B" else hc)
  for case in ("realistic","optimistic"):
    c=CASES[case]
    for nm,kw in [("base",{}),("force",dict(derisk_force=1)),("pov0.10",dict(pov=0.10)),("pov0.05",dict(pov=0.05)),("pov0.10+force",dict(pov=0.10,derisk_force=1)),("pov0.40",dict(pov=0.40))]:
        r=run_cfg(D,S,ok,cap,R.base_params(c["imp"],k_imp=c["k"],**kw),c["lag"]); a=r["1y"]
        rows.append(dict(scen=scen,case=case,cfg=nm,tune=a["tune_ann_pct"],test=a["test_ann_pct"],y1=a["ann_pct"],m6=r["6m"]["ann_pct"],net_wan=a["net_wan"],peak_pct=a["peak_pct_of_S"],avg_wan=a["avg_used_mtm_wan"],cost=a["cost_wan"],fees=a["fees_wan"]))
df=pd.DataFrame(rows); df.to_csv("../out/a3_pov_grid.csv",index=False); print(df.round(2).to_string(index=False))
