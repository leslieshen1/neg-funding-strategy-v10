from v10lib import *
D = R.Data(C.data_dir(None)); ok = R.load_ok(D); hc=os.environ["NFB_HOLDINGS_CAP"]
rows=[]; S=7e7
for scen in ("A","B"):
  cap = R.make_cap(D, ok, S, "equal" if scen=="B" else "holdings", None if scen=="B" else hc)
  for case in ("realistic","optimistic"):
    c=CASES[case]
    for pv in (0.40,0.20,0.15,0.10,0.07,0.05,0.04,0.03,0.02,0.01):
        r=run_cfg(D,S,ok,cap,R.base_params(c["imp"],k_imp=c["k"],pov=pv),c["lag"]); a=r["1y"]
        rows.append(dict(scen=scen,case=case,pov=pv,tune=a["tune_ann_pct"],test=a["test_ann_pct"],y1=a["ann_pct"],m6=r["6m"]["ann_pct"],net_wan=a["net_wan"],peak_pct=a["peak_pct_of_S"],avg_wan=a["avg_used_mtm_wan"],ep=a["episodes"],cost=a["cost_wan"],fees=a["fees_wan"],fund=a["funding_wan"],basis=a["basis_wan"]))
df=pd.DataFrame(rows); df.to_csv("../out/a5_pov_grid.csv",index=False); print(df[df.case=="realistic"].round(2).to_string(index=False))
