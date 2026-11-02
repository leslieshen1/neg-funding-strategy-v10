import itertools, json
from v10lib import *
D = R.Data(C.data_dir(None)); ok = R.load_ok(D); hc = os.environ["NFB_HOLDINGS_CAP"]
rows=[]
dvmed = np.nanmedian(np.where(D.dv>0, D.dv, np.nan), axis=0)
print("pool dv(24h min vol) quantiles M:", np.nanpercentile(dvmed/1e6,[10,25,50,75,90,95,99]).round(1))
S=7e7
for scen in ("A","B"):
  cap = R.make_cap(D, ok, S, "equal" if scen=="B" else "holdings", None if scen=="B" else hc)
  for case in ("realistic","optimistic"):
    c=CASES[case]
    for N,thr in [(1,0)]+[(n,t) for n in (2,3,4,6,9) for t in (0,2e7,5e7,1e8,2e8,5e8)]:
        p=R.base_params(c["imp"],k_imp=c["k"],n_streak=N,big_thr=thr)
        r=run_cfg(D,S,ok,cap,p,c["lag"])
        a=r["1y"]; rows.append(dict(scen=scen,case=case,N=N,big_thr_M=thr/1e6,tune=a["tune_ann_pct"],test=a["test_ann_pct"],y1=a["ann_pct"],m6=r["6m"]["ann_pct"],net_wan=a["net_wan"],ep=a["episodes"]))
df=pd.DataFrame(rows); df.to_csv("../out/a1_grid.csv",index=False)
print(df[(df.case=="realistic")].round(2).to_string(index=False))
