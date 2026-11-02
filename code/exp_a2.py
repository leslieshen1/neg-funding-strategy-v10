from v10lib import *
D = R.Data(C.data_dir(None)); ok = R.load_ok(D)
a,b = C.WINDOWS["1y"]; h0=R.hidx(a); hs=R.hidx(SPLIT)
# tier input: median over the TUNE period of trailing-24h min(spot,perp) quote volume (information up to 2026-06-30 only)
import sys; PRE = len(sys.argv)>1 and sys.argv[1]=="pre"
lo,hi = (24,h0) if PRE else (h0,hs)
v = np.nanmedian(np.where(D.dv[lo:hi]>0, D.dv[lo:hi], np.nan), axis=0); v=np.where(np.isfinite(v), v, np.nanmin(v))
q = np.digitize(v, np.nanpercentile(v,[25,50,75]))   # 0..3
def tier_cap(S, r):
    w = np.where(ok>0, r**q, 0.0); return R.CapSpec(S*w/w.sum())
def cont_cap(S, al):
    w = np.where(ok>0, v**al, 0.0); return R.CapSpec(S*w/w.sum())
rows=[]; S=7e7; c=CASES["realistic"]
for kind,vals in (("tier4",[0.5,0.67,1,1.5,2,3,4,6,10]),("cont",[0,0.25,0.5,0.75,1.0])):
  for x in vals:
    for case in ("realistic","optimistic"):
        c=CASES[case]; cap = tier_cap(S,x) if kind=="tier4" else cont_cap(S,x)
        r=run_cfg(D,S,ok,cap,R.base_params(c["imp"],k_imp=c["k"]),c["lag"]); a_=r["1y"]
        rows.append(dict(kind=kind,x=x,case=case,tune=a_["tune_ann_pct"],test=a_["test_ann_pct"],y1=a_["ann_pct"],m6=r["6m"]["ann_pct"],net_wan=a_["net_wan"],avg_used_wan=a_["avg_used_mtm_wan"],peak_wan=a_["peak_used_mtm_wan"]))
df=pd.DataFrame(rows); df.to_csv("../out/a2_grid"+("_pre" if PRE else "")+".csv",index=False); print(df.round(2).to_string(index=False))
print("tier edges (M USDT median 24h vol):", np.nanpercentile(v,[25,50,75]).round(2)/1e6)
