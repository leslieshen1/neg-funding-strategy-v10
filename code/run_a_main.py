"""A: negative-funding v10 - each item alone and combined vs v9.1, over the capital grid. Parameters were chosen on the tune period only (see exp_*.py)."""
import multiprocessing as mp
from v10lib import *
SEL = dict(n_streak=3, big_thr=2e7, tier_r=2.0, pov=0.03)          # chosen on 2025-10-28..2026-06-30 only
CFG = {"v9.1 基线": {}, "A1 严入场(N=3,≥2000万)": dict(n_streak=3, big_thr=2e7), "A2 分层上限(r=2)": dict(tier=1), "A3 同小时砍回": dict(derisk_force=1),
       "A5 拆单(pov=0.03)": dict(pov=0.03), "A1+A3": dict(n_streak=3, big_thr=2e7, derisk_force=1), "A1+A3+A5": dict(n_streak=3, big_thr=2e7, derisk_force=1, pov=0.03),
       "全部 A1+A2+A3+A5": dict(n_streak=3, big_thr=2e7, tier=1, derisk_force=1, pov=0.03)}
G = {}
def tier_cap(D, ok, S, r):
    a,b = C.WINDOWS["1y"]; h0=R.hidx(a); hs=R.hidx(SPLIT)
    v = np.nanmedian(np.where(D.dv[h0:hs]>0, D.dv[h0:hs], np.nan), axis=0); v=np.where(np.isfinite(v), v, np.nanmin(v))
    q = np.digitize(v, np.nanpercentile(v,[25,50,75])); w = np.where(ok>0, r**q, 0.0); return R.CapSpec(S*w/w.sum())
def one(t):
    scen,case,S,name = t; D,ok=G["D"],G["ok"]; c=CASES[case]; kw=dict(CFG[name]); tier=kw.pop("tier",0)
    if scen=="B": cap = tier_cap(D,ok,S,SEL["tier_r"]) if tier else R.make_cap(D,ok,S,"equal")
    else:
        if tier: return None
        cap = R.make_cap(D,ok,S,"holdings",G["hc"])
    r = run_cfg(D,S,ok,cap,R.base_params(c["imp"],k_imp=c["k"],**kw),c["lag"])
    a=r["1y"]; row=dict(scen=scen,case=case,S_wan=S/1e4,cfg=name,**{k:a[k] for k in a if k!="window"})
    row.update(m6_ann_pct=r["6m"]["ann_pct"], m6_net_wan=r["6m"]["net_wan"]); return row
if __name__=="__main__":
    G["D"]=R.Data(C.data_dir(None)); G["ok"]=R.load_ok(G["D"]); G["hc"]=os.environ["NFB_HOLDINGS_CAP"]
    one(("B","realistic",1e6,"v9.1 基线"))
    tasks=[(s,c,S,n) for s in ("A","B") for c in CASES for S in SGRID for n in CFG]
    with mp.get_context("fork").Pool(8) as p: rows=p.map(one,tasks,chunksize=4)
    df=pd.DataFrame([r for r in rows if r]); df.to_csv("../out/a_main.csv",index=False)
    x=df[(df.S_wan==7000)&(df.case=="realistic")]; print(x[["scen","cfg","tune_ann_pct","test_ann_pct","ann_pct","m6_ann_pct","net_wan","peak_pct_of_S"]].round(2).to_string(index=False))
