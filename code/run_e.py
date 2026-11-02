"""Part E: combined portfolio. neg v10 r2 (A5+A3) + P (large-coin positive funding, Set4 hold, OI-glitch fixed) + C (BTC/ETH quarterly, thr 4%/min 30d) + D (idle-U, reference).
Allocation rule (fixed before looking at the test window): neg gets the capital level that maximises its TUNE-window profit among the grid points <= S; the remainder is split equally
between P and C if each has positive tune-window return in the realistic case (both do); everything not in use is idle (D)."""
import os; os.environ["NFB_DATA_DIR"]="/workspace/nfb-data-big"
import numpy as np, pandas as pd
import run_c as RC
from run_p import runP, SET4
D_=pd.read_csv("../out/a_final.csv"); AAVE=pd.read_csv("../out/idle_yield_aave_v3_usdt_daily.csv",parse_dates=["t"])
def rate(a,b): x=AAVE[(AAVE.t>=a)&(AAVE.t<b)]; return x.apyBase.mean()/100
YR=dict(tune=246/365, test=119/365, m6=180/365, y1=365/365)
RATE=dict(tune=rate("2025-10-28","2026-07-01"), test=rate("2026-07-01","2026-10-28"), m6=rate("2026-05-01","2026-10-28"), y1=rate("2025-10-28","2026-10-28"))
cdata=RC.get_data(["BTC","ETH"])
TIERS=[1e6,2e6,5e6,1e7,2e7,3e7,5e7,7e7]
def neg_row(scen,case,S_wan,cfg): 
    x=D_[(D_.scen==scen)&(D_.case==case)&(D_.S_wan==S_wan)&(D_.cfg==cfg)]; return x.iloc[0]
def neg_part(scen,case,S,cfg):
    grid=sorted(set(D_.S_wan)); cand=[g for g in grid if g*1e4<=S+1]
    best=max(cand,key=lambda g: neg_row(scen,"realistic",g,cfg).tune_net_wan)       # decided on realistic tune profit only
    r=neg_row(scen,case,best,cfg)
    return best*1e4, dict(tune=r.tune_net_wan,test=r.test_net_wan,m6=r.m6_net_wan,y1=r["1y_net_wan"],used=r["1y_avg_used_wan"])
def b_part(S,case):
    r=runP(S,SET4,case); a=r["1y"]; m6=r["6m"]["net_wan"]
    return dict(tune=a["tune_net_wan"],test=a["test_net_wan"],m6=m6,y1=a["net_wan"],used=a["avg_used_mtm_wan"])
def c_part(S,case):
    imp=RC.CASES[case]; cum,used,inf=RC.simulate(S,["BTC","ETH"],0.04,30,imp,data=cdata); m=RC.slice_m(cum,used,S)
    return dict(tune=m["tune_net_wan"],test=m["test_net_wan"],m6=m["m6_ann"]/100*S*YR["m6"]/1e4,y1=m["net_wan"],used=m["avg_used_wan"])
rows=[]
for scen in ("A","B"):
  for case in ("realistic","optimistic"):
    for S in TIERS:
        base=neg_row(scen,case,S/1e4,"v9.1"); Sn,neg=neg_part(scen,case,S,"A5+A3"); rem=S-Sn
        z=dict(tune=0,test=0,m6=0,y1=0,used=0)
        bb=b_part(rem/2,case) if rem>1e5 else z; cc=c_part(rem/2,case) if rem>1e5 else z
        tot_used=neg["used"]+bb["used"]+cc["used"]; idle=max(S/1e4-tot_used,0)
        row=dict(scen=scen,case=case,S_wan=S/1e4,S_neg_wan=Sn/1e4,v91_y1_ann=base["1y_ann_pct"],v91_net=base["1y_net_wan"],idle_wan=idle)
        for k in ("tune","test","m6","y1"):
            n=neg[k]; b=bb[k]; c=cc[k]; d=idle*RATE[k]*YR[k]
            row.update({f"neg_{k}_net":n,f"P_{k}_net":b,f"C_{k}_net":c,f"D_{k}_net":d,
                        f"E_noD_{k}_ann":(n+b+c)/(S/1e4)/YR[k]*100, f"E_D_{k}_ann":(n+b+c+d)/(S/1e4)/YR[k]*100, f"neg_{k}_ann":n/(S/1e4)/YR[k]*100, f"E_negD_{k}_ann":(n+d)/(S/1e4)/YR[k]*100})
        rows.append(row)
df=pd.DataFrame(rows); df.to_csv("../out/e_main.csv",index=False); pd.set_option("display.width",300)
print(RATE)
print(df[["scen","case","S_wan","S_neg_wan","v91_y1_ann","neg_y1_ann","E_noD_y1_ann","E_D_y1_ann","E_noD_tune_ann","E_noD_test_ann","E_noD_m6_ann","neg_y1_net","P_y1_net","C_y1_net","D_y1_net"]].round(2).to_string(index=False))
