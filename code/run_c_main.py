from run_c import *
SG=[1e6,2e6,5e6,1e7,2e7,3e7,5e7,7e7]; THR,DMIN=0.04,30        # selected on the tune window only (best mean of realistic/optimistic tune return)
data=get_data(["BTC","ETH"]); rows=[]
for case,imp in CASES.items():
    for S in SG:
        cum,used,inf=simulate(S,["BTC","ETH"],THR,DMIN,imp,data=data); rows.append(dict(part="C",case=case,S=S/1e6,**slice_m(cum,used,S),**inf))
pd.DataFrame(rows).to_csv("../out/c_main.csv",index=False); pd.set_option("display.width",250); print(pd.DataFrame(rows).round(3).to_string(index=False))
sens=[]
for imp_bp in (30,20,10,5,2,0):
    cum,used,inf=simulate(1e7,["BTC","ETH"],THR,DMIN,imp_bp/1e4,data=data); sens.append(dict(impact_bp=imp_bp,**slice_m(cum,used,1e7)))
    cum,used,inf=simulate(1e7,["BTC","ETH"],0.0,7,imp_bp/1e4,data=data); sens[-1].update({"always_y1":slice_m(cum,used,1e7)["y1_ann"]})
pd.DataFrame(sens).to_csv("../out/c_sens.csv",index=False); print(pd.DataFrame(sens).round(3).to_string(index=False))
