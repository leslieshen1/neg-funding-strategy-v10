"""Part C: long spot + short Binance USDT-M quarterly (delivery) contract, held to expiry. Hourly, causal.
Signal on the previous bar's close; fills at the previous bar's close +/- half-spread +/- fixed impact; fees as in v9.1.
Capital per $1 of hedged value = $1 spot + $1 margin on the 1x short (2x)."""
import json, itertools, numpy as np, pandas as pd
DD="/workspace/funding-strategy/v10/data/quarterly"; SPOT="/workspace/nfb-data-big/spot_klines"
T0=pd.Timestamp("2025-10-28"); T1=pd.Timestamp("2026-10-28"); SPLIT=pd.Timestamp("2026-07-01"); M6=pd.Timestamp("2026-05-01")
EXPIRY={"251226":"2025-12-26 08:00","260327":"2026-03-27 08:00","260626":"2026-06-26 08:00","260925":"2026-09-25 08:00","261225":"2026-12-25 08:00"}
HS_S, HS_F = 0.00064756279577795, 0.0002159691883340496      # pool-median half-spreads, same as every other coin in v9.1
FEE_S, FEE_F, FEE_SETTLE = 0.0010, 0.0005, 0.0005            # delivery settlement fee = taker fee (Binance FAQ)
CASES={"realistic":0.003,"optimistic":0.002}
IDX=pd.date_range(T0,T1,freq="h",inclusive="left"); H=len(IDX)
def arr(df,col): return df[col].reindex(pd.date_range(T0-pd.Timedelta(days=2),T1+pd.Timedelta(days=1),freq="h")).values
GI=pd.date_range(T0-pd.Timedelta(days=2),T1+pd.Timedelta(days=1),freq="h"); OFF=2*24
def load(coin):
    s=pd.read_csv(f"{SPOT}/{coin}USDT.csv"); s.index=pd.to_datetime(s.open_time,unit="ms")
    sp={c:s[c].astype(float).reindex(GI).values for c in ("open","high","low","close")}
    fut={}
    for e,ex in EXPIRY.items():
        try: r=json.load(open(f"{DD}/{coin}USDT_{e}.json"))
        except FileNotFoundError: continue
        d=pd.DataFrame(r,columns="ot o h l c v ct qv n tb tq i".split()); d.index=pd.to_datetime(d.ot,unit="ms")
        d=d[["c","qv"]].astype(float); d=d[(d.qv>0)&(d.index<pd.Timestamp(ex))]
        fut[e]=dict(c=d.c.reindex(GI).values,qv=d.qv.reindex(GI).values,first=d.index[0],ex=pd.Timestamp(ex),exh=int((pd.Timestamp(ex)-GI[0])/pd.Timedelta(hours=1)))
    return dict(sp=sp,fut=fut)
def get_data(coins): return {c:load(c) for c in coins}
def simulate(S, coins, thr, dmin, imp, vol_frac=0.10, data=None):
    n=len(coins); capcoin=S/n
    cum=np.zeros(H); used=np.zeros(H)
    # aggregate book per (coin, contract): q = coin units, cs = sum q*spot cost incl fee, fs = sum q*Fe (entry bid)
    book={}; realised=0.0; ep=0; nlots=0
    for h in range(H):
        g=h+OFF                                   # index into the global arrays for the hour starting at IDX[h]; previous bar = g-1
        for coin in coins:
            D=data[coin]; sp=D["sp"]
            # settle contracts expiring at this hour (08:00 bar boundary): exit hour = g (expiry 08:00 == start of bar g); use bar g-1 (07:00-08:00)
            for e,f in D["fut"].items():
                b=book.get((coin,e))
                if b and b["q"]>0 and f["exh"]==g:
                    o,hh,l,c=(sp[k][g-1] for k in ("open","high","low","close")); sett=(o+hh+l+c)/4.0
                    q=b["q"]; realised+=q*c*(1-HS_S-imp)*(1-FEE_S)-b["cs"] + b["fs"]-q*sett-q*sett*FEE_SETTLE
                    b["q"]=0; b["cs"]=0; b["fs"]=0; ep+=1
            # entry
            P=sp["close"][g-1]
            if not np.isfinite(P): continue
            best=None
            for e,f in D["fut"].items():
                if f["exh"]<=g: continue
                F=f["c"][g-1]; qv=f["qv"][g-1]
                if not (np.isfinite(F) and np.isfinite(qv)) or qv<=0 or GI[g-1]<f["first"]+pd.Timedelta(hours=1): continue
                days=(f["exh"]-g)/24.0
                if days<dmin: continue
                a=(F/P-1)*365/days
                if best is None or a>best[0]: best=(a,e,F,qv)
            if best is None or best[0]<thr: continue
            a,e,F,qv=best
            curc=sum(b["q"]*(b["P"]+b["F"]) for (c_,e_),b in book.items() if c_==coin)
            tot=sum(b["q"]*(b["P"]+b["F"]) for b in book.values())
            room=min(capcoin-curc,S-tot)
            notional=min(vol_frac*qv, room/2.0)
            if notional<=1000: continue
            q=notional/P; Pa=P*(1+HS_S+imp); Fb=F*(1-HS_F-imp)
            b=book.setdefault((coin,e),dict(q=0.0,cs=0.0,fs=0.0,P=P,F=F))
            # keep a quantity-weighted mid for the capital measure
            w=b["q"]+q; b["P"]=(b["P"]*b["q"]+P*q)/w; b["F"]=(b["F"]*b["q"]+F*q)/w
            b["q"]=w; b["cs"]+=q*Pa*(1+FEE_S); b["fs"]+=q*Fb; realised-=q*Fb*FEE_F; nlots+=1
        # mark to market at the end of this hour using bar g-1 close (same information as the signal)
        m=0.0; u=0.0
        for (coin,e),b in book.items():
            if b["q"]<=0: continue
            D=data[coin]; Pt=D["sp"]["close"][g-1]; f=D["fut"][e]; Ft=f["c"][g-1]
            if not np.isfinite(Ft):
                k=g-1
                while k>0 and not np.isfinite(f["c"][k]): k-=1
                Ft=f["c"][k]
            m+=b["q"]*Pt-b["cs"]+b["fs"]-b["q"]*Ft; u+=b["q"]*(Pt+Ft)
        cum[h]=realised+m; used[h]=u
    liq=0.0; openn=0; lock=0.0
    for (coin,e),b in book.items():
        if b["q"]>0:
            D=data[coin]; g=H-1+OFF; Pt=D["sp"]["close"][g]; f=D["fut"][e]; Ft=f["c"][g]
            liq+=b["q"]*(Pt*(HS_S+imp+FEE_S)+Ft*(HS_F+imp+FEE_F)); openn+=1
            lock+=b["q"]*Pt*((1-HS_S-imp)*(1-FEE_S)-1-FEE_SETTLE)-b["cs"]+b["fs"]   # value if the open contract is held to its expiry (price-neutral)
    cum[-1]-=liq
    return cum,used,dict(settled=ep,lots=nlots,open_contracts=openn,lock_if_held_wan=lock/1e4,liq_cost_wan=liq/1e4)
def slice_m(cum,used,S):
    hs=int((SPLIT-T0)/pd.Timedelta(hours=1)); h6=int((M6-T0)/pd.Timedelta(hours=1)); y=lambda k:k/24/365
    return dict(net_wan=cum[-1]/1e4, y1_ann=cum[-1]/S/y(H)*100, tune_ann=cum[hs-1]/S/y(hs)*100, test_ann=(cum[-1]-cum[hs-1])/S/y(H-hs)*100,
                m6_ann=(cum[-1]-cum[h6-1])/S/y(H-h6)*100, peak_pct_of_S=used.max()/S*100, avg_used_wan=used.mean()/1e4,
                tune_net_wan=cum[hs-1]/1e4, test_net_wan=(cum[-1]-cum[hs-1])/1e4)
if __name__=="__main__":
    data=get_data(["BTC","ETH"]); rows=[]; S=1e7
    for coins in (["BTC"],["ETH"],["BTC","ETH"]):
      for case,imp in CASES.items():
        for thr,dmin in itertools.product((0.0,0.02,0.04,0.06,0.08,0.10),(7,30)):
            cum,used,inf=simulate(S,coins,thr,dmin,imp,data=data)
            rows.append(dict(coins="+".join(coins),case=case,thr=thr,dmin=dmin,**slice_m(cum,used,S),**inf))
    df=pd.DataFrame(rows); df.to_csv("../out/c_grid.csv",index=False); pd.set_option("display.width",250)
    print(df[df.coins=="BTC+ETH"].round(3).to_string(index=False))
