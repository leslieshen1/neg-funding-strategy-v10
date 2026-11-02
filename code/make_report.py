import pandas as pd, numpy as np
O="../out/"; a=pd.read_csv(O+"a_final.csv"); p=pd.read_csv(O+"p_main.csv"); c=pd.read_csv(O+"c_main.csv"); e=pd.read_csv(O+"e_main.csv")
main=[]
def addrow(**k): main.append(k)
SN={"A":"A 私有库存口径(只报汇总)","B":"B 公开口径 每币S/267"}
NM={"v9.1":"v9.1","A3 同小时砍回":"v9.1+A3","A5 拆单(pov=0.03)":"v9.1+A5","A5+A3":"v9.1+A5+A3(主结论)","旧捆绑 A1+A3+A5":"旧捆绑A1+A3+A5(A1未验证)"}
for scen in ("A","B"):
  for case in ("realistic","optimistic"):
    for S in (100,500,1000,2000,3000,5000,7000):
      for cfg,tag in NM.items():
        r=a[(a.scen==scen)&(a.case==case)&(a.S_wan==S)&(a.cfg==cfg)].iloc[0]
        addrow(部分="A 负费率",版本=tag,口径=SN[scen],情景=case,本金_万U=S,一年净利_万U=r["1y_net_wan"],一年年化_pct=r["1y_ann_pct"],调参期年化_pct=r.tune_ann_pct,检验期年化_pct=r.test_ann_pct,近六月年化_pct_仅参考=r.m6_ann_pct,峰值占本金_pct=r.peak_pct_of_S,平均占用_万U=r["1y_avg_used_wan"])
for _,r in p.iterrows(): addrow(部分="P 大币正费率(Set4,全年持有)" if r.set=="Set4" else "P 大币正费率(Set10 全持有,仅参考)",版本="v10 r2(OI=0已按缺失处理)",口径="自有U",情景=r.case,本金_万U=r.S*100,一年净利_万U=r.net_wan,一年年化_pct=r.y1_ann,调参期年化_pct=r.tune_ann,检验期年化_pct=r.test_ann,近六月年化_pct_仅参考=r.m6_ann,峰值占本金_pct=r.peak_pct_of_S,平均占用_万U=r.avg_used_wan)
for _,r in c.iterrows(): addrow(部分="C 季度交割基差(BTC+ETH)",版本="v10 r2(C未受OI=0影响,未重跑)",口径="自有U",情景=r.case,本金_万U=r.S*100,一年净利_万U=r.net_wan,一年年化_pct=r.y1_ann,调参期年化_pct=r.tune_ann,检验期年化_pct=r.test_ann,近六月年化_pct_仅参考=r.m6_ann,峰值占本金_pct=r.peak_pct_of_S,平均占用_万U=r.avg_used_wan)
for _,r in e.iterrows():
    for tag,col in (("E 组合(v9.1+A5+A3 + P + C;不含理财)","E_noD"),("E 组合 + Aave参考理财行(单列,不并入主结论)","E_D")):
        addrow(部分=tag,版本="v10 r2(A沿用原数据,P已按OI=0修正,C未重跑)",口径=SN[r.scen],情景=r.case,本金_万U=r.S_wan,一年净利_万U=r.neg_y1_net+r.P_y1_net+r.C_y1_net+(r.D_y1_net if col=="E_D" else 0),
               一年年化_pct=r[f"{col}_y1_ann"],调参期年化_pct=r[f"{col}_tune_ann"],检验期年化_pct=r[f"{col}_test_ann"],近六月年化_pct_仅参考=r[f"{col}_m6_ann"],峰值占本金_pct=np.nan,平均占用_万U=np.nan)
pd.DataFrame(main).round(3).to_csv("../out/main_table.csv",index=False,encoding="utf-8-sig"); print(len(main))
