import pandas as pd, numpy as np
O="../out/"
a=pd.read_csv(O+"a_final.csv"); a0=pd.read_csv(O+"a_main.csv"); p=pd.read_csv(O+"p_main.csv"); c=pd.read_csv(O+"c_main.csv"); e=pd.read_csv(O+"e_main.csv")
ps=pd.read_csv(O+"p_sens.csv"); cs=pd.read_csv(O+"c_cost_sens.csv"); mech=pd.read_csv(O+"a_mechanism.csv"); sq=pd.read_csv(O+"a_sqrt_sens.csv"); oif=pd.read_csv(O+"a_oifix_sens.csv")
wk=pd.read_csv(O+"weekly_fix/coin_table_peak_bind_fixed.csv"); rate=pd.read_csv(O+"idle_yield_aave_v3_usdt_daily.csv",parse_dates=["t"])
R=lambda a_,b_: rate[(rate.t>=a_)&(rate.t<b_)].apyBase.mean()
def md(df):
    h="| "+" | ".join(df.columns)+" |\n|"+"|".join(["---"]*len(df.columns))+"|\n"
    return h+"\n".join("| "+" | ".join(str(x) for x in r)+" |" for r in df.values)
g=lambda x,n=2: "-" if pd.isna(x) else f"{x:.{n}f}"
sg=lambda x,n=2: "-" if pd.isna(x) else f"{x:+.{n}f}"
H0,H1,H3,H5,H53,HB="v9.1","A3 同小时砍回","A5 拆单(pov=0.03)","A5+A3","旧捆绑 A1+A3+A5","v9.1"
def A(scen,case,S,cfg): return a[(a.scen==scen)&(a.case==case)&(a.S_wan==S)&(a.cfg==cfg)].iloc[0]
def A0(scen,case,S,cfg): return a0[(a0.scen==scen)&(a0.case==case)&(a0.S_wan==S)&(a0.cfg==cfg)].iloc[0]
def d(scen,case,cfg,col="1y_ann_pct",base=H0,S=7000): return A(scen,case,S,cfg)[col]-A(scen,case,S,base)[col]
AAVE=dict(y1=R('2025-10-28','2026-10-28'),tune=R('2025-10-28','2026-07-01'),test=R('2026-07-01','2026-10-28'),m6=R('2026-05-01','2026-10-28'))
# ---------- main table
def mt(case):
    rows=[]
    def row(nm,net,y1,tu,te,m6,pk): rows.append([nm,g(net,0),g(y1),g(tu),g(te),g(m6),g(pk,1) if pk==pk else "-"])
    for scen,sn in (("A","A口径(私有库存,仅汇总)"),("B","B口径(公开,每币S/267)")):
        r=A(scen,case,7000,H0); row(f"v9.1 · {sn}",r["1y_net_wan"],r["1y_ann_pct"],r.tune_ann_pct,r.test_ann_pct,r.m6_ann_pct,r.peak_pct_of_S)
    for scen,sn in (("A","A口径(私有库存,仅汇总)"),("B","B口径(公开,每币S/267)")):
        r=A(scen,case,7000,H5); row(f"**v9.1+拆单(A5)+A3(主结论)** · {sn}",r["1y_net_wan"],r["1y_ann_pct"],r.tune_ann_pct,r.test_ann_pct,r.m6_ann_pct,r.peak_pct_of_S)
    pp=p[(p.set=="Set4")&(p.case==case)&(p.S==70)].iloc[0]; cc=c[(c.case==case)&(c.S==70)].iloc[0]
    row("P 大币正费率(BTC/ETH/SOL/BNB 各1/4)·约不赚不亏,不值得做",pp.net_wan,pp.y1_ann,pp.tune_ann,pp.test_ann,pp.m6_ann,pp.peak_pct_of_S)
    row("C 季度交割基差(BTC+ETH)·约不赚不亏,不值得做",cc.net_wan,cc.y1_ann,cc.tune_ann,cc.test_ann,cc.m6_ann,cc.peak_pct_of_S)
    for scen,sn in (("A","A口径"),("B","B口径")):
        r=e[(e.scen==scen)&(e.case==case)&(e.S_wan==7000)].iloc[0]
        row(f"E 组合=主结论+P+C(不含理财)·{sn}(负费率占{g(r.S_neg_wan/100,0)}M)",r.neg_y1_net+r.P_y1_net+r.C_y1_net,r.E_noD_y1_ann,r.E_noD_tune_ann,r.E_noD_test_ann,r.E_noD_m6_ann,np.nan)
    for scen,sn in (("A","A口径"),("B","B口径")):
        r=e[(e.scen==scen)&(e.case==case)&(e.S_wan==7000)].iloc[0]
        row(f"E 组合 + Aave 参考理财行(单列,不并入主结论)·{sn}",r.neg_y1_net+r.P_y1_net+r.C_y1_net+r.D_y1_net,r.E_D_y1_ann,r.E_D_tune_ann,r.E_D_test_ann,r.E_D_m6_ann,np.nan)
    rows.append([f"闲置理财参考利率(Aave v3 USDT,单列,不是币安内收益)","-",g(AAVE['y1']),g(AAVE['tune']),g(AAVE['test']),g(AAVE['m6'])+"(仅参考)","-"])
    return md(pd.DataFrame(rows,columns=["策略(本金70M=7000万U)","一年净利(万U)","一年年化%","调参期年化%(前8月)","检验期年化%(后约4月)","近六月年化%(与调参期重叠,仅参考)","峰值占本金%"]))
# ---------- item deltas
def deltas(case):
    rows=[]
    for scen in ("A","B"):
        for S,nm in ((1000,"10M"),(7000,"70M")):
            base=A(scen,case,S,H0)
            for cfg in ["A1 严入场(N=3,≥2000万)","A2 分层上限(r=2)",H1,H3,H5,H53]:
                if cfg.startswith("A1 ") or cfg.startswith("A2 "):
                    if scen=="A" and cfg.startswith("A2"): continue
                    x=a0[(a0.scen==scen)&(a0.case==case)&(a0.S_wan==S)&(a0.cfg==cfg)]
                    if x.empty: continue
                    r=x.iloc[0]; lab={"A1":"A1 严入场(未验证,不采用)","A2":"A2 分层上限(过拟合,不采用)"}[cfg[:2]]
                else:
                    r=A(scen,case,S,cfg); lab={H1:"A3 同小时砍回(风控必须项)",H3:"A5 拆单",H5:"**A5+A3(主结论)**",H53:"旧捆绑 A1+A3+A5"}[cfg]
                rows.append([scen,nm,lab,sg(r["1y_ann_pct"]-base["1y_ann_pct"]),sg(r.tune_ann_pct-base.tune_ann_pct),sg(r.test_ann_pct-base.test_ann_pct),sg(r.m6_ann_pct-base.m6_ann_pct),sg(r["1y_net_wan"]-base["1y_net_wan"],0)])
    return md(pd.DataFrame(rows,columns=["口径","本金","项目","一年年化变化(百分点)","调参期变化","检验期变化","近六月变化(仅参考)","一年净利变化(万U)"]))
# ---------- headline numbers for A5+A3
def hl(case):
    out=[]
    for scen,sn in (("A","A口径"),("B","B口径")):
        r=A(scen,case,7000,H5); b=A(scen,case,7000,H0)
        out.append(f"{sn}：一年 {g(b['1y_ann_pct'])}% → {g(r['1y_ann_pct'])}%；调参期 {g(b.tune_ann_pct)}% → {g(r.tune_ann_pct)}%；**检验期 {g(b.test_ann_pct)}% → {g(r.test_ann_pct)}%（{sg(r.test_ann_pct-b.test_ann_pct)}pp）**；近六月(仅参考) {g(b.m6_ann_pct)}% → {g(r.m6_ann_pct)}%")
    return out
# ---------- mechanism table
def mechtab(case="realistic"):
    rows=[]
    for scen,sn in (("A","A"),("B","B")):
        for cfg,lab in ((H0,"v9.1"),(H3,"v9.1+A5(仅拆单,不合规)"),(H5,"v9.1+A5+A3(主结论)")):
            r=A(scen,case,7000,cfg); m=mech[(mech.scen==scen)&(mech.cfg==cfg)].iloc[0]
            resid=r.net_wan-r.funding_wan-r.cost_wan-r.fees_wan
            rows.append([sn,lab,f"{int(r.episodes)}",f"{int(r.fills):,}",g(m.avg_notional_per_fill_wan,2),g(r.traded_x_S,2),g(r.funding_wan,0),g(r.cost_wan,0),g(r.fees_wan,0),g(resid,0),g(r.net_wan,0)])
    return md(pd.DataFrame(rows,columns=["口径","配置","开仓次数","成交笔数","平均每笔成交额(万U)","全年总成交额÷本金(倍)","资金费收入(万U)","点差+冲击成本(万U)","手续费(万U)","价差毛收益(万U,倒推)","一年净利(万U)"]))
def captab():
    rows=[]
    for scen in ("A","B"):
        for cfg,lab in ((H0,"v9.1"),(H3,"v9.1+A5(仅拆单)"),(H5,"v9.1+A5+A3")):
            m=mech[(mech.scen==scen)&(mech.cfg==cfg)].iloc[0]; rows.append([scen,lab,f"{int(m.hours_over_vol_cap):,}",f"{int(m.hours_over_oi_cap):,}"])
    return md(pd.DataFrame(rows,columns=["口径","配置","超过成交量上限的“币·小时”数","超过OI上限的“币·小时”数"]))
# ---------- monthly OOS table
MON=[("mo_2026-07","7月"),("mo_2026-08","8月"),("mo_2026-09","9月"),("mo_2026-10","10月(1–27日,含窗口末平仓)")]
def montab(case):
    rows=[]
    for scen in ("A","B"):
        b=A(scen,case,7000,H0); r5=A(scen,case,7000,H5); r3=A(scen,case,7000,H3); ro=A(scen,case,7000,H53)
        rows.append([scen,"v9.1 当月净利(万U)"]+[g(b[m],1) for m,_ in MON]+[g(sum(b[m] for m,_ in MON),1)])
        rows.append([scen,"v9.1+A5+A3 当月净利"]+[g(r5[m],1) for m,_ in MON]+[g(sum(r5[m] for m,_ in MON),1)])
        for lab,r in (("**A5+A3 − v9.1**",r5),("A5(仅拆单) − v9.1",r3),("旧捆绑A1+A3+A5 − v9.1",ro)):
            rows.append([scen,lab]+[sg(r[m]-b[m],1) for m,_ in MON]+[sg(sum(r[m]-b[m] for m,_ in MON),1)])
    return md(pd.DataFrame(rows,columns=["口径","项目"]+[n for _,n in MON]+["检验期合计"]))
def ladder():
    rows=[]
    for scen in ("A","B"):
        for S in (100,300,500,1000,2000,3000,5000,7000):
            b=A(scen,"realistic",S,H0); r=A(scen,"realistic",S,H5)
            rows.append([scen,f"{S//100}M",g(b["1y_ann_pct"]),g(r["1y_ann_pct"]),sg(r["1y_ann_pct"]-b["1y_ann_pct"]),sg(r.tune_ann_pct-b.tune_ann_pct),sg(r.test_ann_pct-b.test_ann_pct)])
    return md(pd.DataFrame(rows,columns=["口径","本金","v9.1 一年年化%","v9.1+A5+A3 一年年化%","一年差(pp)","调参期差(pp)","检验期差(pp)"]))
# ---------- P, C tables
pcols=["case","S","net_wan","y1_ann","tune_ann","test_ann","m6_ann","avg_used_wan","peak_pct_of_S","fund_wan","cost_wan","fees_wan"]
pn=["情景","本金(百万U)","一年净利(万U)","一年年化%","调参期%","检验期%","近六月%(仅参考)","平均占用(万U)","峰值占本金%","资金费(万U)","点差+冲击(万U)","手续费(万U)"]
def tier(df,cols,names): return md(df[cols].round(2).rename(columns=dict(zip(cols,names))))
P4=p[p.set=="Set4"]; P10=p[p.set=="Set10"]
ccols=["case","S","net_wan","y1_ann","tune_ann","test_ann","m6_ann","avg_used_wan","peak_pct_of_S","liq_cost_wan"]
cn=["情景","本金(百万U)","一年净利(万U)","一年年化%","调参期%","检验期%","近六月%(仅参考)","平均占用(万U)","峰值占本金%","窗口末强平成本(万U)"]
def etab(scen,case):
    x=e[(e.scen==scen)&(e.case==case)].copy()
    out=pd.DataFrame({"本金(万U)":x.S_wan.astype(int),"v9.1 一年年化%":x.v91_y1_ann.round(2),"负费率占用本金(万U)":x.S_neg_wan.astype(int),"主结论负费率 年化%":x.neg_y1_ann.round(2),
      "+P+C 年化%(不含理财)":x.E_noD_y1_ann.round(2),"P净利(万U)":x.P_y1_net.round(1),"C净利(万U)":x.C_y1_net.round(1),"闲置U平均(万U)":x.idle_wan.round(0).astype(int),"Aave参考行净利(万U)":x.D_y1_net.round(0).astype(int),
      "含Aave参考 年化%":x.E_D_y1_ann.round(2),"调参期%(不含理财)":x.E_noD_tune_ann.round(2),"检验期%(不含理财)":x.E_noD_test_ann.round(2),"近六月%(仅参考)":x.E_noD_m6_ann.round(2)})
    return md(out)
def E_(scen,case,col): return e[(e.scen==scen)&(e.case==case)&(e.S_wan==7000)][col].iloc[0]
lsk=wk[wk.fut=="LSKUSDT"].iloc[0]; nb=wk.bind_real.value_counts().to_dict()
cb=cs.iloc[0]; c10=c[(c.case=="realistic")&(c.S==10)].iloc[0]
sqt=sq.copy(); sqt["系数"]=sqt.tag.str.replace("sqrt_y=","")
sqrows=[]
for cf,lab in ((H0,"v9.1"),(H5,"v9.1+A5+A3")):
    for y in ("0.5","1.0"):
        r=sqt[(sqt.cfg==cf)&(sqt.系数==y)].iloc[0]; sqrows.append([lab,y,g(r.ann_pct),g(r.tune_ann_pct),g(r.test_ann_pct)])
    r=A("A","realistic",7000,cf); sqrows.insert(len(sqrows)-2,[lab,"本报告固定冲击(30bp,基准)",g(r["1y_ann_pct"]),g(r.tune_ann_pct),g(r.test_ann_pct)])
sqtab=md(pd.DataFrame(sqrows,columns=["配置(A口径,70M,现实)","平方根冲击系数","一年年化%","调参期%","检验期%"]))
oirows=[]
for _,r in oif.iterrows(): oirows.append([r.scen,r.case,f"{int(r.S_wan/100)}M",r.cfg,g(r.ann_pct),g(A(r.scen,r.case,int(r.S_wan),r.cfg)["1y_ann_pct"]),g(r.ann_pct-A(r.scen,r.case,int(r.S_wan),r.cfg)["1y_ann_pct"],3)])
oitab=md(pd.DataFrame(oirows,columns=["口径","情景","本金","配置","OI=0已补 一年年化%","原 一年年化%","差(pp)"]))
r_A5A3=A("A","realistic",7000,H5); 
a3cost={k:d(s,cs_,H1) for k,(s,cs_) in dict(A_r=("A","realistic"),B_r=("B","realistic"),A_o=("A","optimistic"),B_o=("B","optimistic")).items()}
a3cost5={k:d(s,cs_,H5,base=H3) for k,(s,cs_) in dict(A_r=("A","realistic"),B_r=("B","realistic"),A_o=("A","optimistic"),B_o=("B","optimistic")).items()}
mA=mech[(mech.scen=="A")]; 
v91=A("A","realistic",7000,H0); a5=A("A","realistic",7000,H3); a53=A("A","realistic",7000,H5)
liqC=cb.liq_cost_wan
txt=f"""# 负费率策略 v10 回测报告（第二版：采纳两位复核意见后）

> 本报告只在本地生成，未推送、未改公开仓库（基线 main c864f47，PDF v9.1）。所有数字由本目录 `code/` 下的脚本生成，原始结果在 `out/`。
> 简体中文；术语均附白话解释（见文末“术语表”）。第一版结论已按两份复核（独立复核 + 回测审查官）修订，修订清单见文末。

## 0. 先看结论（白话）

**一句话主结论：v9.1 + 拆单(A5) + A3（风控必须项）是 v10 的推荐组合；它让“调参期”大幅提升，但“检验期”基本无提升，所以不能把一年的提升当成已被样本外证明。**

1. **主结论 = v9.1 + 拆单(A5) + A3。** 70M 本金、现实情景：
   - {hl('realistic')[0]}
   - {hl('realistic')[1]}
   - 乐观情景：A口径 检验期 {g(A('A','optimistic',7000,H0).test_ann_pct)}% → {g(A('A','optimistic',7000,H5).test_ann_pct)}%，B口径 {g(A('B','optimistic',7000,H0).test_ann_pct)}% → {g(A('B','optimistic',7000,H5).test_ann_pct)}%，**检验期反而更低**。
   - **检验期基本无提升。** 提升几乎全部来自调参的前8个月。各本金档的检验期差见 §3.6。
   - **不能说 v10 全面优于 v9.1；11.13% 必须和检验期 7.89% 一起说。** A口径现实情景，各本金档一年差（v10−v9.1，见 §3.6 与 `out/main_table.csv`）：1M {sg(A('A','realistic',100,H5)['1y_ann_pct']-A('A','realistic',100,H0)['1y_ann_pct'],1)}pp、3M {sg(A('A','realistic',300,H5)['1y_ann_pct']-A('A','realistic',300,H0)['1y_ann_pct'],1)}pp、5M {sg(A('A','realistic',500,H5)['1y_ann_pct']-A('A','realistic',500,H0)['1y_ann_pct'],1)}pp（**本金≤5M 时 v10 一年反而更低**，5M 基本持平）；10M 一年 {sg(A('A','realistic',1000,H5)['1y_ann_pct']-A('A','realistic',1000,H0)['1y_ann_pct'],1)}pp，但检验期 {sg(A('A','realistic',1000,H5).test_ann_pct-A('A','realistic',1000,H0).test_ann_pct,1)}pp。也就是说，在建议的 5–10M 规模里，**提升只出现在调参期，检验期没有**；只有 20M 以上一年提升明显，但检验期差仍约 0（约 +0.04 到 +0.11pp）。
2. **A3 是风控必须项，不是调出来的。** 含义：持仓超过上限（10%×上一小时OI、10%×成交量）时，当小时就减回到上限，不拖到以后。**合规成本**：在 v9.1 上加 A3，70M 现实 A口径 8.38%→{g(A('A','realistic',7000,H1)['1y_ann_pct'])}%（约 −0.14pp，未取整差 {sg(a3cost['A_r'],3)}pp，与复核一致），B口径 {sg(a3cost['B_r'])}pp；叠在拆单之上，A口径现实 {sg(a3cost5['A_r'])}pp、乐观 {sg(a3cost5['A_o'])}pp。不加 A3 的“仅拆单”会长时间超过上限（§3.4 表），所以**不能单独用**。
3. **A1 降为“未验证”**：调参期选出的参数，样本外没有稳定增益，不同本金档符号会变，**不并入主结论**（§3.3）。**A2 过拟合，不采用**（B口径70M 调参期好看，检验期崩，§3.3）。
4. **“只用 A5（不叠 A1）”这个选择是我看过检验期结果之后做的**（第一版按预先规则选的是 A1+A3+A5，复核指出 A1 只是噪声）。因此**检验期里 A5 相对 v9.1 的差，不能当作证据**；它只是事后描述。
5. **拆单(A5)的机制已更正**：不是“开仓次数变少”（开仓次数反而 {int(v91.episodes)}→{int(a5.episodes)} 增加），成交笔数从约 {v91.fills/1e4:.0f} 万笔增到约 {a5.fills/1e4:.0f} 万笔；**增益来自总成交额下降**（年均成交额÷本金 {g(v91.traded_x_S,1)}倍 → {g(a5.traded_x_S,1)}倍），点差+冲击成本 {g(v91.cost_wan,0)}万 → {g(a5.cost_wan,0)}万U，同时资金费收入少了约 {g(v91.funding_wan-a5.funding_wan,0)} 万U（这是仅拆单(A5)的数字；加上 A3 后见 §3.4 表；资金费减少的原因没有进一步拆分验证）。成本假设没有改（每笔每腿仍是固定30bp/20bp）。详见 §3.4。
6. **P（大币正费率，原“B”，现改名 P 以免与公开口径 B 混淆）和 C（BTC/ETH 季度交割基差）：约不赚不亏，取决于成本假设，均低于闲置理财参考利率，故不值得做。** 这里的成本是“全池中位半点差 + 固定30bp 冲击”，**不是 BTC/ETH 等大币的实测成本**；不同成本假设下的数见敏感性表（§5.3、§6.2）。P 的旧数字曾被数据异常（孤立 OI=0）污染，已按缺失处理后重跑。**C 的检验期数字含窗口末强平的一次性成本，要加回约 {g(liqC*1e4/1e7/(119/365)*100,1)}pp（窗口末强平成本，10M 约 {g(liqC,1)} 万U）才可比**（§3.5）。
7. **D（闲置U理财）**：币安活期 USDT 历史利率要带签名的 API Key 才能查，**不是公开数据**，主表按 0%。**另单列一行** Aave v3（以太坊）USDT 公开存款利率 {g(AAVE['y1'])}%（全窗口）。近六月窗口与调参期重叠两个月，该窗口的所有数字**仅参考**。
8. **E（组合）70M 能不能到 15%？** 现实情景不能：A口径 {g(E_('A','realistic','E_noD_y1_ann'))}%，B口径 {g(E_('B','realistic','E_noD_y1_ann'))}%（不含理财）；加 Aave 参考行 {g(E_('A','realistic','E_D_y1_ann'))}% / {g(E_('B','realistic','E_D_y1_ann'))}%，仍不到15%。乐观情景 A口径 {g(E_('A','optimistic','E_noD_y1_ann'))}%（含参考理财 {g(E_('A','optimistic','E_D_y1_ann'))}%）超过15%，但检验期只有 {g(E_('A','optimistic','E_noD_test_ann'))}%。

## 1. 口径与样本划分（先说清楚）

- **同一年窗口**：与 v9.1 完全相同：回测窗口 2025-10-28 → 2026-10-28（不含），数据最后一小时 2026-10-27 23:00 UTC（回测内时间均为 UTC）。任务里写的“后4个月到 2026-11-01”，本地和币安历史数据只到 10-27，所以后 4 个月实际是 **2026-07-01 → 2026-10-27（119天）**，没有向后外推。
- **调参期（样本内）**：2025-10-28 → 2026-06-30（246天）。**检验期（样本外）**：2026-07-01 → 2026-10-27（119天），不回头改参数。
- **近六个月**：2026-05-01 → 2026-10-27（180天）。**5–6 月两个月与调参期重叠，所以它不是样本外，所有“近六月”数字仅参考。**
- **检验期并不干净**：①本报告是在已经看到全年结果后写的（A5 单独、A1 降级都是看过检验期后的取舍）；②第一版和两轮复核都看过检验期数字。所以检验期只能说明“没有额外证据支持提升”，不能当成一次真正的盲测。
- **做法**：整年因果地（不看未来）跑一遍，再把“累计盈亏（含未平仓按市价估值）”在 2026-07-01 处切开，分别算前后两段年化。年化 = 净利 ÷ 本金 S ÷ 年数。单位：万U。
- **成本口径不变**：现实 = OI 滞后1小时、每笔每腿固定冲击30bp、k=4；乐观 = 滞后0、20bp、k=0；成交量上限10%、OI 上限=上一小时 OI 的 10%、每小时执行预算 pov=0.20（A5 改为0.03）；现货手续费 10bp、永续手续费 5bp，另加半个点差（全池中位数，现货约6.5bp、永续约2.2bp）。
- **两种单币上限口径**：A = 私有库存上限（**只报汇总**）；B = 公开口径，每币 S/267。v9.1 的 8.38% 属于 A 口径，10.50% 属于 B 口径。
- **基线复现**：本目录代码重跑 v9.1：A现实 8.38%、B现实 10.50%（与 PDF 口径一致），另得 A乐观 20.43%、B乐观 17.45%；两位复核也独立复现了这些数字。
- **数据异常处理（孤立 OI=0）**：币安小时 OI 在 2026-10-15 的 11:00–15:00 UTC（集中在 13:00、14:00 两个小时）出现大量 0 而不是缺失。引擎把 0 当“未知=不能持仓”，会强平再重进。**连续≤3小时、前一小时有正值的 OI=0 视为缺失，用前值补**（267币池共 519 个币·小时，其中 369 个在 10-15 的 11–15 点，其余 150 个分散在全年约 92 个其他日期（除 10 月外每月 3–40 个）；大币集 15 个，全部在 10-15 的 13:00/14:00，所以“10-15 13:00–14:00”只适用于大币集；统计见 `out/oi_zero_fill_summary.csv`，脚本 `code/oi_zero_audit.py`）。对负费率 A/B 的影响：70M 档约 0.01–0.02pp，A口径 10M 档约 −0.1pp（见 §7.2 表），不影响 70M 主结论，所以 A/B 主表仍用原数据；对 P 影响大，P 全部用修正后的数据重跑。
- **调参选择规则（事先固定，只在调参期上选）**：A5 的 pov 在 {{0.01…0.40}} 里取调参期 A口径现实年化最大者：0.03（`out/a5_pov_grid.csv`）；A3 是合规规则；A1 在 70M 上网格 N∈{{1,2,3,4,6,9}} × 大币门槛 {{0,2000万,5000万,1亿,2亿,5亿}}（`out/a1_grid.csv`），取调参期现实情景 A、B 两口径年化平均最大者 N=3、门槛 2000万U（“大币”= 24小时成交额池内中位数≥2000万U）；A2 网格见 `out/a2_grid.csv`。注意：这些网格范围是看过全年 v9.1 结果后设定的，网格文件里 tune 与 test 同表存放，没有发现用检验期挑参数的证据，但流程上无法排除。pov 只用 A口径现实情景的调参期选（0.03）；B口径现实调参期最优是 0.04、乐观最优是 0.05，没有采用，所以乐观情景下的 pov 不是最优。
- **容量提示（A口径）**：A口径本金超过约 20–30M 后净利几乎不再增加（见 §8.1 负费率占用）（负费率腿受单币上限和成交量/OI 上限限制），多出来的本金在 A口径里对应的是**没卖出的库存币，不是空闲的U**，所以 A口径下“闲置U赚理财”只是假设性的参考行。

## 2. 主表（本金 70M，一年；现实情景）

{mt('realistic')}

### 同一张表：乐观情景

{mt('optimistic')}

说明：
- “峰值占本金%”= 一年里按市价估值的最大占用 ÷ S；大于100%是币价上涨后已有仓位市值超过本金（v9.1 同样如此）。
- **P、C 行：约不赚不亏，取决于成本假设，均低于闲置理财参考利率，故不值得做**；成本是全池中位半点差+固定30bp，不是大币实测。E 行里仍把 P、C 按同一规则放进去，只是为了回答“15%能不能到”。
- E 行：负费率取“调参期利润最大的本金档”，其余本金平分给 P 和 C（二者调参期现实情景都为正才纳入），分配规则在看检验期前就固定。完整分档（1M–70M）见 `out/main_table.csv`。

## 3. A 负费率：各项相对 v9.1（同窗口、同指标）

### 3.1 现实情景：各项相对 v9.1 的增减

{deltas('realistic')}

### 3.2 乐观情景：各项相对 v9.1 的增减

{deltas('optimistic')}

### 3.3 逐项解读（白话）

- **A3 同小时砍回上限（风控必须项）**：只加 A3，70M A口径现实 {sg(a3cost['A_r'])}pp（8.38%→{g(A('A','realistic',7000,H1)['1y_ann_pct'])}%），这就是合规代价（约 −0.14pp）。叠在拆单上代价更大（A口径现实 {sg(a3cost5['A_r'])}pp、乐观 {sg(a3cost5['A_o'])}pp），因为拆单后仓位调整更慢，更需要强制砍回。
- **A5 拆单**：见 §3.4；单独看 A5 的成绩好看，但它没有 A3 时长期超上限，**不能单独使用**。
- **主结论 A5+A3**：70M 现实 A口径 {g(a53['1y_ann_pct'])}%（v9.1 {g(v91['1y_ann_pct'])}%）；调参期 {g(a53.tune_ann_pct)}% vs {g(v91.tune_ann_pct)}%；**检验期 {g(a53.test_ann_pct)}% vs {g(v91.test_ann_pct)}%，基本无提升**。乐观情景 A口径一年 {g(A('A','optimistic',7000,H5)['1y_ann_pct'])}% vs v9.1 {g(A('A','optimistic',7000,H0)['1y_ann_pct'])}%，**乐观下反而更低**。
- **A1 大币更严的入场（连续 N=3 期为负才入场；“大币”=成交额≥2000万U）→ 未验证**：按复核指出，N 数的是资金费“打印次数”，1小时结算的币 3 次=3 小时、8小时结算的币 3 次=24 小时，含义因币而异，邻近参数格的调参期/检验期波动大且符号不稳；在调参期选出的参数，70M 现实 A口径 {sg(A0('A','realistic',7000,'A1 严入场(N=3,≥2000万)')['1y_ann_pct']-v91['1y_ann_pct'])}pp、B口径 {sg(A0('B','realistic',7000,'A1 严入场(N=3,≥2000万)')['1y_ann_pct']-A('B','realistic',7000,H0)['1y_ann_pct'])}pp，检验期几乎为 0，不同本金档符号会变。**不并入主结论，也不说它有效。**
- **A2 按流动性分层的单币上限 → 过拟合，不采用**：B口径70M 调参期 {g(A0('B','realistic',7000,'A2 分层上限(r=2)').tune_ann_pct)}%（v9.1 {g(A('B','realistic',7000,H0).tune_ann_pct)}%），但检验期 {g(A0('B','realistic',7000,'A2 分层上限(r=2)').test_ann_pct)}%（v9.1 {g(A('B','realistic',7000,H0).test_ann_pct)}%）。
- **A4 周表“峰值/受限约束”两列的修正**：见 §4（只修表，不改回测）。
- **旧捆绑 A1+A3+A5（第一版主结论）**：70M 现实 A口径 {g(A('A','realistic',7000,H53)['1y_ann_pct'])}%，检验期 {g(A('A','realistic',7000,H53).test_ann_pct)}%；与 A5+A3 相比差别主要是 A1 的噪声，所以改用 A5+A3。

### 3.4 拆单（A5）的机制——更正后的说法（70M，现实情景，本人复算）

{mechtab('realistic')}

- **做法**：每小时执行预算 pov（占上一小时成交量的最大参与比例）从 0.20 降到 0.03，大仓位要分更多小时、更小笔建仓。
- **观察到的事实（不是推测）**：①开仓次数没有变少（A口径 {int(v91.episodes)}→{int(a5.episodes)}，主结论 {int(a53.episodes)}）；②成交笔数翻了约一倍（{v91.fills/1e4:.1f}万→{a5.fills/1e4:.1f}万；主结论 {a53.fills/1e4:.1f}万笔），平均每笔成交额降到原来的约1/4（{g(mech[(mech.scen=='A')&(mech.cfg==H0)].avg_notional_per_fill_wan.iloc[0],2)}万→{g(mech[(mech.scen=='A')&(mech.cfg==H3)].avg_notional_per_fill_wan.iloc[0],2)}万U）；③**全年总成交额下降**：A口径 {g(v91.traded_x_S,1)}倍本金→{g(a5.traded_x_S,1)}倍（主结论 {g(a53.traded_x_S,1)}倍）；④点差+冲击成本因此从 {g(v91.cost_wan,0)}万降到 {g(a5.cost_wan,0)}万U（主结论 {g(a53.cost_wan,0)}万），**资金费收入同时少了约 {g(v91.funding_wan-a5.funding_wan,0)}万U**（主结论少 {g(v91.funding_wan-a53.funding_wan,0)}万），净利 {g(v91.net_wan,0)}→{g(a5.net_wan,0)}万U（主结论 {g(a53.net_wan,0)}万）。
- **成本假设没有改**：没有客观数据（没有历史盘口深度/逐笔成交）能证明“小单每笔冲击更低”，所以每笔每条腿的冲击仍是固定 30bp（乐观20bp），不因拆细打折。收益完全来自“总成交额下降”，不是“单位冲击变低”。引擎里为什么总成交额会下降（哪类成交减少）本报告没有进一步拆分，只描述上面的事实。
- **为什么 A3 是必须项（本人复算）**：仅拆单时仓位调整太慢，会长时间超过上限；加 A3 后超上限的“币·小时”大幅减少（甚至少于 v9.1）：

{captab()}

### 3.5 检验期逐月表（2026-07-01 → 10-27，70M，现实情景；万U，含未平仓市值）

{montab('realistic')}

- **为什么 A口径 7–9 月合计比 v9.1 少、检验期年化却高 0.04pp（用实际数字核对）**：A5+A3 相对 v9.1 在 7–9 月合计 {sg(sum(a53[m]-v91[m] for m in ('mo_2026-07','mo_2026-08','mo_2026-09')),1)} 万U（即少约 {g(-sum(a53[m]-v91[m] for m in ('mo_2026-07','mo_2026-08','mo_2026-09')),1)} 万U）；10月 {sg(a53['mo_2026-10']-v91['mo_2026-10'],1)} 万U（含窗口末平仓），检验期合计 {sg(sum(a53[m]-v91[m] for m,_ in MON),1)} 万U。年化 = 检验期净利差 ÷ 7000万U ÷ (119/365 年) = {sg(sum(a53[m]-v91[m] for m,_ in MON)/7000/(119/365)*100,2)}pp，分母是整个检验期 119 天（含10月），而 7–9 月只有 92 天，所以两个数不矛盾：差额来自 10 月和年化分母。逐月原始数在 `out/oos_monthly.csv`（脚本 `code/make_oos_monthly.py`）。
- 8、9月旧捆绑（A1+A3+A5）相对 v9.1 分别 {sg(A('A','realistic',7000,H53)['mo_2026-08']-v91['mo_2026-08'],1)} / {sg(A('A','realistic',7000,H53)['mo_2026-09']-v91['mo_2026-09'],1)} 万U（A口径），与复核的“约 −10.7 / −7.8”一致；主结论 A5+A3 为 {sg(a53['mo_2026-08']-v91['mo_2026-08'],1)} / {sg(a53['mo_2026-09']-v91['mo_2026-09'],1)} 万U。7月和10月为正，合计检验期几乎持平。
- **窗口末强平的一次性成本**：回测在 2026-10-28 把仍未平仓的仓位按当时价格强平并扣成本。A口径负费率腿这笔成本只有 {g(A('A','realistic',7000,H5).resid_liq_cost_wan,1)} 万U（≈{g(A('A','realistic',7000,H5).resid_liq_cost_wan/7000/(119/365)*100,2)}pp 检验期年化，可忽略）；**C 腿这笔成本较大：10M 约 {g(liqC,1)} 万U，把检验期年化压低约 {g(liqC*1e4/1e7/(119/365)*100,1)}pp**（C 的检验期数字要加回约这么多才可比；P 的情况：引擎在最后 24 小时主动平仓，成本已计入表中）。
- 乐观情景（下表，同一口径）：A口径 8、9、10月 A5+A3 均低于 v9.1（检验期合计 {g(sum(A('A','optimistic',7000,H5)[m] for m,_ in MON),1)} vs {g(sum(A('A','optimistic',7000,H0)[m] for m,_ in MON),1)} 万U）；B口径主要是 9 月拖累。

{montab('optimistic')}

### 3.6 各本金档：主结论相对 v9.1（现实情景）

{ladder()}

- 大本金档（A口径 20M–70M、B口径各档）检验期差在 0 到 +0.6pp 之间，几乎为零；A口径小本金档（1M–10M）检验期差忽大忽小（−1.4 到 +4.8pp，符号不定，样本很小），增益主要出现在调参期。**A口径本金≤5M 时，主结论一年反而比 v9.1 低（−6.2 到 −1.0pp）**，因为 pov=0.03 是在 70M 上调的，小本金没有必要限速。

## 4. A4：周表（weekly_2026-11-01）峰值与受限约束两列的修正

- **问题1（峰值口径）**：旧表“峰值”按**成本价**算，而上限约束的是**市值**。LSK 现实情景：成本口径峰值约 {lsk.realistic_peak_cost_basis/1e4:.0f} 万U，按市值是 **{lsk.realistic_peak_mtm/1e4:.0f} 万U**（乐观 {lsk.optimistic_peak_mtm/1e4:.0f} 万U）。
- **问题2（受限约束口径）**：旧表把峰值和“整个持仓期的平均上限”比较，改为和**峰值那一小时当时有效的上限**比较。LSK 峰值 {lsk.realistic_peak_mtm/1e4:.0f} 万U ≈ 当时 OI 上限 {lsk.realistic_oi_cap_at_peak/1e4:.0f} 万U（比值 {lsk.realistic_peak_over_oicap:.2f}），**受 OI 上限约束**；成交量上限当时 {lsk.realistic_vol_cap_at_peak/1e4:.0f} 万U，没触到。
- 全表（{len(wk)} 行）现实情景受限约束分布：{ '；'.join(f'{k} {v} 个' for k,v in nb.items()) }。
- 产物：`out/weekly_fix/coin_table_peak_bind_fixed.csv`；脚本 `code/fix_weekly_table.py`。引擎和数据没有改动。

## 5. P 大币正费率（买现货 + 空 1x 永续；原名“B”）

- **结论：约不赚不亏，取决于成本假设，均低于闲置理财参考利率，故不值得做。** 用的成本是**全池中位半点差 + 固定30bp 冲击（乐观20bp），不是 BTC/ETH 等大币的实测成本**。
- **标的**：币安上同时有现货和 U 本位永续的币中，按 CoinGecko 总市值取大币：BTC、ETH、BNB、XRP、SOL、TRX、ZEC、DOGE、LINK、ADA（Set10）；其中 **Set4 = BTC、ETH、SOL、BNB**。数据取自币安公开接口。
- **口径**：现货 10bp + 永续 5bp 手续费、半个点差、30bp/20bp 每笔每腿冲击、OI 上限10%、成交量上限10%、pov 0.20；每币上限 = 本金 ÷ 币数。占用 = 现货 $1 + 空头保证金 $1 = 每 $1 对冲仓位占 $2（1x）。负费率腿对这些币关闭。**P 腿没有“k=4 自身对资金费的冲击”（那个只在负费率腿建模）**，这对 P 偏有利。
- **数据修正**：孤立 OI=0 已按缺失处理（见 §1），整个 P 重新跑了调参网格（`out/p_grid.csv`，{len(pd.read_csv(O+'p_grid.csv'))} 组）、主表（`out/p_main.csv`）和敏感性（`out/p_sens.csv`）。**修正前的旧结果由数据异常造成，已作废，归档在 `superseded/out_r1/`。**
- **调参（仅调参期）**：入场阈值（24小时平均费率）、出场阈值、最长等待小时数（`out/p_grid.csv`）。Set4 调参期最优 = 全年持有、不择时；Set10 调参期最优 ≤ 0（等于“不开仓”），下面 Set10 只列“全年持有”作参考。

### 5.1 Set4（调参期选中规则：全年持有），按本金分档

{tier(P4,pcols,pn)}

### 5.2 Set10（仅参考；调参期最优是“不开仓”）

{tier(P10,pcols,pn)}

- **容量**：Set4 平均占用占本金的比例（{g((P4[P4.case=='realistic'].avg_used_wan/(P4[P4.case=='realistic'].S*100)).min()*100,0)}%–{g((P4[P4.case=='realistic'].avg_used_wan/(P4[P4.case=='realistic'].S*100)).max()*100,0)}%，1M–70M）基本不随本金下降，说明规模不是瓶颈，**收益率才是**。

### 5.3 敏感性：每笔每腿冲击（Set4，10M，现实 lag；半点差仍用全池中位数）

{md(ps[ps.set=='Set4'].round(3).rename(columns=dict(set='组合',impact_bp='冲击(bp)',y1_ann='一年年化%',tune_ann='调参期%',test_ann='检验期%',m6='近六月%(仅参考)',net_wan='一年净利(万U)')))}

没有客观盘口/成交数据证明大币的真实冲击，所以不改成本假设，只作敏感性展示；**无论冲击取多少，这里的数都低于 Aave 参考利率 {g(AAVE['y1'])}%。**

## 6. C 季度交割基差（买现货 + 空币安 U 本位季度合约，持有到交割）

- **结论：约不赚不亏，取决于成本假设，均低于闲置理财参考利率，故不值得做。** 成本同样是全池中位半点差+固定30bp，**非大币实测**。
- **标的与数据**：币安 U 本位季度合约**只有 BTC、ETH 两个**。已下载 2025-12、2026-03、2026-06、2026-09、2026-12 交割合约的小时K线（`data/quarterly/`）。
- **交割规则**：每季度最后一个周五 08:00 UTC 交割；**交割手续费 = 吃单费率 5bp**。结算价用现货交割前最后一小时（07:00–08:00）K线 (开+高+低+收)/4 近似；现货在交割时刻按收盘价卖出，付 10bp 手续费 + 半点差 + 冲击。COIN-M 未纳入。
- **入场规则**：用上一小时收盘价算年化基差，取剩余≥dmin 天的合约中最高者；≥阈值就分批建仓，每小时建仓额 = min(10%×该合约上一小时成交额, 剩余额度÷2)，每币上限 = 本金÷币数；持有到交割。占用 = 每 $1 对冲仓位 $2。
- **参数怎么选的（只在调参期）**：阈值 {{0,2,4,6,8,10%}} × 最少剩余天数 {{7,30}}（`out/c_grid.csv`），取调参期现实与乐观收益均值最大者 = **阈值 4%、剩余≥30天**。**网格里阈值6%/30天的一年和检验期更好，但调参期更低，所以按规则没有选它，也不回头改。**
- **没有 OI 上限**：没取到交割合约的历史 OI，季度腿只有成交量上限。

{tier(c,ccols,cn)}

- 窗口截止 2026-10-28 时仍有 2026-12 合约在手，按“当天强平”扣出场成本（表最后一列）；这是一次性成本，**10M 约 {g(liqC,1)} 万U，把检验期年化压低约 {g(liqC*1e4/1e7/(119/365)*100,1)}pp**；若持有到交割则不发生。
- **容量**：平均占用占本金比例 10M 时 {g(c[(c.case=='realistic')&(c.S==10)].avg_used_wan.iloc[0]/10,0)}%，70M 时 {g(c[(c.case=='realistic')&(c.S==70)].avg_used_wan.iloc[0]/70,0)}%，本金越大占用比例越低，受合约成交量限制。

### 6.1 按本金分档（见上表）与窗口末强平

（上表即分档结果，窗口末强平说明见表后。）

### 6.2 敏感性：成本假设（10M，现实 lag，阈值4%/30天）

{md(cs[['label','hs_spot_bp','hs_perp_bp','impact_bp','y1_ann','tune_ann','test_ann','test_ann_ex_liq','m6_ann','net_wan']].round(3).rename(columns=dict(label='成本假设',hs_spot_bp='现货半点差(bp)',hs_perp_bp='永续半点差(bp)',impact_bp='冲击(bp)',y1_ann='一年年化%',tune_ann='调参期%',test_ann='检验期%',test_ann_ex_liq='检验期%(加回窗口末强平)',m6_ann='近六月%(仅参考)',net_wan='一年净利(万U)')))}

- “大币式半点差”只是敏感性假设（没有大币实测盘口数据支持）。该表说明结果对成本假设高度敏感；成本取最低（点差冲击全为0，只剩手续费）一年也只有 {g(cs.iloc[-1].y1_ann)}%，低于 Aave 参考利率。

## 7. D 闲置U理财（单列）与其他敏感性

### 7.1 D 闲置U理财

- **目标来源**：币安“活期理财（Simple Earn Flexible）USDT”历史利率接口 `/sapi/v1/simple-earn/flexible/history/rateHistory` **必须带用户 API Key 和签名**，**不是公开数据**；我不使用账户密钥去取，所以**无法客观取得 → 主表按 0%**。
- **单列参考行（不并入主结论）**：公开数据 DeFiLlama，Aave v3 以太坊 USDT 池（pool f981a304-bb6c-45b8-b0c5-fd2f515ad23a）逐日基础存款利率：

| 窗口 | Aave v3 USDT 平均存款利率 |
|---|---|
| 全窗口 2025-10-28→2026-10-27 | **{g(AAVE['y1'])}%/年** |
| 调参期 | {g(AAVE['tune'])}% |
| 检验期 | {g(AAVE['test'])}% |
| 近六月（与调参期重叠，**仅参考**） | {g(AAVE['m6'])}% |

  数据：`out/idle_yield_aave_v3_usdt_daily.csv`。
- **提醒**：这是链上借贷协议，要把U转出币安，承担智能合约和跨平台风险，不等同于币安内闲置U。闲置本金 = 本金 − 各策略一年平均占用；A口径下闲置的是未卖出的库存币，不是U（见 §1 容量提示）。

### 7.2 敏感性（只在此处出现，不进主结论）

**(a) 平方根冲击（冲击随下单量变大而变大的另一种模型）**：每腿冲击 = 系数 × 日波动率 × √(下单量÷当天成交额)，替代固定30bp；**系数没有任何数据可以标定，这里只是看结果对冲击模型有多敏感**。A口径 70M 现实：

{sqtab}

系数0.5时 v9.1 一年只剩 {g(sq[(sq.cfg==H0)&(sq.tag=='sqrt_y=0.5')].ann_pct.iloc[0])}%，系数1时为 {g(sq[(sq.cfg==H0)&(sq.tag=='sqrt_y=1.0')].ann_pct.iloc[0])}%（亏损）；拆单（总成交额小）在这种模型下相对更有利，但绝对水平完全取决于未标定的系数，**不作为结论依据**。

**(b) 孤立 OI=0 补值对负费率 A/B 的影响**（补前 vs 补后，其余完全相同）：

{oitab}

补值对负费率 A/B 的影响（以上表为准）：70M 档（现实/乐观）约 0.01–0.02pp；10M 档现实情景 A口径约 −0.10pp、B口径约 +0.02–0.04pp。A口径 10M 的 −0.1pp 不算可忽略，但主结论是 70M，且该档在主表中仍用原数据，所以主表沿用原数据。

补值范围：267币池共 519 个币·小时，其中 369 个集中在 2026-10-15 的 11–15 点（13:00、14:00 两个小时占 349 个），其余 150 个分散在全年；大币集的 15 个全部在 10-15 的 13:00/14:00。

## 8. E 组合：主结论负费率 + P + C（+ D 参考）

分配规则（看检验期之前固定）：①负费率取“调参期利润最大”的本金档（≤总本金）；②剩余本金平分给 P 和 C（二者调参期现实都为正才纳入）；③用不到的本金视为闲置。A口径的负费率调参期利润最大档为 20M，与“A口径容量约20–30M”的提示一致。

### 8.1 现实情景 · A口径（私有库存，仅汇总）
{etab('A','realistic')}

### 8.2 现实情景 · B口径（公开 S/267）
{etab('B','realistic')}

### 8.3 乐观情景 · A口径
{etab('A','optimistic')}

### 8.4 乐观情景 · B口径
{etab('B','optimistic')}

**结论**：
- 70M 现实情景：A口径 {g(E_('A','realistic','E_noD_y1_ann'))}%，B口径 {g(E_('B','realistic','E_noD_y1_ann'))}%，**达不到15%**；加 Aave 参考理财行 {g(E_('A','realistic','E_D_y1_ann'))}% / {g(E_('B','realistic','E_D_y1_ann'))}%，**仍达不到**。
- 70M 乐观情景：A口径 {g(E_('A','optimistic','E_noD_y1_ann'))}%，B口径 {g(E_('B','optimistic','E_noD_y1_ann'))}%（含参考理财 {g(E_('A','optimistic','E_D_y1_ann'))}% / {g(E_('B','optimistic','E_D_y1_ann'))}%）；只有 A口径超过15%，检验期仅 {g(E_('A','optimistic','E_noD_test_ann'))}%，近六月 {g(E_('A','optimistic','E_noD_m6_ann'))}%（仅参考）。
- 15% 的缺口主要取决于负费率腿本身的容量；P、C 对总收益约为零，且低于理财参考利率，不值得做。

## 9. 缺口与局限（请如实看待）

1. **日期**：检验期实际到 2026-10-27，不是 11-01。
2. **检验期不干净**：A5 单独、A1 降级的取舍是看过检验期之后做的；近六月与调参期重叠两个月，仅参考。
3. **样本外只有约4个月、一个行情段**，结论是“检验期基本无提升”，不是“已证明有效”。
4. **拆单**：成本假设没改；增益来自总成交额下降（模型内效应），实盘真实冲击要靠成交回执验证；pov=0.03 只在 70M、A口径现实情景上调参（A口径本金≤5M 时一年反而更低，§3.6；B口径现实/乐观的调参期最优 pov 不同，未采用）。
5. **A1 未验证、A2 过拟合未采用。**
6. **P、C 的成本不是大币实测**：半点差用全池中位数（现货约6.5bp、永续约2.2bp），冲击固定30bp/20bp；P 没有 k=4 的自身冲击建模；C 无 OI 上限、结算价近似、COIN-M 未纳入、窗口末强平一次性成本较大（§6）。
7. **A 口径为私有库存，只给汇总**；公开结论以 B 口径为准。
8. **平方根冲击系数未标定**，仅敏感性。
9. **D**：币安活期历史利率非公开 → 0%；参考行是链上 Aave，口径不同。
10. **组合 E**：假设各策略资金互不占用、盈亏相加；保证金交叉占用等未建模。
11. 脚本里有 `/workspace/...` 绝对路径（数据目录路径），换机器需要改；复现命令见各脚本头部。

## 10. 修订清单（相对第一版）

1. 主结论由 A1+A3+A5 改为 **v9.1+A5+A3**；A3 标为风控必须项并写明合规成本；A1 降为未验证；A2 不采用；A5+A3 的一年/调参期/检验期/近六月数字已重跑（A、B口径，现实/乐观，全部本金档）；写明“只 A5”选择在看过检验期后做出，检验期基本无提升。
2. 更正拆单机制：开仓次数不变少，成交笔数翻倍，总成交额下降带来成本下降，资金费收入同时减少（§3.4，本人复算）。
3. 孤立 OI=0 按缺失处理后重跑；“B（正费率）”改名 **P**；P、C 结论改为“约不赚不亏…不值得做”；正文不再写具体亏损幅度和“盈亏平衡冲击”之类措辞，数字只留在表格/敏感性表；说明成本非大币实测。
4. 新增检验期逐月表（§3.5）；写明窗口末强平一次性成本对 C 检验期年化约 −1.3pp。
5. 平方根冲击只放敏感性表，注明系数未标定。
6. 闲置理财：Aave 单列一行；近六月与调参期重叠，标“仅参考”。
7. 推送前小修（不改任何主数字）：主结论补“v10 不是全面优于 v9.1、5–10M 提升只在调参期、11.13% 须与检验期 7.89% 同报”；C 的窗口末强平 +1.3pp 提示写进主结论；逐月表补 10 月说明与 `out/oos_monthly.csv` 的 10 月列；OI=0 补值影响的措辞改成与 §7.2 表一致并核实 519 个币·小时的分布；“P 窗口末”措辞更正；旧口径脚本和第一版结果移到 `superseded/`；`main_table.csv` 版本标签统一。
8. 另外补充：C 参数选择说明、A口径容量与闲置说明、检验期不干净、P 无 k=4、脚本绝对路径。

## 11. 产物清单（`/workspace/funding-strategy/v10/`）

- `code/`：`engine.py` / `run_backtest.py`（新增 n_streak、big_thr、derisk_force、镜像腿 ok_m/cap_m 等参数；默认值下重跑出 v9.1 的 8.38%/10.50%）；`v10lib.py`（样本切分、`fix_oi_zero`）；`run_final_a.py`（A 主跑+机制+逐月+平方根+OI敏感性）、`run_a_main.py`/`exp_a1/a2/a3.py`/`exp_pov.py`（第一版 A 与参数网格）；`run_p.py`（P）；`run_c.py`、`run_c_main.py`、`run_c_final.py`（C）；`run_e.py`（E）；`fix_weekly_table.py`（A4）；`make_report.py`、`write_report.py`、`make_oos_monthly.py`（逐月表）、`oi_zero_audit.py`（OI=0 补值统计）；第一版/旧口径脚本已移到 `superseded/`（不属于公开结果）。
- `out/`：`main_table.csv`、`a_final.csv`（含机制列与逐月列）、`a_mechanism.csv`、`oos_monthly.csv`、`a_sqrt_sens.csv`、`a_oifix_sens.csv`、`a_main.csv`（A1/A2 数据来源）、`a1/a2/a3/a5 网格`、`p_grid.csv`、`p_main.csv`、`p_sens.csv`、`c_grid.csv`、`c_main.csv`、`c_sens.csv`、`c_cost_sens.csv`、`e_main.csv`、`idle_yield_aave_v3_usdt_daily.csv`、`weekly_fix/…`、`oi_zero_fill_summary.csv`；第一版被替换的 B 结果与主表已移到 `superseded/out_r1/`。
- `data/`：CoinGecko 前40、exchangeInfo、季度合约小时K线、DeFiLlama 池列表。

## 12. 术语表（白话）

- **负费率**：永续资金费为负时空头付钱给多头；我们卖现货、持1x永续多头，吃这笔费。
- **P（大币正费率）**：反过来，用自己的U买现货、空1x永续，吃正的资金费。
- **公开口径 B**：负费率的单币上限=S/267（与 P 无关）。
- **季度交割基差（C）**：买现货、空季度合约，价差到期收敛。
- **每笔每腿冲击**：我们的单子自己把价格推偏的损失（1bp=0.01%）。
- **半点差**：买卖价差的一半。
- **OI（持仓量）上限**：仓位不超过上一小时全市场该合约持仓量的10%。
- **pov（执行预算）**：每小时成交不超过上一小时成交量的这个比例。
- **A3（同小时砍回）**：超上限就在同一小时减回上限。
- **调参期/检验期**：用来选参数的时段/选完后只看结果、不再改参数的时段。
- **年化**：净利÷本金÷年数。**S**：总本金。**占用**：已被仓位占住的本金（含市值变化）。
"""
open("../REPORT.md","w",encoding="utf-8").write(txt); print(len(txt))
