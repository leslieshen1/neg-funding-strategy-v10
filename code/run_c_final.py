"""C (BTC/ETH quarterly basis): v10 r2 cost-sensitivity table (10M, realistic lag). The base cost = pool-median half-spread + fixed 30bp impact is NOT measured for BTC/ETH.
Rows: base; pool half-spread with lower fixed impact; 'majors-like' half-spread 0.5/0.2bp + impact 5bp / 2bp; zero spread & impact (fees only). Also end-of-window forced-exit cost."""
import run_c as RC
from run_c import *
data = get_data(["BTC", "ETH"]); S = 1e7; rows = []
HS0 = (RC.HS_S, RC.HS_F)
for lab, hs, imp in (("基准: 全池中位半点差+固定冲击30bp", HS0, 0.003), ("乐观: 全池半点差+固定冲击20bp", HS0, 0.002), ("全池半点差+冲击10bp", HS0, 0.001), ("全池半点差+冲击5bp", HS0, 0.0005),
                     ("大币式半点差(现货0.5bp/永续0.2bp)+冲击5bp", (5e-5, 2e-5), 0.0005), ("大币式半点差+冲击2bp", (5e-5, 2e-5), 0.0002), ("点差和冲击全为0(只剩手续费)", (0.0, 0.0), 0.0)):
    RC.HS_S, RC.HS_F = hs
    cum, used, inf = simulate(S, ["BTC", "ETH"], 0.04, 30, imp, data=data); m = slice_m(cum, used, S)
    ytest = (H - int((SPLIT - T0) / pd.Timedelta(hours=1))) / 24 / 365
    rows.append(dict(label=lab, hs_spot_bp=hs[0] * 1e4, hs_perp_bp=hs[1] * 1e4, impact_bp=imp * 1e4, **m, liq_cost_wan=inf["liq_cost_wan"], test_ann_ex_liq=m["test_ann"] + inf["liq_cost_wan"] * 1e4 / S / ytest * 100))
RC.HS_S, RC.HS_F = HS0
df = pd.DataFrame(rows); df.to_csv("../out/c_cost_sens.csv", index=False); pd.set_option("display.width", 250); print(df.round(3).to_string(index=False))
g = pd.read_csv("../out/c_grid.csv"); x = g[(g.coins == "BTC+ETH") & (g.dmin == 30)].pivot(index="thr", columns="case", values=["tune_ann", "test_ann", "y1_ann"]); print(x.round(3))
