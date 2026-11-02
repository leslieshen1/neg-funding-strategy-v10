#!/usr/bin/env python3
"""Robustness tables: (1) first half vs second half of the 1y window, (2) headline excluding the top-5 / top-10 most profitable coins.

(1) the 1y window is cut into two equal halves (~182.5 days each); net incl. mark-to-market of open positions, annualised on S per half.
(2) coins are ranked by their own episode PnL (price + funding - fees, i.e. NET) in the BASE run of that window; the top-k are then removed from the pool
    (ok=0) and the whole portfolio is re-simulated (per-coin caps unchanged; freed capital can be re-used by the remaining coins).  Ranking is ex-post by construction:
    this is a concentration check, not a tradable rule.  No coin names or per-coin numbers are written.
Writes results/robustness_<cap>.csv.
"""
import argparse
import numpy as np, pandas as pd
import config as C
import run_backtest as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None); ap.add_argument("--S", type=float, default=7e7)
    ap.add_argument("--cap-mode", choices=["equal", "holdings"], default="equal"); ap.add_argument("--holdings-cap", default=None)
    ap.add_argument("--out-dir", default=str(C.REPO / "results"))
    a = ap.parse_args()
    D = R.Data(C.data_dir(a.data_dir)); S = a.S; ok0 = R.load_ok(D); P = R.base_params(); rows = []
    for w in ("1y", "6m"):
        cap = R.make_cap(D, ok0, S, a.cap_mode, a.holdings_cap)
        r = D.run(w, S, ok0, cap, P); m = R.metrics(r)
        rows.append(dict(test="base", window=w, part="full", net_wan=m["net_wan"], ann_on_S_pct=m["ann_on_S_pct"], avg_used_wan=m["avg_used_wan"],
                         ann_on_used_pct=m["ann_on_used_pct"], coins_removed=0))
        if w == "1y":
            cum = np.concatenate([[0.0], r["cum"]]); H = len(cum) - 1; e = [0, H // 2, H]
            for i, nm in enumerate(("first half", "second half")):
                lo, hi = e[i], e[i + 1]; yrs = (hi - lo) / 8760.0; net = cum[hi] - cum[lo]; au = r["used"][lo:hi].mean()
                rows.append(dict(test="halves", window=w, part=nm, net_wan=net / 1e4, ann_on_S_pct=net / S / yrs * 100, avg_used_wan=au / 1e4,
                                 ann_on_used_pct=net / au / yrs * 100, coins_removed=0))
        pnl = R.coin_pnl(r, D.N, net=True)         # ranked by NET pnl (price + funding - fees)
        order = np.argsort(-pnl)
        for k in (5, 10):
            ok = ok0.copy(); ok[order[:k]] = 0.0
            mk = R.metrics(D.run(w, S, ok, cap, P))   # per-coin cap unchanged (S/267 for equal cap)
            rows.append(dict(test=f"ex_top{k}", window=w, part="full", net_wan=mk["net_wan"], ann_on_S_pct=mk["ann_on_S_pct"],
                             avg_used_wan=mk["avg_used_wan"], ann_on_used_pct=mk["ann_on_used_pct"], coins_removed=k))
    df = pd.DataFrame(rows).round(2)
    out = __import__("pathlib").Path(a.out_dir); out.mkdir(exist_ok=True)
    df.to_csv(out / f"robustness_{a.cap_mode}.csv", index=False); print(df.to_string(index=False))


if __name__ == "__main__":
    main()
