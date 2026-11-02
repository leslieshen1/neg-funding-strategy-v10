#!/usr/bin/env python3
"""Build the PRIVATE per-symbol inventory cap from your holdings workbook.  The output must NOT be committed (it is git-ignored).

Writes fut_symbol, cap_usdt, qty  (PRIVATE, git-ignored).
  qty      = 'Borrowable Qty @ 30% (Token)' used directly (no scaling): the engine keeps a TOKEN ledger (v8) - remaining sellable coins = qty - coins already sold
  cap_usdt = qty x last spot close BEFORE config.DATA_END (clipped to the data window; only legacy/diagnostic use - the v8 engine does not price the cap with a window-end price)

  python make_holdings_cap.py --holdings /path/holdings.xlsx --out holdings_cap.csv

Expected holdings columns: 'Binance Symbols' (perp symbol), 'Borrowable Qty @ 30% (Token)'.
Only symbols of the pool (symbols_signal.csv) are written. Nothing from this script's input is stored in the repo.
"""
import argparse, sys
import numpy as np, pandas as pd
import config as C

QTY = "Borrowable Qty @ 30% (Token)"; SYM = "Binance Symbols"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdings", required=True); ap.add_argument("--out", default="holdings_cap.csv")
    ap.add_argument("--data-dir", default=None); ap.add_argument("--symbols-file", default=str(C.REPO / "symbols_signal.csv"))
    a = ap.parse_args()
    ddir = C.data_dir(a.data_dir)
    h = pd.read_excel(a.holdings)
    h = h[h[SYM].notna()]
    qty = h.groupby(SYM)[QTY].sum().astype(float)
    sy = C.load_symbols(a.symbols_file); rows = []
    for r in sy.itertuples():
        p = ddir / "spot_klines" / f"{r.spot_symbol}.csv"
        if r.fut_symbol not in qty.index or not p.exists(): continue
        kl = pd.read_csv(p, usecols=["open_time", "close"]); kl = kl[kl.open_time < int(pd.Timestamp(C.DATA_END, tz="UTC").value // 10**6)]
        px = float(kl.sort_values("open_time").close.iloc[-1])
        rows.append((r.fut_symbol, float(qty[r.fut_symbol]) * px, float(qty[r.fut_symbol])))
    out = pd.DataFrame(rows, columns=["fut_symbol", "cap_usdt", "qty"])
    out.to_csv(a.out, index=False)
    print(f"{len(out)} of {len(sy)} pool symbols have inventory -> {a.out}  (PRIVATE: do not commit)")

if __name__ == "__main__":
    main()
