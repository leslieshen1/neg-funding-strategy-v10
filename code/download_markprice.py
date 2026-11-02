#!/usr/bin/env python3
"""Download hourly USDT-M MARK PRICE klines (public data.binance.vision) for the pool -> <data-dir>/mark_klines/{FUT}.csv

Used by build_dataset.py: funding payments are valued at the mark price at settlement time (the open of the hour whose
first instant is the settlement), not at a later close.   Same monthly-zip (+ daily fallback) logic as download_data.py.
  python download_markprice.py [--data-dir D] [--symbols-file F] [--workers 8]
"""
import argparse, calendar, concurrent.futures as cf
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import config as C
from download_data import get_zip, parse_klines, month_iter, covered


def fetch_mark(symbol, start: date, end_excl: date):
    base = f"{C.BINANCE_VISION}/futures/um"
    frames = []
    for y, m in month_iter(start, end_excl):
        mstart = date(y, m, 1); mend = date(y, m, calendar.monthrange(y, m)[1])
        raw, _ = get_zip(f"{base}/monthly/markPriceKlines/{symbol}/1h/{symbol}-1h-{y}-{m:02d}.zip")
        if raw is not None:
            frames.append(parse_klines(raw)); continue
        d = max(mstart, start)
        while d <= mend and d < end_excl:
            raw, _ = get_zip(f"{base}/daily/markPriceKlines/{symbol}/1h/{symbol}-1h-{d.isoformat()}.zip")
            if raw is not None: frames.append(parse_klines(raw))
            d += timedelta(days=1)
    if not frames: return None
    return pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")


def one(sym, ddir, start, end_excl, force):
    p = ddir / "mark_klines" / f"{sym}.csv"; p.parent.mkdir(parents=True, exist_ok=True)
    if not force and covered(p, "open_time", end_excl, 2): return sym, "cached", 0
    df = fetch_mark(sym, start, end_excl)
    if df is None: return sym, "missing", 0
    df[["open_time", "open", "high", "low", "close"]].to_csv(p, index=False)
    return sym, "ok", len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None); ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); ddir = C.data_dir(a.data_dir); sym = C.load_symbols(a.symbols_file)
    start = date.fromisoformat(C.DATA_START); end = date.fromisoformat(C.DATA_END)
    rows = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda s: one(s, ddir, start, end, a.force), sym.fut_symbol):
            rows.append(r)
    st = pd.DataFrame(rows, columns=["fut", "status", "rows"]); st.to_csv(ddir / "mark_download_status.csv", index=False)
    print(st.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
