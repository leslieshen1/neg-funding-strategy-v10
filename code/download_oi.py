#!/usr/bin/env python3
"""Download daily open interest (USDT value) for the USDT-M perps (public data only).

Source: https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{date}.zip  (5-minute snapshots)
Daily value = median of that day's 5-minute `sum_open_interest_value`.
Output: <data-dir>/oi/{FUT}.csv   (date, oi_usdt)   -- incremental: only missing days are fetched (NaN = not published).

build_dataset.py shifts this by one day (the OI of day d-1 is the limit input for day d), so no same-day information is used.
"""
import argparse, io, zipfile
import concurrent.futures as cf
from datetime import date, timedelta
import numpy as np, pandas as pd, requests
import config as C

S = requests.Session(); S.headers.update({"User-Agent": "neg-funding-backtest/1.0"})


def fetch_day(sym, d):
    url = f"{C.BINANCE_VISION}/futures/um/daily/metrics/{sym}/{sym}-metrics-{d.isoformat()}.zip"
    for _ in range(3):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404: return sym, d, np.nan
            if r.status_code != 200: continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]))
            return sym, d, float(pd.to_numeric(df["sum_open_interest_value"], errors="coerce").median())
        except Exception:
            continue
    return sym, d, None          # transient failure: leave missing so a re-run retries it


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None); ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--start", default=C.DATA_START); ap.add_argument("--end", default=C.DATA_END, help="exclusive")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    ddir = C.data_dir(a.data_dir); (ddir / "oi").mkdir(exist_ok=True)
    sy = C.load_symbols(a.symbols_file)
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [d0 + timedelta(i) for i in range((d1 - d0).days)]
    have, jobs = {}, []
    for f in sy.fut_symbol:
        p = ddir / "oi" / f"{f}.csv"
        have[f] = {}
        if p.exists():
            x = pd.read_csv(p); have[f] = {date.fromisoformat(str(r.date)): r.oi_usdt for r in x.itertuples()}
        jobs += [(f, d) for d in days if d not in have[f]]
    print(f"{len(jobs)} OI days to fetch", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, (f, d, v) in enumerate(ex.map(lambda j: fetch_day(*j), jobs, chunksize=16), 1):
            if v is not None: have[f][d] = v
            if i % 5000 == 0: print(f"{i}/{len(jobs)}", flush=True)
    for f, m in have.items():
        pd.DataFrame(sorted(m.items()), columns=["date", "oi_usdt"]).to_csv(ddir / "oi" / f"{f}.csv", index=False)
    print("done ->", ddir / "oi")


if __name__ == "__main__":
    main()
