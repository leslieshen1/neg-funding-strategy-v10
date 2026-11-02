#!/usr/bin/env python3
"""Download HOURLY open interest (USDT value) for the USDT-M perps (public data only).

Why not the API: `openInterestHist` (period=1h) only reaches back ~30 days, so a 1-year hourly series cannot be rebuilt from it.
Source: https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{date}.zip  (5-minute snapshots, UTC)
Hourly value of hour h = the LAST 5-minute `sum_open_interest_value` snapshot whose create_time lies in [h, h+1h)  (the freshest value known at h+1h).
Output: <data-dir>/oi_hourly/{FUT}.csv   (hour, snap_time, oi_usdt)  -- incremental: only days not yet fetched are requested.
        <data-dir>/oi_hourly/{FUT}.nodata   dates for which Binance Vision has no file (HTTP 404), so re-runs do not retry them.
        <data-dir>/oi_hourly_coverage.csv   per-symbol coverage summary (written at the end).

build_dataset.py turns this into `oih`[T, N]: the OI of the PREVIOUS completed hour, so the limit at hour t uses only data up to t.
Missing hours carry the last known value forward (same convention as the daily version); before the first value -> NaN = unknown -> cannot hold.
"""
import argparse, io, zipfile
import concurrent.futures as cf
from datetime import date, timedelta
import numpy as np, pandas as pd, requests
import config as C

S = requests.Session(); S.headers.update({"User-Agent": "neg-funding-backtest/1.0"})
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))


def fetch_day(sym, d):
    """-> (sym, d, DataFrame[hour, snap_time, oi_usdt]) | (sym, d, 'nodata') | (sym, d, None) on transient failure"""
    url = f"{C.BINANCE_VISION}/futures/um/daily/metrics/{sym}/{sym}-metrics-{d.isoformat()}.zip"
    for _ in range(4):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404: return sym, d, "nodata"
            if r.status_code != 200: continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]), usecols=["create_time", "sum_open_interest_value"])
            df["t"] = pd.to_datetime(df["create_time"]); df["v"] = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
            df = df.dropna(subset=["v"]).sort_values("t")
            if df.empty: return sym, d, "nodata"
            df["hour"] = df["t"].dt.floor("h")
            g = df.groupby("hour").tail(1)                      # last snapshot of each hour
            return sym, d, pd.DataFrame({"hour": g["hour"].dt.strftime("%Y-%m-%d %H:00:00"), "snap_time": g["t"].dt.strftime("%Y-%m-%d %H:%M:%S"), "oi_usdt": g["v"].to_numpy()})
        except Exception:
            continue
    return sym, d, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None); ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--start", default=C.DATA_START); ap.add_argument("--end", default=C.DATA_END, help="exclusive")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    ddir = C.data_dir(a.data_dir); od = ddir / "oi_hourly"; od.mkdir(exist_ok=True)
    sy = C.load_symbols(a.symbols_file)
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [d0 + timedelta(i) for i in range((d1 - d0).days)]
    frames, nodata, jobs = {}, {}, []
    for f in sy.fut_symbol:
        p, pn = od / f"{f}.csv", od / f"{f}.nodata"
        frames[f] = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["hour", "snap_time", "oi_usdt"])
        have = set(frames[f]["hour"].astype(str).str[:10]) if len(frames[f]) else set()
        nodata[f] = set(pn.read_text().split()) if pn.exists() else set()
        jobs += [(f, d) for d in days if d.isoformat() not in have and d.isoformat() not in nodata[f]]
    print(f"{len(jobs)} symbol-days to fetch", flush=True)
    new = {f: [] for f in frames}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, (f, d, v) in enumerate(ex.map(lambda j: fetch_day(*j), jobs, chunksize=16), 1):
            if isinstance(v, pd.DataFrame): new[f].append(v)
            elif v == "nodata": nodata[f].add(d.isoformat())
            if i % 5000 == 0: print(f"{i}/{len(jobs)}", flush=True)
    cov = []
    for f in frames:
        x = pd.concat([frames[f]] + new[f], ignore_index=True).drop_duplicates("hour").sort_values("hour") if (new[f] or len(frames[f])) else frames[f]
        x.to_csv(od / f"{f}.csv", index=False)
        (od / f"{f}.nodata").write_text("\n".join(sorted(nodata[f])))
        h = pd.to_datetime(x["hour"]) if len(x) else pd.Series([], dtype="datetime64[ns]")
        cov.append(dict(fut_symbol=f, hours=len(x), first=(h.min() if len(h) else None), last=(h.max() if len(h) else None), nodata_days=len(nodata[f])))
    pd.DataFrame(cov).to_csv(ddir / "oi_hourly_coverage.csv", index=False)
    print("done ->", od)


if __name__ == "__main__":
    main()
