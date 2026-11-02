#!/usr/bin/env python3
"""(Optional) Download USDT-M perp order-book depth snapshots (Binance Vision `bookDepth`, daily zips)
and reduce them to hourly depth:  depth/{FUT}.csv  with columns  hour, cap02_p50, cap1_p50
  cap02_p50 / cap1_p50 = hourly median of min(bid, ask) notional (USDT) within +-0.2% / +-1% of mid.

Raw zips are streamed and discarded (each is ~0.5 MB; 90 symbols x ~210 days ~ 19k files ~ 10 GB of traffic).
The OPS execution model uses this depth only to cap the size of each fill (dfrac x depth within 2%).
If you skip this step, build_dataset.py falls back to a volume-based depth proxy (see README).

Examples
  python download_depth.py --workers 16
  python download_depth.py --symbols ACEUSDT,BTCUSDT --start 2026-09-01
"""
import argparse, io, time, zipfile
import concurrent.futures as cf
from datetime import date, timedelta
import numpy as np, pandas as pd, requests
import config as C

BASE = f"{C.BINANCE_VISION}/futures/um/daily/bookDepth"
S = requests.Session()
S.headers.update({"User-Agent": "neg-funding-backtest/1.0"})


def hourly_cap(df, pct):
    sub = df[df.percentage.isin([-pct, pct])]
    if sub.empty:
        return None
    sub = sub.assign(side=np.where(sub.percentage < 0, "bid", "ask"))
    pv = sub.pivot_table(index="timestamp", columns="side", values="notional", aggfunc="last")
    if "bid" not in pv or "ask" not in pv:
        return None
    cap = pv[["bid", "ask"]].min(axis=1).dropna()
    return cap.groupby(cap.index.floor("h")).median() if len(cap) else None


def fetch_day(sym, day):
    url = f"{BASE}/{sym}/{sym}-bookDepth-{day.isoformat()}.zip"
    for i in range(4):
        try:
            r = S.get(url, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                time.sleep(0.5 * (i + 1)); continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                df = pd.read_csv(zf.open(zf.namelist()[0]))
            df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
            df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            c02, c1 = hourly_cap(df, 0.2), hourly_cap(df, 1.0)
            if c02 is None and c1 is None:
                return None
            out = pd.DataFrame({"cap02_p50": c02, "cap1_p50": c1})
            out.index.name = "hour"
            return out.reset_index()
        except Exception:
            time.sleep(0.5 * (i + 1))
    return None


def process(sym, ddir, start, end_excl, workers):
    path = ddir / "depth" / f"{sym}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path, parse_dates=["hour"]) if path.exists() and path.stat().st_size > 50 else None
    have = set()
    if old is not None:
        have = set(pd.to_datetime(old.hour, utc=True).dt.date)
    days, d = [], start
    while d < end_excl:
        if d not in have: days.append(d)
        d += timedelta(days=1)
    parts = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for g in ex.map(lambda x: fetch_day(sym, x), days):
            if g is not None: parts.append(g)
    if old is not None: parts.append(old)
    if not parts:
        return sym, 0
    out = pd.concat(parts)
    out["hour"] = pd.to_datetime(out["hour"], utc=True)
    out = out.drop_duplicates("hour", keep="last").sort_values("hour")
    out.to_csv(path, index=False)
    return sym, len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", default=C.DATA_START)
    ap.add_argument("--end", default=C.DATA_END, help="exclusive")
    ap.add_argument("--workers", type=int, default=16, help="parallel day downloads per symbol")
    a = ap.parse_args()
    ddir = C.data_dir(a.data_dir)
    sy = C.load_symbols(a.symbols_file)
    if a.symbols: sy = sy[sy.fut_symbol.isin(a.symbols.split(","))]
    if a.limit: sy = sy.head(a.limit)
    start, end_excl = date.fromisoformat(a.start), date.fromisoformat(a.end)
    print(f"depth: {len(sy)} symbols {start}..{end_excl} -> {ddir/'depth'}", flush=True)
    t0 = time.time()
    for i, s in enumerate(sy.fut_symbol, 1):
        sym, n = process(s, ddir, start, end_excl, a.workers)
        print(f"[{i}/{len(sy)}] {sym} hours={n} elapsed={time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
