#!/usr/bin/env python3
"""Download hourly spot klines, USDT-M perp klines and perp funding history (public data only).

Sources
  * https://data.binance.vision  (monthly zips; daily zips as fallback for the current month)
  * funding for months not yet published on Vision: public REST  GET /fapi/v1/fundingRate
    (tried on fapi.binance.com, then www.binance.com; if you are in a restricted region use a proxy/VPN)

Output (under --data-dir, default ./data):
  spot_klines/{SPOT}.csv   fut_klines/{FUT}.csv   funding/{FUT}.csv

Examples
  python download_data.py                       # all 267 pool symbols, 2025-10-01 .. 2026-10-28
  python download_data.py --symbols ACEUSDT,BTCUSDT --limit 3
"""
import argparse, io, time, zipfile, calendar
import concurrent.futures as cf
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pandas as pd, requests
import config as C

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
FAPI_HOSTS = ["https://fapi.binance.com", "https://www.binance.com"]
S = requests.Session()
S.headers.update({"User-Agent": "neg-funding-backtest/1.0"})


def get(url, params=None, retries=4):
    for i in range(retries):
        try:
            r = S.get(url, params=params, timeout=90)
            if r.status_code == 404:
                return None, 404
            if r.status_code == 200:
                return r, 200
            time.sleep(0.7 * (i + 1))
        except requests.RequestException:
            time.sleep(0.7 * (i + 1))
    return None, -1


def get_zip(url):
    r, st = get(url)
    if r is None:
        return None, st
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            return zf.read(zf.namelist()[0]), 200
    except Exception:
        return None, -2


def month_iter(start: date, end_excl: date):
    y, m = start.year, start.month
    while date(y, m, 1) < end_excl:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def parse_klines(raw: bytes) -> pd.DataFrame:
    first = raw.split(b"\n", 1)[0].decode("utf-8", "replace").lower()
    if "open_time" in first:
        df = pd.read_csv(io.BytesIO(raw))
        df = df.rename(columns={"count": "n_trades", "taker_buy_volume": "taker_buy_base",
                                "taker_buy_quote_volume": "taker_buy_quote"})
    else:
        df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
    df = df[[c for c in KLINE_COLS if c in df.columns]].copy()
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    ot = df["open_time"].astype("int64")
    df["open_time"] = ot.where(ot < 10**13, ot // 1000)   # some spot files use microseconds
    return df


def fetch_klines(market, symbol, start: date, end_excl: date):
    base = f"{C.BINANCE_VISION}/spot" if market == "spot" else f"{C.BINANCE_VISION}/futures/um"
    frames, miss = [], 0
    for y, m in month_iter(start, end_excl):
        mstart = date(y, m, 1)
        mend = date(y, m, calendar.monthrange(y, m)[1])
        raw, st = get_zip(f"{base}/monthly/klines/{symbol}/1h/{symbol}-1h-{y}-{m:02d}.zip")
        if raw is not None:               # monthly zip exists (month complete on Vision)
            frames.append(parse_klines(raw)); continue
        d = max(mstart, start)
        while d <= mend and d < end_excl:          # daily fallback (current / not-yet-monthly month)
            raw, st = get_zip(f"{base}/daily/klines/{symbol}/1h/{symbol}-1h-{d.isoformat()}.zip")
            if raw is not None: frames.append(parse_klines(raw))
            else: miss += 1
            d += timedelta(days=1)
    if not frames:
        return None, miss
    df = pd.concat(frames).drop_duplicates("open_time").sort_values("open_time")
    return df, miss


def parse_funding(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))  # calc_time, funding_interval_hours, last_funding_rate
    return pd.DataFrame({
        "fundingTime": pd.to_numeric(df["calc_time"], errors="coerce").astype("int64"),
        "fundingRate": pd.to_numeric(df["last_funding_rate"], errors="coerce"),
        "interval_h": pd.to_numeric(df.get("funding_interval_hours", 8), errors="coerce"),
    })


def api_funding(symbol, t_start_ms, t_end_ms):
    rows = []
    for host in FAPI_HOSTS:
        rows, cur, ok = [], t_start_ms, True
        while cur < t_end_ms:
            r, st = get(f"{host}/fapi/v1/fundingRate",
                        {"symbol": symbol, "startTime": cur, "endTime": t_end_ms, "limit": 1000})
            if r is None:
                ok = False; break
            js = r.json()
            if not isinstance(js, list):
                ok = False; break
            if not js:
                break
            rows += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in js]
            cur = rows[-1][0] + 1
            if len(js) < 1000:
                break
        if ok:
            break
        rows = []
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["fundingTime", "fundingRate"])
    df["interval_h"] = df.fundingTime.diff().median() / 3.6e6 if len(df) > 2 else 8
    return df


def fetch_funding(symbol, start: date, end_excl: date):
    frames, api_ranges = [], []
    for y, m in month_iter(start, end_excl):
        mstart = date(y, m, 1)
        nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
        raw, _ = get_zip(f"{C.BINANCE_VISION}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip")
        if raw is not None:
            frames.append(parse_funding(raw))
        else:
            api_ranges.append((max(mstart, start), min(nxt, end_excl)))
    api_ok = True
    for a, b in api_ranges:
        ms = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
        df = api_funding(symbol, ms(a), ms(b))
        if df is None:
            api_ok = False
        else:
            frames.append(df)
    if not frames:
        return None, False
    df = pd.concat(frames).dropna(subset=["fundingTime"]).drop_duplicates("fundingTime").sort_values("fundingTime")
    return df, api_ok


def covered(path: Path, col, end_excl: date, slack_h):
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        mx = pd.read_csv(path, usecols=[col])[col].max()
    except Exception:
        return False
    end_ms = int(datetime(end_excl.year, end_excl.month, end_excl.day, tzinfo=timezone.utc).timestamp() * 1000)
    return mx >= end_ms - slack_h * 3_600_000


def process(row, ddir: Path, start, end_excl, force):
    out = {"fut": row.fut_symbol, "spot": row.spot_symbol}
    fut, spot = row.fut_symbol, row.spot_symbol
    pf, ps, pu = ddir / "funding" / f"{fut}.csv", ddir / "spot_klines" / f"{spot}.csv", ddir / "fut_klines" / f"{fut}.csv"
    for p in (pf, ps, pu):
        p.parent.mkdir(parents=True, exist_ok=True)
    if force or not covered(pu, "open_time", end_excl, 2):
        df, miss = fetch_klines("um", fut, start, end_excl)
        if df is not None: df.to_csv(pu, index=False)
        out["fut_rows"] = 0 if df is None else len(df)
    if force or not covered(ps, "open_time", end_excl, 2):
        df, miss = fetch_klines("spot", spot, start, end_excl)
        if df is not None: df.to_csv(ps, index=False)
        out["spot_rows"] = 0 if df is None else len(df)
    if force or not covered(pf, "fundingTime", end_excl, 12):
        df, api_ok = fetch_funding(fut, start, end_excl)
        if df is not None: df.to_csv(pf, index=False)
        out["funding_rows"] = 0 if df is None else len(df)
        out["funding_api_ok"] = api_ok
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--symbols", default=None, help="comma list of fut symbols to download (default: all in symbols_signal.csv)")
    ap.add_argument("--limit", type=int, default=None, help="only first N symbols (smoke test)")
    ap.add_argument("--start", default=C.DATA_START)
    ap.add_argument("--end", default=C.DATA_END, help="exclusive")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="re-download even if files look complete")
    a = ap.parse_args()
    ddir = C.data_dir(a.data_dir)
    sy = C.load_symbols(a.symbols_file)
    if a.symbols:
        sy = sy[sy.fut_symbol.isin(a.symbols.split(","))]
    if a.limit:
        sy = sy.head(a.limit)
    start, end_excl = date.fromisoformat(a.start), date.fromisoformat(a.end)
    print(f"downloading {len(sy)} symbols {start} .. {end_excl} -> {ddir}", flush=True)
    res = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, r, ddir, start, end_excl, a.force): r.fut_symbol for r in sy.itertuples()}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                o = f.result(); res.append(o)
                print(f"[{i}/{len(sy)}] {futs[f]} {o}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(sy)}] {futs[f]} ERROR {e}", flush=True)
                res.append({"fut": futs[f], "error": str(e)})
    pd.DataFrame(res).to_csv(ddir / "download_status.csv", index=False)
    bad = [r for r in res if "error" in r or r.get("funding_api_ok") is False]
    if bad:
        print(f"WARNING: {len(bad)} symbols had errors or missing funding for the latest month "
              f"(public funding API unreachable?). See {ddir/'download_status.csv'}. "
              f"The build step will warn about incomplete funding.", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
