#!/usr/bin/env python3
"""Build the hourly [T, N] arrays used by the engine from the downloaded CSVs -> <data-dir>/dataset.npz (+ dataset_meta.csv).

Per symbol:  spot / perp hourly VWAP, close, quote volume ; funding events per hour ; r8 (latest funding rate scaled to 8h) ;
             r8s (24h average, only used by the disabled positive-funding leg) ; hourly depth (cap02, cap1) ;
             mk (v8: mark price at the start of the hour, from download_markprice.py; funding events are valued with it) ;
             oi (open interest in USDT of the PREVIOUS UTC day, broadcast to the 24 hours of the current day; NaN = unknown).

Funding interval: per event = time since the previous settlement of that symbol (1..8 h), so r8 = rate * 8 / interval_of_that_event.

Depth:
  --depth auto   (default) use depth/{FUT}.csv from download_depth.py when present, otherwise a volume-based proxy
  --depth real   require real depth files
  --depth proxy  always use the proxy:  cap1 = 2.4 x , cap02 = 0.26 x  trailing-24h median of hourly min(spot, perp) quote volume
                 (ratios = cross-sectional medians of real depth / volume over the 2026-05..10 window)
"""
import argparse, sys
import numpy as np, pandas as pd
import config as C

T0 = pd.Timestamp(C.DATA_START, tz="UTC")
T1 = pd.Timestamp(C.DATA_END, tz="UTC")
T0_MS = int(T0.value // 10**6)
H_MS = 3_600_000
T = int((T1 - T0).total_seconds() // 3600)
PROXY_K1, PROXY_K02 = 2.4, 0.26


def load_one(ddir, fut, spot, mult, interval="event"):
    ff, fs, fu = ddir / "funding" / f"{fut}.csv", ddir / "spot_klines" / f"{spot}.csv", ddir / "fut_klines" / f"{fut}.csv"
    if not (ff.exists() and fs.exists() and fu.exists()):
        return None
    fd, sd, ud = pd.read_csv(ff), pd.read_csv(fs), pd.read_csv(fu)
    if len(fd) < 10 or len(sd) < 100 or len(ud) < 100:
        return None
    out = {}
    for name, d in (("s", sd), ("f", ud)):
        d = d.drop_duplicates("open_time").sort_values("open_time")
        idx = ((d.open_time.astype("int64") - T0_MS) // H_MS).to_numpy()
        ok = (idx >= 0) & (idx < T)
        d, idx = d[ok], idx[ok]
        v = np.where(d["volume"] > 0, d["quote_volume"] / d["volume"], d["close"])
        arrv, arrc, arrq = (np.full(T, np.nan) for _ in range(3))
        div = mult if name == "f" else 1.0           # 1000BONKUSDT perp price / 1000 = spot-equivalent price
        arrv[idx] = v / div; arrc[idx] = d["close"].to_numpy() / div; arrq[idx] = d["quote_volume"].to_numpy()
        out[name] = (arrv, arrc, arrq)
    fd = fd.drop_duplicates("fundingTime").sort_values("fundingTime")
    gaps = fd.fundingTime.diff() / 3.6e6
    med = float(gaps.median()) if gaps.notna().any() else 8.0
    if not np.isfinite(med) or med <= 0: med = 8.0
    med = min(max(round(med), 1), 8)                 # median interval, only used for the 24h window of r8s
    gap = gaps.bfill().fillna(med).round().clip(1, 8).to_numpy()   # per-event interval (first event: next gap)
    if interval == "median": gap = np.full(len(gap), float(med))     # legacy: one median interval per symbol (only for the bridge table)
    ft = fd.fundingTime.astype("int64").to_numpy(); rate = fd.fundingRate.astype(float).to_numpy()
    idx = np.clip(np.ceil((ft - T0_MS - 60_000) / H_MS).astype(int), 0, None)
    ok = idx < T
    ev = np.zeros(T); np.add.at(ev, idx[ok], rate[ok])          # funding paid in that hour
    tmp = np.full(T, np.nan); tmp[idx[ok]] = (rate * (8.0 / gap))[ok]
    r8 = pd.Series(tmp).ffill().to_numpy()                       # latest rate scaled to 8h
    w = max(1, int(round(24 / med)))
    ser = pd.Series(rate * (8.0 / gap)).rolling(w, min_periods=w).mean().to_numpy()
    tmp2 = np.full(T, np.nan); tmp2[idx[ok]] = ser[ok]
    r8s = pd.Series(tmp2).ffill().to_numpy()
    last_f = fd.fundingTime.max()
    return out, ev, r8, r8s, last_f


def load_mark(ddir, fut, mult):
    """Hourly [T] mark price at the START of each hour (open of the 1h markPriceKlines bar = mark at hh:00:00), spot-equivalent (perp / mult).
    NaN where the bar is missing (the engine then falls back to the previous perp close and counts it in diag[10])."""
    p = ddir / "mark_klines" / f"{fut}.csv"
    out = np.full(T, np.nan)
    if not p.exists(): return out
    d = pd.read_csv(p).drop_duplicates("open_time").sort_values("open_time")
    idx = ((d.open_time.astype("int64") - T0_MS) // H_MS).to_numpy()
    ok = (idx >= 0) & (idx < T)
    o = d["open"].to_numpy(float) / mult
    out[idx[ok]] = np.where(o[ok] > 0, o[ok], np.nan)
    return out


def lagged_oi(ddir, fut):
    """Hourly [T] OI of the previous UTC day (causal). Missing days carry the last known value forward; before the first value -> NaN."""
    p = ddir / "oi" / f"{fut}.csv"
    out = np.full(T, np.nan)
    if not p.exists(): return out
    d = pd.read_csv(p)
    if d.empty: return out
    s = pd.Series(d.oi_usdt.to_numpy(float), index=pd.to_datetime(d.date, utc=True)).sort_index()
    days = pd.date_range(T0.normalize(), T1, freq="D", inclusive="left", tz="UTC")
    s = s.reindex(days).ffill()
    prev = s.shift(1).to_numpy()                                  # value of day d-1 applies to day d
    return np.repeat(prev, 24)[:T]


def lagged_oi_hourly(ddir, fut):
    """Hourly [T] OI of the PREVIOUS COMPLETED hour (causal): value at index t = last 5-min snapshot of hour t-1.
    Source oi_hourly/{FUT}.csv (download_oi_hourly.py). Missing hours carry the last known value forward (as in the daily version); before the first value -> NaN."""
    p = ddir / "oi_hourly" / f"{fut}.csv"
    out = np.full(T, np.nan)
    if not p.exists(): return out
    d = pd.read_csv(p)
    if d.empty: return out
    h = pd.to_datetime(d.hour, utc=True)
    idx = ((h - T0) // pd.Timedelta(hours=1)).to_numpy().astype(int)
    ok = (idx >= 0) & (idx < T)
    cur = np.full(T, np.nan); cur[idx[ok]] = d.oi_usdt.to_numpy(float)[ok]
    cur = pd.Series(cur).ffill().to_numpy()                      # latest known value per hour (incl. that hour's last snapshot)
    out[1:] = cur[:-1]                                           # limit at hour t uses hour t-1 only
    return out


def real_depth(ddir, fut):
    p = ddir / "depth" / f"{fut}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p)
    if len(d) < 24: return None
    h = pd.to_datetime(d.hour, utc=True)
    idx = ((h - T0) // pd.Timedelta(hours=1)).to_numpy().astype(int)
    ok = (idx >= 0) & (idx < T)
    a1 = np.full(T, np.nan); a02 = np.full(T, np.nan)
    a1[idx[ok]] = d.cap1_p50.to_numpy()[ok]; a02[idx[ok]] = d.cap02_p50.to_numpy()[ok]
    if not np.isfinite(a02).any():
        a02 = a1 * 0.115                       # fallback ratio if only the 1% bucket exists
    out = []
    for a in (a02, a1):
        s = pd.Series(a).ffill(limit=2)
        med = np.nanmedian(a) if np.isfinite(a).any() else np.nan
        out.append(s.fillna(med).to_numpy())
    return out


def proxy_depth(sq, fq):
    mv = pd.Series(np.minimum(sq, fq)).where(lambda s: s > 0)
    base = mv.rolling(24, min_periods=6).median().bfill().ffill().to_numpy()
    base = np.nan_to_num(base)
    return [PROXY_K02 * base, PROXY_K1 * base]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--depth", choices=["auto", "real", "proxy"], default="auto")
    ap.add_argument("--out-name", default="dataset")
    ap.add_argument("--interval", choices=["event", "median"], default="event", help="funding interval used to scale r8: per event (default) or legacy single median")
    a = ap.parse_args()
    ddir = C.data_dir(a.data_dir)
    sym = C.load_symbols(a.symbols_file)
    sp = pd.read_csv(C.REPO / "spread_by_symbol.csv")
    hs_s_known = sp[sp.market == "spot"].set_index("symbol").half_spread_med.to_dict()
    hs_f_known = sp[sp.market == "fut"].set_index("symbol").half_spread_med.to_dict()
    hs_med = float(np.nanmedian(list(hs_s_known.values())))
    ratio_f = np.nanmedian([hs_f_known[s] / hs_s_known[s] for s in hs_s_known if s in hs_f_known and hs_s_known[s] > 0])
    tr = pd.read_csv(C.REPO / "tier_ratios.csv").set_index("symbol")
    r2_med, r5_med = float(tr.r2.median()), float(tr.r5.median())
    keys = ["sv", "sc", "sq", "fv", "fc", "fq", "ev", "r8", "r8s", "d02", "d1", "oi", "oih", "mk"]
    arrs = {k: [] for k in keys}; meta = []; skipped = []; stale = []
    for r in sym.itertuples():
        x = load_one(ddir, r.fut_symbol, r.spot_symbol, r.mult, a.interval)
        if x is None:
            skipped.append(r.fut_symbol); continue
        o, ev, r8, r8s, last_f = x
        if np.isfinite(o["s"][0]).sum() < 200 or np.isfinite(o["f"][0]).sum() < 200:
            skipped.append(r.fut_symbol); continue
        dd = None if a.depth == "proxy" else real_depth(ddir, r.fut_symbol)
        if dd is None:
            if a.depth == "real":
                skipped.append(r.fut_symbol); continue
            dd, dsrc = proxy_depth(o["s"][2], o["f"][2]), "proxy"
        else:
            dsrc = "real"
        if last_f < T1.value // 10**6 - 36 * H_MS:
            stale.append(r.fut_symbol)
        for k, v in zip(["sv", "sc", "sq"], o["s"]): arrs[k].append(v)
        for k, v in zip(["fv", "fc", "fq"], o["f"]): arrs[k].append(v)
        arrs["ev"].append(ev); arrs["r8"].append(r8); arrs["r8s"].append(r8s)
        arrs["d02"].append(dd[0]); arrs["d1"].append(dd[1]); arrs["oi"].append(lagged_oi(ddir, r.fut_symbol)); arrs["oih"].append(lagged_oi_hourly(ddir, r.fut_symbol)); arrs["mk"].append(load_mark(ddir, r.fut_symbol, r.mult))
        hs = hs_s_known.get(r.spot_symbol, hs_med)
        hf = hs_f_known.get(r.fut_symbol, hs * ratio_f)
        meta.append(dict(fut=r.fut_symbol, spot=r.spot_symbol, mult=r.mult, hs_s=hs, hs_f=hf,
                         r2=float(tr.r2.get(r.fut_symbol, r2_med)), r5=float(tr.r5.get(r.fut_symbol, r5_med)),
                         depth_source=dsrc))
    if not meta:
        sys.exit("no symbols could be built - run download_data.py first")
    M = pd.DataFrame(meta)
    out = {k: np.stack(v, axis=1).astype(np.float64 if k in ("sv", "sc", "fv", "fc", "r8", "r8s", "ev", "oi", "oih", "mk") else np.float32)
           for k, v in arrs.items()}
    np.savez_compressed(ddir / f"{a.out_name}.npz", **out)
    M.to_csv(ddir / f"{a.out_name}_meta.csv", index=False)
    print(f"built {len(M)} symbols, T={T} hours ({C.DATA_START}..{C.DATA_END}); depth: {M.depth_source.value_counts().to_dict()}")
    if skipped: print(f"WARNING: skipped {len(skipped)} symbols with missing/short data: {skipped[:10]}{'...' if len(skipped)>10 else ''}")
    if stale: print(f"WARNING: funding history for {len(stale)} symbols ends >36h before {C.DATA_END} "
                    f"(funding API unreachable at download time?). Results for the last days will be incomplete: {stale[:10]}")
    print("->", ddir / f"{a.out_name}.npz")


if __name__ == "__main__":
    main()
