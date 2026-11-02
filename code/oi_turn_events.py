#!/usr/bin/env python3
"""'Funding turns negative' event study: how did open interest move in the 6 hours BEFORE a coin's funding turned negative?

DEFINITIONS (all on the 267-coin pool, hourly OI dataset, window = the 1y window 2025-10-28 .. 2026-10-28; aggregate numbers only):
  settlement event   an hour t with a funding settlement of the coin (ev[t] != 0); its rate is r8[t] (that event's rate scaled to 8h).
  previous event     the coin's previous settlement event (any time before t, also before the window start).
  TURN-NEGATIVE      (primary, "strict")  rate(event) < 0  AND  rate(previous event) >= 0.   Every such transition is one event (no de-duplication).
  variants           depth threshold: rate(event) < -thr  for thr = 1 / 5 / 10 bp (8h-scaled);  de-duplication: at most one event per coin per UTC day.
  OI change          OI(t) / OI(t-6) - 1, where OI(t) is the hourly OI dataset value at t = last 5-min snapshot of hour t-1 (known before the settlement hour),
                     so the span is the 6 hours that END just before the settlement.  Events whose OI is unknown at either end are counted in 'events'
                     but not in the median / share-up ('events_with_oi').
  baseline           same 6h OI change over ALL coin-hours of the pool in the window (hours with OI known at both ends).
Writes results/oi_turn_events.csv.   python oi_turn_events.py
"""
import numpy as np, pandas as pd
import config as C
import run_backtest as R


def main():
    D = R.Data(C.data_dir(None)); ok = R.load_ok(D)
    a, b = C.WINDOWS["1y"]; h0, h1 = R.hidx(a), R.hidx(b)
    ev, r8, oi = D.arr[6], D.arr[7], D.oih
    rows = []
    def study(name, thr, daily_dedupe):
        n = 0; ch = []
        for j in range(D.N):
            if ok[j] <= 0: continue
            hs = np.where(ev[:, j] != 0)[0]
            if len(hs) < 2: continue
            rate = r8[hs, j]; seen_day = set()
            for i in range(1, len(hs)):
                t = hs[i]
                if not (h0 <= t < h1): continue
                if not (np.isfinite(rate[i]) and np.isfinite(rate[i - 1])): continue
                if not (rate[i] < -thr and rate[i - 1] >= 0): continue
                if daily_dedupe:
                    d = t // 24
                    if d in seen_day: continue
                    seen_day.add(d)
                n += 1
                if t >= 6 and np.isfinite(oi[t, j]) and np.isfinite(oi[t - 6, j]) and oi[t - 6, j] > 0: ch.append(oi[t, j] / oi[t - 6, j] - 1)
        ch = np.array(ch)
        rows.append(dict(definition=name, events=n, events_with_oi=len(ch), median_oi_change_6h_pct=np.median(ch) * 100 if len(ch) else np.nan,
                         share_up_pct=(ch > 0).mean() * 100 if len(ch) else np.nan))
    study("STRICT: event rate < 0 and previous event rate >= 0 (primary)", 0.0, False)
    for bp in (1, 5, 10): study(f"event rate < -{bp}bp and previous event rate >= 0", bp * 1e-4, False)
    study("STRICT, at most one event per coin per UTC day", 0.0, True)
    X = oi[h0:h1]; Y = oi[h0 - 6:h1 - 6]
    with np.errstate(all="ignore"): c = (X / Y - 1)[(np.isfinite(X) & np.isfinite(Y) & (Y > 0)) & (ok > 0)[None, :]]
    rows.append(dict(definition="BASELINE: all coin-hours of the pool (6h OI change)", events=int(c.size), events_with_oi=int(c.size),
                     median_oi_change_6h_pct=np.median(c) * 100, share_up_pct=(c > 0).mean() * 100))
    df = pd.DataFrame(rows).round(2); df.to_csv(C.REPO / "results" / "oi_turn_events.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
