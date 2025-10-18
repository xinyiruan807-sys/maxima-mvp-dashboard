#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Yahoo Finance fetcher
- Reads tickers from --tickers, env TICKERS, or --universe tickers.txt (one per line)
- Fetches 1d OHLCV and writes tidy CSV: date,ticker,open,high,low,close,adj_close,volume
- Supports --incremental append with de-dup (date+ticker)
Usage:
  python scripts/fetch_market.py --universe maxima-mvp-risk-demo/tickers.txt \
    --out maxima-mvp-risk-demo/data/market_latest.csv --start 2018-01-01 --incremental
"""

import os, time, argparse, logging
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import yfinance as yf

# ---------------- Logging ----------------
def setup_logger():
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------- Args ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/market_latest.csv")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p.add_argument("--universe", default="tickers.txt", help="Path to tickers list file")
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep", type=float, default=1.2)
    return p.parse_args()

# ---------------- Ticker loading ----------------
def load_tickers(explicit: Optional[str], file_path: str) -> List[str]:
    raw = explicit or os.getenv("TICKERS")
    if raw:
        parts = [t.strip().upper() for t in raw.replace("\n", ",").replace(" ", ",").split(",")]
        return [t for t in parts if t]

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            txt = f.read()
        parts = [t.strip().upper() for t in txt.replace("\n", ",").replace(" ", ",").split(",")]
        ticks = [t for t in parts if t]
        if ticks:
            return ticks

    # sensible defaults
    return ["AAPL", "MSFT", "NVDA", "META", "AMZN", "SPY", "QQQ", "TLT", "IEF", "GLD", "BTC-USD", "^GSPC"]

# ---------------- Yahoo fetch helpers ----------------
def _download_batch(tickers: List[str], start: str, end: Optional[str], max_retries: int) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers=tickers,
                start=start, end=end,
                interval="1d",
                auto_adjust=False,
                threads=True,
                group_by="ticker",
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                records = []
                # Some symbols might be missing; guard with presence check
                present = set(df.columns.levels[0])
                for t in tickers:
                    if t not in present:
                        continue
                    sub = df[t].copy()
                    sub.columns = [c.lower() for c in sub.columns]
                    sub["ticker"] = t
                    sub.index.name = "date"
                    records.append(sub.reset_index())
                out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
            else:
                # single-ticker frame
                sub = df.copy()
                sub.columns = [c.lower() for c in sub.columns]
                sub["ticker"] = tickers[0]
                sub.index.name = "date"
                out = sub.reset_index()

            colmap = {"adj close": "adj_close", "adj_close": "adj_close"}
            out = out.rename(columns=colmap)
            out = out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
            out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date
            return out
        except Exception as e:
            last_exc = e
            logging.warning(f"Batch failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(2.5 * attempt)

    if last_exc:
        raise last_exc
    return pd.DataFrame()

def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def fetch_all(tickers: List[str], start: str, end: Optional[str], max_retries: int, sleep_sec: float) -> pd.DataFrame:
    frames = []
    for batch in chunked(tickers, 10):
        logging.info(f"Fetching {len(batch)} tickers: {batch}")
        df = _download_batch(batch, start, end, max_retries)
        if not df.empty:
            frames.append(df)
        time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    out = out.loc[~out[price_cols].isna().all(axis=1)].copy()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)

def read_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
            return df
        except Exception as e:
            logging.warning(f"Failed to read existing CSV ({path}): {e}")
    return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

def dedup_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    base = base.drop_duplicates(subset=["date", "ticker"], keep="last")
    return base.sort_values(["ticker", "date"]).reset_index(drop=True)

# ---------------- Main ----------------
def main():
    setup_logger()
    args = parse_args()

    tickers = load_tickers(args.tickers, args.universe)
    if not tickers:
        logging.error("No tickers provided")
        raise SystemExit(2)

    end = args.end or datetime.now(timezone.utc).date().isoformat()
    logging.info(f"Universe size={len(tickers)} | Start={args.start} End={end} | Out={args.out}")

    new = fetch_all(tickers, args.start, end, args.max_retries, args.sleep)
    all_df = dedup_concat(read_existing(args.out), new) if args.incremental else new

    ordered = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    for c in ordered:
        if c not in all_df.columns:
            all_df[c] = pd.NA
    all_df = all_df[ordered]

    # ---- Safety: fail the job instead of committing an empty CSV ----
    if all_df.empty:
        logging.error("No data fetched. Failing the job to avoid committing an empty CSV.")
        raise SystemExit(1)

    # Write
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_df.to_csv(args.out, index=False)
    logging.info(f"Wrote {len(all_df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
