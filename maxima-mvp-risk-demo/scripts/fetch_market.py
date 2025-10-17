#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Yahoo Finance fetcher
- Reads tickers from env TICKERS or file tickers.txt (comma/space/line separated)
- Fetches OHLCV (1d) and writes a tidy CSV (date,ticker,open,high,low,close,adj_close,volume)
- Supports incremental append with de-duplication
Usage:
  python scripts/fetch_market.py --out data/market_latest.csv --start 2018-01-01 --incremental
"""
import os, time, argparse, logging
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import yfinance as yf

def setup_logger():
    logging.basicConfig(level=os.getenv("LOGLEVEL","INFO"),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/market_latest.csv")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--tickers", default=None)
    p.add_argument("--universe", default="tickers.txt")
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep", type=float, default=1.2)
    return p.parse_args()

def load_tickers(explicit: Optional[str], file_path: str) -> List[str]:
    raw = explicit or os.getenv("TICKERS")
    if raw:
        parts = [t.strip().upper() for t in raw.replace("\n",",").replace(" ",",").split(",")]
        return [t for t in parts if t]
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            txt = f.read()
        parts = [t.strip().upper() for t in txt.replace("\n",",").replace(" ",",").split(",")]
        tickers = [t for t in parts if t]
        if tickers: return tickers
    return ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","SPY","QQQ","TLT","IEF","GLD"]

def _download_batch(tickers: List[str], start: str, end: Optional[str], max_retries: int) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, max_retries+1):
        try:
            df = yf.download(tickers=tickers, start=start, end=end, interval="1d",
                             auto_adjust=False, threads=True, group_by="ticker", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                records = []
                for t in tickers:
                    if t not in df.columns.levels[0]: continue
                    sub = df[t].copy()
                    sub.columns = [c.lower() for c in sub.columns]
                    sub["ticker"] = t
                    sub.index.name = "date"
                    sub = sub.reset_index()
                    records.append(sub)
                out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
            else:
                sub = df.copy()
                sub.columns = [c.lower() for c in sub.columns]
                sub["ticker"] = tickers[0]
                sub.index.name = "date"
                out = sub.reset_index()
            colmap = {"adj close":"adj_close","adj_close":"adj_close"}
            out = out.rename(columns=colmap)
            out = out[["date","ticker","open","high","low","close","adj_close","volume"]]
            out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date
            return out
        except Exception as e:
            last_exc = e
            logging.warning(f"Batch failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(2.5 * attempt)
    if last_exc: raise last_exc
    return pd.DataFrame()

def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n): yield lst[i:i+n]

def fetch_all(tickers: List[str], start: str, end: Optional[str], max_retries: int, sleep_sec: float) -> pd.DataFrame:
    frames = []
    for batch in chunked(tickers, 10):
        logging.info(f"Fetching {len(batch)} tickers: {batch}")
        df = _download_batch(batch, start, end, max_retries)
        if not df.empty: frames.append(df)
        time.sleep(sleep_sec)
    if not frames:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])
    out = pd.concat(frames, ignore_index=True)
    price_cols = ["open","high","low","close","adj_close","volume"]
    out = out.loc[~out[price_cols].isna().all(axis=1)].copy()
    return out.sort_values(["ticker","date"]).reset_index(drop=True)

def read_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
            return df
        except Exception as e:
            logging.warning(f"Read existing failed: {e}")
    return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])

def dedup_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    base = base.drop_duplicates(subset=["date","ticker"], keep="last")
    return base.sort_values(["ticker","date"]).reset_index(drop=True)

def main():
    setup_logger()
    args = parse_args()
    tickers = load_tickers(args.tickers, args.universe)
    if not tickers:
        logging.error("No tickers provided"); return
    end = args.end or datetime.now(timezone.utc).date().isoformat()
    logging.info(f"Universe={len(tickers)} Start={args.start} End={end} Out={args.out}")
    new = fetch_all(tickers, args.start, end, args.max_retries, args.sleep)
    all_df = dedup_concat(read_existing(args.out), new) if args.incremental else new
    ordered = ["date","ticker","open","high","low","close","adj_close","volume"]
    for c in ordered:
        if c not in all_df.columns: all_df[c] = pd.NA
    all_df = all_df[ordered]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_df.to_csv(args.out, index=False)
    logging.info(f"Wrote {len(all_df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
