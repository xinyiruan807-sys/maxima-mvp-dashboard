#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Yahoo Finance fetcher for GitHub Actions
- Uses a requests.Session with User-Agent and HTTP retries (fixes empty/HTML responses)
- Batch download via yfinance.download with session, plus per-ticker fallback via Ticker().history
- Writes tidy CSV: date,ticker,open,high,low,close,adj_close,volume
- Supports --incremental append with de-dup (date+ticker)
"""

import os, time, argparse, logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- Logging ----------------
def setup_logger():
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------- HTTP Session ----------------
def build_session() -> requests.Session:
    s = requests.Session()
    # UA 很重要，防止返回空/HTML
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    retry = Retry(
        total=5, backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

# ---------------- Args ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/market_latest.csv")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--tickers", default=None, help="Comma-separated tickers")
    p.add_argument("--universe", default="tickers.txt", help="Path to tickers list file")
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--sleep", type=float, default=2.0)
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

# ---------------- Fetch helpers ----------------
def _normalize(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return df
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "adj_close"})
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    # yfinance.history 返回列名首字母大写
    rename_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
    }
    df = df.rename(columns=rename_map)
    if "adj_close" not in df.columns:
        df["adj_close"] = df.get("close")
    # index 可能是 DatetimeIndex
    if df.index.name in (None, "", "Date", "date"):
        df = df.reset_index()
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
    if ticker is not None and "ticker" not in df.columns:
        df["ticker"] = ticker
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]

def download_batch(tickers: List[str], start: str, end: Optional[str], session) -> pd.DataFrame:
    """
    批量下载；如果 Yahoo 返回空/HTML，本函数会抛异常，由上层处理。
    """
    df = yf.download(
        tickers=tickers,
        start=start, end=end,
        interval="1d",
        auto_adjust=False,
        threads=True,
        group_by="ticker",
        progress=False,
        session=session,
    )

    if isinstance(df.columns, pd.MultiIndex):
        recs = []
        present = set(df.columns.levels[0])
        for t in tickers:
            if t not in present:
                continue
            recs.append(_normalize(df[t].copy(), ticker=t))
        if recs:
            return pd.concat(recs, ignore_index=True)
        return pd.DataFrame()
    else:
        # 单票
        return _normalize(df.copy(), ticker=tickers[0])

def download_single(ticker: str, start: str, end: Optional[str], session) -> pd.DataFrame:
    """
    单票兜底：Ticker().history
    """
    tk = yf.Ticker(ticker, session=session)
    hist = tk.history(start=start, end=end, auto_adjust=False, interval="1d")
    return _normalize(hist, ticker=ticker)

def fetch_all(tickers: List[str], start: str, end: Optional[str], sleep_sec: float, max_retries: int, session) -> pd.DataFrame:
    frames = []
    # 先尝试批量（效率高）
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Fetching {len(tickers)} tickers in batch (attempt {attempt}/{max_retries})")
            df = download_batch(tickers, start, end, session)
            if not df.empty:
                frames.append(df)
            break
        except Exception as e:
            logging.warning(f"Batch attempt {attempt} failed: {e}")
            time.sleep(attempt * 1.0)

    # 如果批量不成功或为空，就逐票兜底
    got = set(frames[0]["ticker"].unique()) if frames else set()
    remaining = [t for t in tickers if t not in got]
    for t in remaining:
        for attempt in range(1, max_retries + 1):
            try:
                df1 = download_single(t, start, end, session)
                if not df1.empty:
                    frames.append(df1)
                    logging.info(f"Fetched {t} with fallback, rows={len(df1)}")
                    break
            except Exception as e:
                logging.warning(f"Single {t} attempt {attempt} failed: {e}")
                time.sleep(0.8 * attempt)
        time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])

    out = pd.concat(frames, ignore_index=True)
    # 丢弃全空行
    price_cols = ["open","high","low","close","adj_close","volume"]
    out = out.loc[~out[price_cols].isna().all(axis=1)].copy()
    return out.sort_values(["ticker","date"]).reset_index(drop=True)

# ---------------- Incremental helpers ----------------
def read_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
            return df
        except Exception as e:
            logging.warning(f"Failed to read existing CSV ({path}): {e}")
    return pd.DataFrame(columns=["date","ticker","open","high","low","close","adj_close","volume"])

def dedup_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    base = base.drop_duplicates(subset=["date","ticker"], keep="last")
    return base.sort_values(["ticker","date"]).reset_index(drop=True)

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

    session = build_session()
    new = fetch_all(tickers, args.start, end, sleep_sec=args.sleep, max_retries=args.max_retries, session=session)
    all_df = dedup_concat(read_existing(args.out), new) if args.incremental else new

    ordered = ["date","ticker","open","high","low","close","adj_close","volume"]
    for c in ordered:
        if c not in all_df.columns:
            all_df[c] = pd.NA
    all_df = all_df[ordered]

    # 空即失败，避免提交空 CSV
    if all_df.empty:
        logging.error("No data fetched. Failing the job to avoid committing an empty CSV.")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_df.to_csv(args.out, index=False)
    logging.info(f"Wrote {len(all_df):,} rows to {args.out}")

if __name__ == "__main__":
    main()

