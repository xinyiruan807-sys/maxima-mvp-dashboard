#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust market fetcher for GitHub Actions
- Primary: yahooquery（更抗限流/反爬）
- Fallback: yfinance（带 UA + HTTP 重试）
- Output: tidy CSV -> date,ticker,open,high,low,close,adj_close,volume
- Incremental de-dup (date+ticker)

Usage:
  python scripts/fetch_market.py --universe maxima-mvp-risk-demo/tickers.txt \
    --out maxima-mvp-risk-demo/data/market_latest.csv --start 2022-01-01 --incremental
"""

import os, time, argparse, logging
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import yahooquery as yq
import yfinance as yf

# ---------------- Logging ----------------
def setup_logger():
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------- HTTP Session for yfinance fallback ----------------
def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
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
    p.add_argument("--start", default="2022-01-01")   # 验证通路更稳；成功后可改更早
    p.add_argument("--end", default=None)
    p.add_argument("--tickers", default=None)
    p.add_argument("--universe", default="tickers.txt")
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--sleep", type=float, default=1.5)
    return p.parse_args()

# ---------------- Tickers ----------------
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
        if ticks: return ticks
    return ["AAPL","MSFT","NVDA","META","AMZN","SPY","QQQ","TLT","IEF","GLD","BTC-USD","^GSPC"]

# ---------------- Normalization ----------------
def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "open":"open","Open":"open",
        "high":"high","High":"high",
        "low":"low","Low":"low",
        "close":"close","Close":"close",
        "adjclose":"adj_close","Adj Close":"adj_close","AdjClose":"adj_close",
        "volume":"volume","Volume":"volume",
        "date":"date","Date":"date"
    })
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index":"date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.date
    df["ticker"] = ticker
    for c in ["open","high","low","close","adj_close","volume"]:
        if c not in df.columns:
            df[c] = pd.NA
    out = df[["date","ticker","open","high","low","close","adj_close","volume"]].dropna(subset=["date"])
    return out

# ---------------- Fetchers ----------------
def fetch_with_yahooquery(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """首选：yahooquery（默认重试，CI 环境更稳）"""
    tq = yq.Ticker(tickers)  # 重要：不传 backoff_factor / max_retries
    # NOTE: yahooquery.history 没有 adj_close 参数
    hist = tq.history(start=start, end=end, interval="1d")
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        frames = []
        if isinstance(hist.index, pd.MultiIndex):
            # index: (symbol, date)
            hist = hist.reset_index()
            for t in tickers:
                df_t = hist[hist["symbol"].str.upper() == t.upper()][
                    ["date","open","high","low","close","adjclose","volume"]
                ].copy()
                frames.append(_normalize(df_t, t))
        else:
            frames.append(_normalize(hist, tickers[0]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame()

def fetch_with_yfinance_fallback(tickers: List[str], start: str, end: Optional[str],
                                 session: requests.Session, max_retries: int, sleep_sec: float) -> pd.DataFrame:
    """兜底：单票 yfinance + UA + HTTP 重试"""
    frames = []
    for t in tickers:
        ok = False
        for attempt in range(1, max_retries+1):
            try:
                tk = yf.Ticker(t, session=session)
                df = tk.history(start=start, end=end, auto_adjust=False, interval="1d")
                df_norm = _normalize(df, t)
                if not df_norm.empty:
                    frames.append(df_norm); ok = True; break
            except Exception as e:
                logging.warning(f"yfinance fallback {t} attempt {attempt} failed: {e}")
                time.sleep(0.7 * attempt)
        if not ok:
            logging.error(f"Fallback failed for {t}")
        time.sleep(sleep_sec)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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
        logging.error("No tickers provided"); raise SystemExit(2)

    end = args.end or datetime.now(timezone.utc).date().isoformat()
    logging.info(f"Universe size={len(tickers)} | Start={args.start} End={end} | Out={args.out}")

    # 1) 首选 yahooquery
    df = fetch_with_yahooquery(tickers, args.start, end)
    if df.empty:
        logging.warning("yahooquery returned empty. Trying yfinance fallback...")
        # 2) yfinance 兜底（带 UA + 重试）
        session = build_session()
        df = fetch_with_yfinance_fallback(tickers, args.start, end, session, args.max_retries, args.sleep)

    # 清洗 + 去重
    if df.empty:
        logging.error("No data fetched. Failing the job to avoid committing an empty CSV.")
        raise SystemExit(1)

    price_cols = ["open","high","low","close","adj_close","volume"]
    df = df.loc[~df[price_cols].isna().all(axis=1)].copy()
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    if args.incremental:
        old = read_existing(args.out)
        df = dedup_concat(old, df)

    ordered = ["date","ticker","open","high","low","close","adj_close","volume"]
    for c in ordered:
        if c not in df.columns: df[c] = pd.NA
    df = df[ordered]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    logging.info(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()

