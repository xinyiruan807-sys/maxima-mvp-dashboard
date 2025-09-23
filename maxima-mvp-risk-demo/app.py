# app.py — Maxima Wealth Dashboard (robust path + Excel fallback)

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Maxima Wealth - Investor Helper (Risk Slider)",
                   layout="wide", page_icon="🧭")

# =============================== Data loaders ===============================
@st.cache_data(show_spinner=False)
def load_mock_data(path: str | None = None):
    """
    读取仓库同级目录下的 Investor_MockData.xlsx。
    如果不存在，返回空 DataFrame（后续由 UI 引导上传）。
    """
    base = Path(__file__).parent
    xls_path = base / "Investor_MockData.xlsx" if path is None else Path(path)

    if not xls_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    trades = pd.read_excel(xls_path, sheet_name="Trades", engine="openpyxl")
    mkt = pd.read_excel(xls_path, sheet_name="MarketBackground", engine="openpyxl")

    # 清洗
    for c in ["Open Time", "Close Time"]:
        if c in trades.columns:
            trades[c] = pd.to_datetime(trades[c], errors="coerce")
    for c in ["Size","Open Price","Close Price","Commission","Swap","Profit",
              "RecommendedPosition","RiskScore","InitCapital"]:
        if c in trades.columns:
            trades[c] = pd.to_numeric(trades[c], errors="coerce")
    if "Date" in mkt.columns:
        mkt["Date"] = pd.to_datetime(mkt["Date"], errors="coerce").dt.date
    return trades, mkt

@st.cache_data(show_spinner=False)
def parse_mt5_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    兼容 MT5 History Center 导出（多为 TAB 分隔，列名 <DATE> <TIME> <OPEN> …）。
    输出索引为 datetime，列包含 open/high/low/close/volume（若有）。
    """
    # 先按 Tab；失败再按逗号
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes))
    mapper = {
        "<DATE>":"date", "<TIME>":"time", "<OPEN>":"open", "<HIGH>":"high",
        "<LOW>":"low", "<CLOSE>":"close", "<TICKVOL>":"tickvol",
        "<VOL>":"volume", "<SPREAD>":"spread"
    }
    df = df.rename(columns=mapper)
    if "date" in df and "time" in df:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                            errors="coerce")
    elif "time" in df:
        dt = pd.to_datetime(df["time"], errors="coerce")
    else:
        raise ValueError("Cannot find date/time columns in uploaded file.")
    df = df.assign(datetime=dt).dropna(subset=["datetime"]).set_index("datetime").sort_index()
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[keep].astype(float)

# =============================== Helpers ===============================
def compute_daily_from_trades(trades: pd.DataFrame):
    t = trades.copy()
    t["date"] = t["Close Time"].dt.date
    sym = t.groupby(["date","Symbol"], as_index=False)["Profit"] \
           .sum().rename(columns={"Profit":"SymPNL"})
    strat = t.groupby(["date","Strategy"], as_index=False)["Profit"] \
             .sum().rename(columns={"Profit":"StratPNL"})
    total = t.groupby("date", as_index=False)["Profit"] \
             .sum().rename(columns={"Profit":"DailyPNL"})
    return total, sym, strat

def risk_to_allocation(risk: int, symbols: list[str]):
    conservative = {"XAUUSD": 0.35, "US30": 0.30, "AAPL": 0.20, "EURUSD": 0.10, "BTCUSD": 0.05}
    balanced     = {"XAUUSD": 0.25, "US30": 0.25, "AAPL": 0.25, "EURUSD": 0.15, "BTCUSD": 0.10}
    aggressive   = {"XAUUSD": 0.10, "US30": 0.15, "AAPL": 0.30, "EURUSD": 0.10, "BTCUSD": 0.35}

    def blend(a, b, w):
        all_syms = set(a) | set(b) | set(symbols)
        out = {s: (a.get(s,0)*(1-w) + b.get(s,0)*w) for s in all_syms}
        ssum = sum(out.values()) or 1.0
        return {k: v/ssum for k, v in out.items()}

    if risk <= 3:
        alloc = blend(conservative, balanced, (risk-1)/2.0)
    elif risk <= 7:
        mid = blend(conservative, balanced, 1.0)
        alloc = blend(mid, aggressive, (risk-4)/3.0)
    else:
        alloc = blend(balanced, aggressive, 0.5 + 0.5*(risk-8)/2.0)

    alloc = {s: alloc.get(s,0.0) for s in symbols}
    ssum = sum(alloc.values()) or 1.0
    return {k: v/ssum for k,v in alloc.items()}

def portfolio_curve_from_alloc(sym_daily: pd.DataFrame, alloc: dict):
    if sym_daily.empty:
        return pd.DataFrame()
    pivot = sym_daily.pivot(index="date", columns="Symbol", values="SymPNL").fillna(0.0)
    cols = pivot.columns.tolist()
    weights = np.array([alloc.get(c, 0.0) for c in cols])
    if weights.sum() == 0:
        return pd.DataFrame({"date": pivot.index, "DailyPNL": 0.0, "CumPNL": 0.0})
    weights = weights / weights.sum()
    daily = pivot.values @ weights
    out = pd.DataFrame({"date": pivot.index, "DailyPNL": daily})
    out["CumPNL"] = out["DailyPNL"].cumsum()
    return out

def compute_metrics_from_returns(r: pd.Series, freq_per_year=252):
    r = r.dropna()
    total_return = (1 + r).prod() - 1
    vol = r.std(ddof=1) * np.sqrt(freq_per_year) if len(r) > 1 else np.nan
    sharpe = (r.mean() * freq_per_year) / vol if vol and vol > 0 else np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    max_dd = (cum / peak - 1).min() if len(cum) else np.nan
    return dict(total_return=total_return, vol=vol, sharpe=sharpe, max_drawdown=max_dd)

def pick_optima(candidates: dict):
    items = list(candidates.items())
    # A: Highest Sharpe（平手用收益打破）
    a = max(items, key=lambda x: ((x[1]["sharpe"] if pd.notna(x[1]["sharpe"]) else -1e9),
                                  x[1]["total_return"]))
    # B: 最大收益
    b = max(items, key=lambda x: x[1]["total_return"])
    # C: 最小风险且收益>0（若都≤0，则找全局最小波动）
    profitable = [it for it in items if it[1]["total_return"] > 0]
    base = profitable if profitable else items
    c = min(base, key=lambda x: (x[1]["vol"] if pd.notna(x[1]["vol"]) else 1e9))
    return {"highest_reward_risk": a, "max_profit": b, "min_risk_pos_profit": c}

def apply_time_range(df_index, preset, df):
    if df is None or len(df_index) == 0:
        return df
    now = df_index.max()
    if preset == "Daily":
        start = now - pd.Timedelta(days=1)
        return df.loc[df_index >= start]
    elif preset == "Weekly":
        start = now - pd.Timedelta(weeks=1)
        return df.loc[df_index >= start]
    elif preset == "Monthly":
        start = now - pd.DateOffset(months=1)
        return df.loc[df_index >= start]
    elif preset == "Yearly":
        start = now - pd.DateOffset(years=1)
        return df.loc[df_index >= start]
    elif preset == "Maximum":
        return df
    else:
        start_date, end_date = st.sidebar.date_input(
            "Custom range",
            value=(df_index.min().date(), df_index.max().date())
        )
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        return df.loc[(df_index >= start) & (df_index < end)]

def risk_alerts(mkt_row):
    msgs = []
    if mkt_row is None or len(mkt_row) == 0:
        return ["No market data."]
    vix = float(mkt_row["VIX"])
    rate = float(mkt_row["FedFundsRate"])
    inf = float(mkt_row["InflationYoY"])
    if vix >= 25: msgs.append("High volatility (VIX ≥ 25): consider reducing leverage and tightening stops.")
    elif vix >= 18: msgs.append("Elevated volatility: watch position sizing and avoid over-trading.")
    else: msgs.append("Calm market regime: conditions favorable for trend-following.")
    if rate > 5.5: msgs.append("Rising rate environment may pressure growth stocks; favor quality & cash-flow assets.")
    if inf > 3.5: msgs.append("Sticky inflation: commodities/defensives may outperform; keep duration risk controlled.")
    return msgs

# =============================== Load data ===============================
trades, mkt = load_mock_data()

# =============================== Sidebar ===============================
with st.sidebar:
    st.header("Investor Controls")

    # 数据源：Mock Trades 或 MT5 OHLCV
    data_source = st.radio("Data source", ["Mock demo (trades)", "Upload MT5 OHLCV"], index=0)

    # 如果 mock 数据为空，给 Excel 上传入口
    if trades.empty or mkt.empty:
        xls_up = st.file_uploader("Upload Investor_MockData.xlsx (optional)", type=["xlsx"])
        if xls_up is not None:
            # 直接从上传读取两张表
            trades = pd.read_excel(xls_up, sheet_name="Trades", engine="openpyxl")
            mkt = pd.read_excel(xls_up, sheet_name="MarketBackground", engine="openpyxl")
            for c in ["Open Time", "Close Time"]:
                if c in trades.columns:
                    trades[c] = pd.to_datetime(trades[c], errors="coerce")
            if "Date" in mkt.columns:
                mkt["Date"] = pd.to_datetime(mkt["Date"], errors="coerce").dt.date

    # 市场筛选（Mock 用）
    syms_all = sorted(trades["Symbol"].dropna().unique().tolist()) if not trades.empty else []
    sel_syms = st.multiselect("Symbols (mock demo)", syms_all, default=syms_all)

    risk = st.slider("Risk tolerance (1 = conservative, 10 = aggressive)", 1, 10, 5)

    accounts = sorted(trades["Account"].dropna().unique().tolist()) if not trades.empty else []
    strats = sorted(trades["Strategy"].dropna().unique().tolist()) if not trades.empty else []
    sel_accounts = st.multiselect("Accounts", accounts, default=accounts)
    sel_strats = st.multiselect("Strategies", strats, default=strats)

    st.divider()
    st.subheader("Time Range")
    preset = st.radio("Preset",
                      ["Daily","Weekly","Monthly","Yearly","Maximum","Custom"],
                      index=4, horizontal=True)

    uploaded = st.file_uploader(
        "Upload MT5 export (Tools → History Center → Export)",
        type=["csv","txt","tsv"],
        help="Supports TAB-separated or CSV."
    )

# =============================== Header ===============================
st.markdown("<h2 style='margin-bottom:0'>Investor Helper – Optimal Strategy Guide</h2>",
            unsafe_allow_html=True)
st.caption("Adjust the risk slider, choose market/time range, and view optimal choices.")

# =============================== Main logic ===============================
if data_source.startswith("Upload") and uploaded is not None:
    # ===== Path B: MT5 OHLCV =====
    try:
        oh = parse_mt5_csv(uploaded.read())  # index=datetime, has close
        oh_f = apply_time_range(oh.index, preset, oh)
        if oh_f is None or oh_f.empty:
            st.info("No data in selected range.")
        else:
            r = oh_f["close"].pct_change()
            metrics = {"Buy & Hold": compute_metrics_from_returns(r)}
            opt = pick_optima(metrics)

            col1, col2, col3 = st.columns(3)
            with col1:
                name, m = opt["highest_reward_risk"]
                st.subheader("Highest Reward/Risk")
                st.metric("Sharpe", f"{m['sharpe']:.2f}" if pd.notna(m['sharpe']) else "N/A")
                st.caption(name)
            with col2:
                name, m = opt["max_profit"]
                st.subheader("Maximum Profit")
                st.metric("Total Return", f"{m['total_return']*100:.1f}%")
                st.caption(name)
            with col3:
                name, m = opt["min_risk_pos_profit"]
                st.subheader("Minimum Risk (Profit > 0)")
                st.metric("Annualized σ", f"{m['vol']:.2%}" if pd.notna(m['vol']) else "N/A")
                st.caption(name)

            st.divider()
            st.subheader("Price (filtered)")
            st.line_chart(oh_f["close"])
            st.caption("Metrics computed from uploaded MT5 OHLCV.")
    except Exception as e:
        st.error(f"Failed to parse MT5 file: {e}")

else:
    # ===== Path A: Mock trades =====
    if trades.empty or mkt.empty:
        st.warning("Demo Excel not found. Upload Investor_MockData.xlsx on the left, "
                   "or switch to 'Upload MT5 OHLCV' mode.")
    else:
        f = trades.copy()
        if sel_accounts: f = f[f["Account"].isin(sel_accounts)]
        if sel_strats:   f = f[f["Strategy"].isin(sel_strats)]
        if sel_syms:     f = f[f["Symbol"].isin(sel_syms)]

        total, sym_daily, strat_daily = compute_daily_from_trades(f)

        # 应用时间范围（把 date 转为 datetime 索引）
        if not sym_daily.empty:
            sym_dt = sym_daily.copy()
            sym_dt["date_dt"] = pd.to_datetime(sym_dt["date"])
            sym_dt = sym_dt.set_index("date_dt")
            sym_dt = apply_time_range(sym_dt.index, preset, sym_dt)
            sym_daily = sym_dt.reset_index(drop=True)
        if not total.empty:
            tot_dt = total.copy()
            tot_dt["date_dt"] = pd.to_datetime(tot_dt["date"])
            tot_dt = tot_dt.set_index("date_dt")
            tot_dt = apply_time_range(tot_dt.index, preset, tot_dt)
            total = tot_dt.reset_index(drop=True)
        if not strat_daily.empty:
            strat_dt = strat_daily.copy()
            strat_dt["date_dt"] = pd.to_datetime(strat_dt["date"])
            strat_dt = strat_dt.set_index("date_dt")
            strat_dt = apply_time_range(strat_dt.index, preset, strat_dt)
            strat_daily = strat_dt.reset_index(drop=True)

        syms_for_alloc = sorted(f["Symbol"].dropna().unique().tolist())
        alloc = risk_to_allocation(risk, syms_for_alloc)
        port = portfolio_curve_from_alloc(sym_daily, alloc) if not sym_daily.empty else pd.DataFrame()

        # 市场风险提示
        latest_date = mkt["Date"].max() if not mkt.empty else None
        latest_row = mkt[mkt["Date"] == latest_date].iloc[0] if latest_date is not None else None
        alerts = risk_alerts(latest_row) if latest_row is not None else ["No market data."]

        c1, c2 = st.columns([2,1])
        with c1:
            alloc_df = pd.DataFrame({"Symbol": list(alloc.keys()), "Weight": list(alloc.values())}) \
                        .sort_values("Weight", ascending=False)
            fig_alloc = px.bar(alloc_df, x="Symbol", y="Weight",
                               title="Optimal Allocation (by Risk)", text_auto=".0%")
            fig_alloc.update_yaxes(tickformat=".0%")
            fig_alloc.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=360)
            st.plotly_chart(fig_alloc, use_container_width=True)
        with c2:
            st.markdown("#### Risk Alerts (latest)")
            if latest_row is not None:
                st.write(f"VIX: {latest_row['VIX']:.1f} | "
                         f"FedFundsRate: {latest_row['FedFundsRate']:.2f}% | "
                         f"Inflation: {latest_row['InflationYoY']:.2f}%")
            for m in alerts:
                st.info(m)

        tab1, tab2, tab3 = st.tabs(["Portfolio Backtest", "Strategy Insights", "Trades"])

        with tab1:
            if not port.empty:
                fig = px.line(port, x="date", y=["DailyPNL","CumPNL"],
                              title="Backtest: Daily & Cumulative PnL (Optimal Mix)")
                fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=400, legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

                # 用 DailyPNL 近似收益（演示）
                r_demo = port["DailyPNL"].replace(0, np.nan) / (port["DailyPNL"].abs().mean() + 1e-9)
                metrics = {"Buy & Hold (demo mix)": compute_metrics_from_returns(r_demo)}
                opt = pick_optima(metrics)

                k1,k2,k3 = st.columns(3)
                k1.metric("Total PnL (demo)", f"${float(port['DailyPNL'].sum()):,.0f}")
                mdd = float((port["CumPNL"].cummax() - port["CumPNL"]).max()) if not port.empty else 0.0
                k2.metric("Max Drawdown", f"${mdd:,.0f}")
                k3.metric("Sharpe (demo)", f"{opt['highest_reward_risk'][1]['sharpe']:.2f}")
            else:
                st.info("Not enough data for the current filters.")

        with tab2:
            if not strat_daily.empty:
                recent_dates = sorted(strat_daily["date"].unique())
                keep_n = min(20, len(recent_dates))
                last_dates = set(recent_dates[-keep_n:])
                recent = strat_daily[strat_daily["date"].isin(last_dates)]
                g = recent.groupby("Strategy", as_index=False)["StratPNL"].sum() \
                          .sort_values("StratPNL", ascending=False)
                fig = px.bar(g, x="Strategy", y="StratPNL",
                             title="Recent Strategy Contribution (last ~20 days)", text_auto=".2s")
                fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=360)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Tip: Prefer strategies with consistent positive contribution under your chosen risk.")
            else:
                st.info("No strategy data available.")

        with tab3:
            st.dataframe(f.sort_values("Open Time", ascending=False),
                         use_container_width=True, height=420)
            st.download_button("Download filtered trades (CSV)",
                               data=f.to_csv(index=False).encode("utf-8"),
                               file_name="filtered_trades.csv", mime="text/csv")

st.caption("Mock demo for UX exploration. Not investment advice.")

