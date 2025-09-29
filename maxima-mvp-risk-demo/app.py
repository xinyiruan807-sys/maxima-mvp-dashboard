# app.py — Maxima Wealth Dashboard (full replacement with enhancements 1–6)
# Requirements: streamlit, pandas, numpy, plotly, openpyxl

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Maxima Wealth - Investor Helper (Optimal Strategies)",
    layout="wide",
    page_icon="🧭"
)

# =============================== Data loaders ===============================
@st.cache_data(show_spinner=False)
def load_mock_data(path: str | None = None):
    base = Path(__file__).parent
    xls_path = base / "Investor_MockData.xlsx" if path is None else Path(path)
    if not xls_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    trades = pd.read_excel(xls_path, sheet_name="Trades", engine="openpyxl")
    mkt = pd.read_excel(xls_path, sheet_name="MarketBackground", engine="openpyxl")

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
    # Try TAB first, then comma
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
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
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
    sym = t.groupby(["date","Symbol"], as_index=False)["Profit"].sum().rename(columns={"Profit":"SymPNL"})
    strat = t.groupby(["date","Strategy"], as_index=False)["Profit"].sum().rename(columns={"Profit":"StratPNL"})
    total = t.groupby("date", as_index=False)["Profit"].sum().rename(columns={"Profit": "DailyPNL"})
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

def annualized_vol(r: pd.Series, freq_per_year=252) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(freq_per_year)) if len(r) > 1 else np.nan

def sharpe_ratio(r: pd.Series, freq_per_year=252) -> float:
    r = r.dropna()
    vol = r.std(ddof=1) * np.sqrt(freq_per_year) if len(r) > 1 else np.nan
    return float((r.mean() * freq_per_year) / vol) if (vol and vol > 0) else np.nan

def pct_returns_from_price(price: pd.Series) -> pd.Series:
    return price.sort_index().pct_change()

def cum_return(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return (1 + r).cumprod() - 1

def drawdown_curve(r: pd.Series) -> pd.Series:
    eq = (1 + r.fillna(0)).cumprod()
    peak = eq.cummax()
    return eq / peak - 1

def rolling_stat(r: pd.Series, window: int, kind: str = "sharpe"):
    r = r.dropna()
    if kind == "sharpe":
        mu = r.rolling(window).mean()
        sd = r.rolling(window).std(ddof=1)
        return (mu / sd) * np.sqrt(252)
    elif kind == "vol":
        return r.rolling(window).std(ddof=1) * np.sqrt(252)
    else:
        return r.rolling(window).mean()

def tiny_sparkline(y: pd.Series, height=120):
    fig = go.Figure(go.Scatter(x=y.index, y=y.values, mode="lines", line=dict(width=2)))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

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

# =============================== Load data ===============================
trades, mkt = load_mock_data()

# =============================== Sidebar ===============================
with st.sidebar:
    st.header("Investor Controls")

    data_source = st.radio("Data source", ["Mock demo (trades)", "Upload MT5 OHLCV"], index=0)

    # If mock missing, allow upload
    if trades.empty or mkt.empty:
        xls_up = st.file_uploader("Upload Investor_MockData.xlsx (optional)", type=["xlsx"])
        if xls_up is not None:
            trades = pd.read_excel(xls_up, sheet_name="Trades", engine="openpyxl")
            mkt = pd.read_excel(xls_up, sheet_name="MarketBackground", engine="openpyxl")
            for c in ["Open Time", "Close Time"]:
                if c in trades.columns:
                    trades[c] = pd.to_datetime(trades[c], errors="coerce")
            if "Date" in mkt.columns:
                mkt["Date"] = pd.to_datetime(mkt["Date"], errors="coerce").dt.date

    syms_all = sorted(trades["Symbol"].dropna().unique().tolist()) if not trades.empty else []
    sel_syms = st.multiselect("Symbols (mock demo)", syms_all, default=syms_all)

    risk = st.slider("Risk tolerance (1 = conservative, 10 = aggressive)", 1, 10, 5)

    accounts = sorted(trades["Account"].dropna().unique().tolist()) if not trades.empty else []
    strats = sorted(trades["Strategy"].dropna().unique().tolist()) if not trades.empty else []
    sel_accounts = st.multiselect("Accounts", accounts, default=accounts)
    sel_strats = st.multiselect("Strategies", strats, default=strats)

    st.divider()
    st.subheader("Time Range")
    preset = st.radio(
        "Preset",
        ["Daily","Weekly","Monthly","Yearly","Maximum","Custom"],
        index=4, horizontal=True
    )

    uploaded = st.file_uploader(
        "Upload MT5 export (Tools → History Center → Export)",
        type=["csv","txt","tsv"],
        help="Supports TAB-separated or CSV."
    )

# =============================== Header ===============================
st.markdown("<h2 style='margin-bottom:0'>Investor Helper – Optimal Strategy Guide</h2>", unsafe_allow_html=True)
st.caption("Adjust the risk slider, choose market/time range, and view optimal choices.")

# =============================== Main logic ===============================
# -------- PATH B: MT5 OHLCV upload --------
if data_source.startswith("Upload") and uploaded is not None:
    try:
        oh = parse_mt5_csv(uploaded.read())  # index=datetime, has close
        oh_f = apply_time_range(oh.index, preset, oh)
        if oh_f is None or oh_f.empty:
            st.info("No data in selected range.")
        else:
            price = oh_f["close"].rename("price")
            ret = pct_returns_from_price(price)
            cr = cum_return(ret)
            dd = drawdown_curve(ret)

            # 1) Metric cards + sparklines
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Total Return", f"{(cr.iloc[-1] if len(cr) else 0)*100:.1f}%")
                st.plotly_chart(tiny_sparkline(cr.dropna()), use_container_width=True)
            with colB:
                st.metric("Volatility (ann.)", f"{annualized_vol(ret):.2%}")
                st.plotly_chart(tiny_sparkline(rolling_stat(ret, 30, "vol").dropna()), use_container_width=True)
            with colC:
                st.metric("Sharpe", f"{sharpe_ratio(ret):.2f}")
                st.plotly_chart(tiny_sparkline(rolling_stat(ret, 30, "sharpe").dropna()), use_container_width=True)
            with colD:
                st.metric("Max Drawdown", f"{dd.min():.2%}")
                st.plotly_chart(tiny_sparkline(dd.dropna()), use_container_width=True)

            # 2) Toggle chart + rolling metric
            toggle = st.radio("View", ["Price", "Cumulative Return", "Drawdown"], horizontal=True, key="mt5_view")
            fig_main = go.Figure()
            if toggle == "Price":
                fig_main.add_trace(go.Scatter(x=price.index, y=price, mode="lines", name="Price"))
            elif toggle == "Cumulative Return":
                fig_main.add_trace(go.Scatter(x=cr.index, y=cr, mode="lines", name="CumReturn"))
            else:
                fig_main.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown"))
            fig_main.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=360, legend_title_text="")
            st.plotly_chart(fig_main, use_container_width=True)

            roll_win = st.select_slider("Rolling window (days)", options=[20, 30, 60, 90], value=30, key="mt5_roll")
            roll_series = rolling_stat(ret, roll_win, "sharpe")
            st.plotly_chart(
                go.Figure(go.Scatter(x=roll_series.index, y=roll_series, mode="lines"))
                .update_layout(margin=dict(l=0,r=0,t=10,b=0), height=200),
                use_container_width=True
            )

            # 3) Strategy Lab – comparison (Buy & Hold baseline)
            st.subheader("Strategy Lab – Comparison")
            def row_from_returns(name, r):
                return dict(
                    Strategy=name,
                    TotalReturn=float(cum_return(r).iloc[-1]) if len(r.dropna()) else 0.0,
                    Vol=float(annualized_vol(r)),
                    Sharpe=float(sharpe_ratio(r)),
                    MaxDD=float(drawdown_curve(r).min())
                )
            comp_df = pd.DataFrame([row_from_returns("Buy & Hold", ret)])
            if not comp_df.empty:
                best_sharpe = comp_df.loc[comp_df["Sharpe"].idxmax(), "Strategy"]
                best_return = comp_df.loc[comp_df["TotalReturn"].idxmax(), "Strategy"]
                prof = comp_df[comp_df["TotalReturn"] > 0]
                base = prof if len(prof) else comp_df
                best_minrisk = base.loc[base["Vol"].idxmin(), "Strategy"]
                st.write(f"Highest Reward/Risk: {best_sharpe} | Maximum Profit: {best_return} | Minimum Risk (Profit>0): {best_minrisk}")
                st.dataframe(
                    comp_df.assign(
                        TotalReturn=lambda d: (d["TotalReturn"]*100).round(2),
                        Vol=lambda d: (d["Vol"]*100).round(2),
                        Sharpe=lambda d: d["Sharpe"].round(2),
                        MaxDD=lambda d: (d["MaxDD"]*100).round(2)
                    ).rename(columns={"TotalReturn":"Return %","Vol":"Vol %","MaxDD":"MaxDD %"}),
                    use_container_width=True, height=220
                )

            # 4) Distribution & Drawdown
            st.subheader("Return Distribution and Drawdown")
            colL, colR = st.columns(2)
            with colL:
                st.caption("Return Distribution (per period)")
                hist = px.histogram(ret.dropna(), nbins=40, opacity=0.85)
                hist.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
                st.plotly_chart(hist, use_container_width=True)
            with colR:
                st.caption("Drawdown Curve")
                dd_fig = go.Figure(go.Scatter(x=dd.index, y=dd, mode="lines"))
                dd_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
                st.plotly_chart(dd_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to parse MT5 file: {e}")

# -------- PATH A: Mock trades --------
else:
    if trades.empty or mkt.empty:
        st.warning("Demo Excel not found. Upload Investor_MockData.xlsx on the left, or switch to 'Upload MT5 OHLCV' mode.")
    else:
        f = trades.copy()
        if sel_accounts: f = f[f["Account"].isin(sel_accounts)]
        if sel_strats:   f = f[f["Strategy"].isin(sel_strats)]
        if sel_syms:     f = f[f["Symbol"].isin(sel_syms)]

        total, sym_daily, strat_daily = compute_daily_from_trades(f)

        # Apply time range by converting date to datetime index
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

        # Market risk alerts
        latest_date = mkt["Date"].max() if not mkt.empty else None
        latest_row = mkt[mkt["Date"] == latest_date].iloc[0] if latest_date is not None else None
        alerts = risk_alerts(latest_row) if latest_row is not None else ["No market data."]

        # Allocation + Alerts
        c1, c2 = st.columns([2,1])
        with c1:
            alloc_df = pd.DataFrame({"Symbol": list(alloc.keys()), "Weight": list(alloc.values())}).sort_values("Weight", ascending=False)
            fig_alloc = px.bar(alloc_df, x="Symbol", y="Weight", title="Optimal Allocation (by Risk)", text_auto=".0%")
            fig_alloc.update_yaxes(tickformat=".0%")
            fig_alloc.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=360)
            st.plotly_chart(fig_alloc, use_container_width=True)
        with c2:
            st.markdown("#### Risk Alerts (latest)")
            if latest_row is not None:
                st.write(f"VIX: {latest_row['VIX']:.1f} | FedFundsRate: {latest_row['FedFundsRate']:.2f}% | Inflation: {latest_row['InflationYoY']:.2f}%")
            for mmsg in alerts:
                st.info(mmsg)

        # Backtest tab + metric cards based on CumPNL-derived equity
        tab1, tab2, tab3 = st.tabs(["Portfolio Backtest", "Strategy Insights", "Trades"])
        with tab1:
            if not port.empty:
                fig = px.line(port, x="date", y=["DailyPNL","CumPNL"], title="Backtest: Daily & Cumulative PnL (Optimal Mix)")
                fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=400, legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

                equity = port.set_index("date")["CumPNL"]
                equity = (equity - equity.min()) + 100.0
                price = equity.rename("equity_price")
                ret = pct_returns_from_price(price)
                cr = cum_return(ret)
                dd = drawdown_curve(ret)

                colA, colB, colC, colD = st.columns(4)
                with colA:
                    st.metric("Total Return", f"{(cr.iloc[-1] if len(cr) else 0)*100:.1f}%")
                    st.plotly_chart(tiny_sparkline(cr.dropna()), use_container_width=True)
                with colB:
                    st.metric("Volatility (ann.)", f"{annualized_vol(ret):.2%}")
                    st.plotly_chart(tiny_sparkline(rolling_stat(ret, 30, "vol").dropna()), use_container_width=True)
                with colC:
                    st.metric("Sharpe", f"{sharpe_ratio(ret):.2f}")
                    st.plotly_chart(tiny_sparkline(rolling_stat(ret, 30, "sharpe").dropna()), use_container_width=True)
                with colD:
                    st.metric("Max Drawdown", f"{dd.min():.2%}")
                    st.plotly_chart(tiny_sparkline(dd.dropna()), use_container_width=True)

                # Toggle chart + rolling
                toggle = st.radio("View", ["Cumulative Return", "Drawdown"], horizontal=True, key="mock_view")
                fig_main = go.Figure()
                if toggle == "Cumulative Return":
                    fig_main.add_trace(go.Scatter(x=cr.index, y=cr, mode="lines", name="CumReturn"))
                else:
                    fig_main.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown"))
                fig_main.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=320, legend_title_text="")
                st.plotly_chart(fig_main, use_container_width=True)

                roll_win = st.select_slider("Rolling window (days)", options=[20, 30, 60, 90], value=30, key="mock_roll")
                roll_series = rolling_stat(ret, roll_win, "sharpe")
                st.plotly_chart(
                    go.Figure(go.Scatter(x=roll_series.index, y=roll_series, mode="lines"))
                    .update_layout(margin=dict(l=0,r=0,t=10,b=0), height=200),
                    use_container_width=True
                )

                # Distribution & Drawdown
                st.subheader("Return Distribution and Drawdown")
                colL, colR = st.columns(2)
                with colL:
                    st.caption("Return Distribution (per period)")
                    hist = px.histogram(ret.dropna(), nbins=40, opacity=0.85)
                    hist.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
                    st.plotly_chart(hist, use_container_width=True)
                with colR:
                    st.caption("Drawdown Curve")
                    dd_fig = go.Figure(go.Scatter(x=dd.index, y=dd, mode="lines"))
                    dd_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280)
                    st.plotly_chart(dd_fig, use_container_width=True)

            else:
                st.info("Not enough data for the current filters.")

        with tab2:
            # Strategy Insights (recent contribution by strategy)
            if not strat_daily.empty:
                recent_dates = sorted(strat_daily["date"].unique())
                keep_n = min(20, len(recent_dates))
                last_dates = set(recent_dates[-keep_n:])
                recent = strat_daily[strat_daily["date"].isin(last_dates)]
                g = recent.groupby("Strategy", as_index=False)["StratPNL"].sum().sort_values("StratPNL", ascending=False)
                fig = px.bar(g, x="Strategy", y="StratPNL", title="Recent Strategy Contribution (last ~20 days)", text_auto=".2s")
                fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy data available.")

            # Strategy Lab – comparison table (baseline Buy & Hold on equity_price)
            st.subheader("Strategy Lab – Comparison")
            def row_from_returns(name, r):
                return dict(
                    Strategy=name,
                    TotalReturn=float(cum_return(r).iloc[-1]) if len(r.dropna()) else 0.0,
                    Vol=float(annualized_vol(r)),
                    Sharpe=float(sharpe_ratio(r)),
                    MaxDD=float(drawdown_curve(r).min())
                )
            if not port.empty:
                comp_df = pd.DataFrame([row_from_returns("Buy & Hold", ret)])
                if not comp_df.empty:
                    best_sharpe = comp_df.loc[comp_df["Sharpe"].idxmax(), "Strategy"]
                    best_return = comp_df.loc[comp_df["TotalReturn"].idxmax(), "Strategy"]
                    prof = comp_df[comp_df["TotalReturn"] > 0]
                    base = prof if len(prof) else comp_df
                    best_minrisk = base.loc[base["Vol"].idxmin(), "Strategy"]
                    st.write(f"Highest Reward/Risk: {best_sharpe} | Maximum Profit: {best_return} | Minimum Risk (Profit>0): {best_minrisk}")
                    st.dataframe(
                        comp_df.assign(
                            TotalReturn=lambda d: (d["TotalReturn"]*100).round(2),
                            Vol=lambda d: (d["Vol"]*100).round(2),
                            Sharpe=lambda d: d["Sharpe"].round(2),
                            MaxDD=lambda d: (d["MaxDD"]*100).round(2)
                        ).rename(columns={"TotalReturn":"Return %","Vol":"Vol %","MaxDD":"MaxDD %"}),
                        use_container_width=True, height=220
                    )

            # Multi-asset compare + correlation (based on sym_daily)
            if not sym_daily.empty:
                st.subheader("Multi-Asset View")
                pivot_sym = sym_daily.pivot(index="date", columns="Symbol", values="SymPNL").fillna(0.0)
                eq = pivot_sym.cumsum()
                eq = eq - eq.min() + 100.0
                ret_mat = eq.pct_change()

                all_syms = [c for c in pivot_sym.columns if pivot_sym[c].abs().sum() != 0]
                pick = st.multiselect("Select assets to compare (2 recommended)", all_syms, default=all_syms[:2])
                if len(pick) >= 2:
                    cr_mat = (1 + ret_mat[pick]).cumprod() - 1
                    fig_cmp = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_cmp.add_trace(go.Scatter(x=cr_mat.index, y=cr_mat[pick[0]], name=pick[0]), secondary_y=False)
                    fig_cmp.add_trace(go.Scatter(x=cr_mat.index, y=cr_mat[pick[1]], name=pick[1]), secondary_y=True)
                    fig_cmp.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=340, legend_title_text="")
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    corr = ret_mat[pick].corr()
                    heat = go.Figure(data=go.Heatmap(
                        z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1
                    ))
                    heat.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
                    st.plotly_chart(heat, use_container_width=True)

        with tab3:
            st.dataframe(f.sort_values("Open Time", ascending=False), use_container_width=True, height=420)
            st.download_button("Download filtered trades (CSV)",
                               data=f.to_csv(index=False).encode("utf-8"),
                               file_name="filtered_trades.csv", mime="text/csv")

        # Asset recent contribution (under Allocation area)
        if not sym_daily.empty:
            st.subheader("Asset Contribution")
            lookback = st.select_slider("Contribution window (days)", options=[20, 60, 90], value=20, key="contrib_win")
            last_dates = sorted(sym_daily["date"].unique())[-lookback:] if len(sym_daily) else []
            contrib = sym_daily[sym_daily["date"].isin(last_dates)].groupby("Symbol", as_index=False)["SymPNL"].sum()
            contrib = contrib.sort_values("SymPNL", ascending=False)
            fig_contrib = px.bar(contrib, x="Symbol", y="SymPNL",
                                 title=f"Recent Contribution (last ~{lookback} days)", text_auto=".2s")
            fig_contrib.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=300)
            st.plotly_chart(fig_contrib, use_container_width=True)

st.caption("Mock demo for UX exploration. Not investment advice.")
