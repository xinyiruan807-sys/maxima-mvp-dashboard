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

    # 数据源选择
data_source = st.radio(
    "Data source",
    ["Mock demo (trades)", "Upload MT5 OHLCV", "Upload Market CSV (OHLCV)"],
    index=0
)

mktcsv = st.file_uploader(
    "Upload market CSV (date,symbol,open,high,low,close[,adj_close,volume])",
    type=["csv"],
    key="mktcsv_uploader"
)
# 如果 mock 数据缺失，允许上传 Excel
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

    # 固定候选：完整符号列表（不会因为用户选择变化而消失）
    all_symbols = sorted(trades["Symbol"].dropna().unique().tolist()) if not trades.empty else []
    sel_syms = st.multiselect("Symbols (mock demo)", options=all_symbols, default=all_symbols)

    # 风险滑块
    risk = st.slider("Risk tolerance (1 = conservative, 10 = aggressive)", 1, 10, 5)

    # 账户/策略过滤
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


# =============================== Header ===============================
st.markdown("<h2 style='margin-bottom:0'>Investor Helper – Optimal Strategy Guide</h2>", unsafe_allow_html=True)
st.caption("Adjust the risk slider, choose market/time range, and view optimal choices.")

# =============================== Main logic ===============================

# -------- PATH C: Market CSV (OHLCV) upload --------
if data_source == "Upload Market CSV (OHLCV)":
    if mktcsv is None:
        st.info("Please upload a market CSV.")
    else:
        # ======== 读取与清洗 ========
        df_mkt = pd.read_csv(mktcsv)
        needed = {"date", "symbol", "close"}
        miss = needed - set(df_mkt.columns)
        if miss:
            st.error(f"CSV 缺少列：{sorted(list(miss))}；至少需要 date,symbol,close。")
            st.stop()

        df_mkt["date"] = pd.to_datetime(df_mkt["date"], errors="coerce")
        df_mkt = df_mkt.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])
        if "adj_close" not in df_mkt.columns:
            df_mkt["adj_close"] = df_mkt["close"]
        if "volume" not in df_mkt.columns:
            df_mkt["volume"] = pd.NA

        st.success(f"✅ Loaded {len(df_mkt):,} rows • {df_mkt['symbol'].nunique()} symbols")

        # ======== 选择资产与时间 ========
        all_syms = sorted(df_mkt["symbol"].unique())
        pick = st.multiselect("Symbols", all_syms, default=all_syms[: min(5, len(all_syms))])
        dmin, dmax = df_mkt["date"].min(), df_mkt["date"].max()
        dr = st.slider(
            "Date range",
            min_value=dmin.to_pydatetime(),
            max_value=dmax.to_pydatetime(),
            value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
        )

        view = df_mkt[
            (df_mkt["symbol"].isin(pick))
            & (df_mkt["date"].between(dr[0], dr[1]))
        ].copy()

        # ======== 多资产价格曲线 ========
        st.subheader("Close Price (multi-asset)")
        pivot = view.pivot_table(index="date", columns="symbol", values="close")
        st.line_chart(pivot, use_container_width=True)

        # ======== Quick Metrics ========
        st.subheader("Quick Metrics")
        rows = []
        for sym in pick:
            sub = view[view["symbol"] == sym].set_index("date").sort_index()
            r = pct_returns_from_price(sub["close"])
            cr = cum_return(r)
            dd = drawdown_curve(r)
            rows.append(
                {
                    "Symbol": sym,
                    "Total Return": f"{(cr.iloc[-1] * 100):.1f}%" if len(cr) else None,
                    "Vol (ann.)": f"{(annualized_vol(r) * 100):.1f}%" if len(r.dropna()) else None,
                    "Sharpe": f"{sharpe_ratio(r):.2f}" if len(r.dropna()) else None,
                    "Max DD": f"{(dd.min() * 100):.1f}%" if len(dd.dropna()) else None,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        # ======== 单资产图表 + 滚动指标 ========
        if pick:
            sym0 = pick[0]
            sub0 = view[view["symbol"] == sym0].set_index("date").sort_index()
            r0 = pct_returns_from_price(sub0["close"])
            cr0 = cum_return(r0)
            dd0 = drawdown_curve(r0)

            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Total Return", f"{(cr0.iloc[-1] if len(cr0) else 0)*100:.1f}%")
                st.plotly_chart(tiny_sparkline(cr0.dropna()), use_container_width=True)
            with colB:
                st.metric("Volatility (ann.)", f"{annualized_vol(r0):.2%}")
                st.plotly_chart(tiny_sparkline(rolling_stat(r0, 30, "vol").dropna()), use_container_width=True)
            with colC:
                st.metric("Sharpe", f"{sharpe_ratio(r0):.2f}")
                st.plotly_chart(tiny_sparkline(rolling_stat(r0, 30, "sharpe").dropna()), use_container_width=True)
            with colD:
                st.metric("Max Drawdown", f"{dd0.min():.2%}")
                st.plotly_chart(tiny_sparkline(dd0.dropna()), use_container_width=True)

            st.subheader(f"{sym0} — Price / CumReturn / Drawdown")
            toggle = st.radio("View", ["Price", "Cumulative Return", "Drawdown"], horizontal=True, key="csv_view")
            fig = go.Figure()
            if toggle == "Price":
                fig.add_trace(go.Scatter(x=sub0.index, y=sub0["close"], mode="lines", name="Price"))
            elif toggle == "Cumulative Return":
                fig.add_trace(go.Scatter(x=cr0.index, y=cr0, mode="lines", name="CumReturn"))
            else:
                fig.add_trace(go.Scatter(x=dd0.index, y=dd0, mode="lines", name="Drawdown"))
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=360, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        # ======== Portfolio Backtest（含自定义权重） ========
        st.subheader("Portfolio Backtest (Equal / Custom Weights)")
        import json

        st.markdown("**Weights**")
        wmode = st.radio("Weighting", ["Equal weight", "Custom JSON"], horizontal=True, key="wmode_csv")
        w_text = st.text_input(
            'Custom weights JSON (e.g. {"AAPL":0.3,"BTC-USD":0.2,"XAUUSD=X":0.5})',
            value="", placeholder='{"AAPL":0.3,"BTC-USD":0.2}'
        )

        def make_weights(symbols: list[str], mode: str, text: str) -> dict[str, float]:
            if mode == "Equal weight" or not text.strip():
                n = len(symbols)
                return {s: (1.0 / n if n > 0 else 0.0) for s in symbols}
            try:
                raw = json.loads(text)
                w = {s: float(raw.get(s, 0.0)) for s in symbols}
                ssum = sum(v for v in w.values() if v > 0)
                if ssum <= 0:
                    raise ValueError("sum<=0")
                w = {k: max(0.0, v) / ssum for k, v in w.items()}
                extra = [k for k in raw.keys() if k not in symbols]
                if extra:
                    st.info(f"Ignored symbols not in selection: {extra}")
                return w
            except Exception:
                st.warning("Invalid JSON weights. Fallback to equal weight.")
                n = len(symbols)
                return {s: (1.0 / n if n > 0 else 0.0) for s in symbols}

        px = view.pivot_table(index="date", columns="symbol", values="close").sort_index()
        px = px.dropna(axis=0, how="any")
        if px.shape[1] >= 1:
            symbols_in_view = list(px.columns)
            weights = make_weights(symbols_in_view, wmode, w_text)
            w_vec = np.array([weights.get(s, 0.0) for s in symbols_in_view])
            if w_vec.sum() == 0:
                st.info("All weights are zero. Please adjust.")
            else:
                w_vec = w_vec / w_vec.sum()
                ret_mat = px.pct_change().dropna()
                port_ret = pd.Series(ret_mat.values @ w_vec, index=ret_mat.index, name="port_ret")
                port_eq = (1 + port_ret).cumprod()
                port_dd = (port_eq / port_eq.cummax() - 1.0)

                def _ann_vol(r):
                    r = r.dropna()
                    return float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else None
                def _sharpe(r):
                    r = r.dropna()
                    vol = r.std(ddof=1) * np.sqrt(252)
                    return float((r.mean() * 252) / vol) if (len(r) > 1 and vol > 0) else None

                cA, cB, cC, cD = st.columns(4)
                cA.metric("Portfolio Return", f"{(port_eq.iloc[-1] - 1)*100:.1f}%")
                vol_ann = _ann_vol(port_ret);  cB.metric("Volatility (ann.)", f"{(vol_ann*100):.1f}%" if vol_ann is not None else "–")
                shp = _sharpe(port_ret);       cC.metric("Sharpe", f"{shp:.2f}" if shp is not None else "–")
                cD.metric("Max Drawdown", f"{port_dd.min()*100:.1f}%")

                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(x=port_eq.index, y=port_eq, mode="lines", name="Portfolio NAV"))
                fig_port.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=340, legend_title_text="")
                st.plotly_chart(fig_port, use_container_width=True)

                rows = []
                for sym in symbols_in_view:
                    r = px[sym].pct_change().dropna()
                    eq = (1 + r).cumprod()
                    dd = (eq / eq.cummax() - 1.0).min() if len(eq) else np.nan
                    rows.append({
                        "Symbol": sym,
                        "Weight": f"{weights.get(sym,0.0)*100:.1f}%",
                        "Return %": (eq.iloc[-1] - 1) * 100 if len(eq) else np.nan,
                        "Vol %": _ann_vol(r) * 100 if _ann_vol(r) is not None else np.nan,
                        "Sharpe": _sharpe(r),
                        "MaxDD %": dd * 100 if pd.notna(dd) else np.nan
                    })
                rows.append({
                    "Symbol": "Portfolio",
                    "Weight": "—",
                    "Return %": (port_eq.iloc[-1] - 1) * 100,
                    "Vol %": vol_ann * 100 if vol_ann is not None else np.nan,
                    "Sharpe": shp,
                    "MaxDD %": port_dd.min() * 100
                })
                tbl = pd.DataFrame(rows).round({"Return %": 2, "Vol %": 2, "Sharpe": 2, "MaxDD %": 2})
                st.dataframe(tbl, use_container_width=True, height=300)

                st.download_button(
                    "Download current view (CSV)",
                    data=view.to_csv(index=False).encode("utf-8"),
                    file_name="market_view_filtered.csv",
                    mime="text/csv"
                )
        else:
            st.info("Select at least one symbol to build the portfolio.")

        # ===== Markowitz (SciPy) —— 无需 pypfopt =====
        st.subheader("Optimal Portfolio (Markowitz)")
        from math import sqrt
        from scipy.optimize import minimize

        prices = pivot.copy()  # 这里直接用 pivot（date×symbol 的 close）
        if prices.shape[1] >= 2:
            rets = prices.pct_change().dropna()
            mu_daily = rets.mean()
            mu_ann   = mu_daily * 252
            cov      = rets.cov() * 252
            n        = prices.shape[1]
            syms     = list(prices.columns)

            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
            bounds = tuple((0.0, 1.0) for _ in range(n))
            w0 = np.repeat(1.0/n, n)

            def port_stats(w):
                ret = float(mu_ann @ w)
                vol = float(sqrt(w @ cov.values @ w))
                shp = (ret/vol) if vol>0 else 0.0
                return ret, vol, shp

            def f_min_var(w):  # 最小方差
                return float(w @ cov.values @ w)

            res_mv = minimize(f_min_var, w0, bounds=bounds, constraints=cons, method="SLSQP")
            w_mv = res_mv.x if res_mv.success else w0
            mv_ret, mv_vol, mv_shp = port_stats(w_mv)

            def f_max_sharpe(w):  # 最大夏普 = 最小化 -Sharpe
                ret, vol, _ = port_stats(w)
                return -ret/vol if vol>0 else 1e9

            res_ms = minimize(f_max_sharpe, w0, bounds=bounds, constraints=cons, method="SLSQP")
            w_ms = res_ms.x if res_ms.success else w0
            ms_ret, ms_vol, ms_shp = port_stats(w_ms)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Max Sharpe Weights**")
            df_ms = pd.DataFrame({"Symbol": syms, "Weight": w_ms}).sort_values("Weight", ascending=False)
            st.dataframe(
                df_ms.assign(Weight=lambda d: (d["Weight"] * 100).round(1)).rename(columns={"Weight": "Weight %"}),
                use_container_width=True, height=260
            )
            st.caption(f"Performance → Return: {ms_ret:.2%} | Vol: {ms_vol:.2%} | Sharpe: {ms_shp:.2f}")

        with col2:
            st.markdown("**Min Variance Weights**")
            df_mv = pd.DataFrame({"Symbol": syms, "Weight": w_mv}).sort_values("Weight", ascending=False)
            st.dataframe(
                df_mv.assign(Weight=lambda d: (d["Weight"] * 100).round(1)).rename(columns={"Weight": "Weight %"}),
                use_container_width=True, height=260
            )
            st.caption(f"Performance → Return: {mv_ret:.2%} | Vol: {mv_vol:.2%} | Sharpe: {mv_shp:.2f}")

    else:
        st.info("Need at least 2 symbols for Markowitz optimization.")

    # ===== 风险贡献 (MCR/RC) =====
    st.subheader("Risk Contribution")

    rets_full = pivot.pct_change().dropna()
    if rets_full.shape[1] >= 2:
        cov_full = rets_full.cov()
        sym_list = list(pivot.columns)

        # 等权权重（如需改用自定义权重，可替换 w_eq）
        w_eq = np.repeat(1 / len(sym_list), len(sym_list))

        port_var = float(w_eq.T @ cov_full.values @ w_eq)
        mcr = (cov_full.values @ w_eq) / np.sqrt(port_var)
        rc = w_eq * mcr
        rc_df = (
            pd.DataFrame({"Symbol": sym_list, "RiskContribPct": rc / rc.sum()})
            .sort_values("RiskContribPct", ascending=False)
        )

        fig_rc = px.bar(rc_df, x="Symbol", y="RiskContribPct")
        fig_rc.update_traces(
            text=(rc_df["RiskContribPct"] * 100).round(1).astype(str) + "%",
            textposition="outside"
        )
        fig_rc.update_yaxes(tickformat=".0%")
        fig_rc.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.info("Need at least 2 symbols for risk contribution.")


# -------- PATH A: Mock trades --------
elif data_source == "Mock demo (trades)":
    if trades.empty or mkt.empty:
        st.warning("Demo Excel not found. Upload Investor_MockData.xlsx on the left, or switch to CSV mode.")
    else:
        # ======== 过滤 Trades ========
        f = trades.copy()
        if sel_accounts:
            f = f[f["Account"].isin(sel_accounts)]
        if sel_strats:
            f = f[f["Strategy"].isin(sel_strats)]
        if sel_syms:
            f = f[f["Symbol"].isin(sel_syms)]

        # ======== 聚合为日频 ========
        total, sym_daily, strat_daily = compute_daily_from_trades(f)

        # ======== 应用时间范围（把 date 作为索引以复用 apply_time_range） ========
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

        # ======== 按风险给出配置并生成组合 PnL 曲线 ========
        syms_for_alloc = sorted(f["Symbol"].dropna().unique().tolist())
        alloc = risk_to_allocation(risk, syms_for_alloc)
        port = portfolio_curve_from_alloc(sym_daily, alloc) if not sym_daily.empty else pd.DataFrame()

        # ======== 市场风险提示 ========
        latest_date = mkt["Date"].max() if not mkt.empty else None
        latest_row = mkt[mkt["Date"] == latest_date].iloc[0] if latest_date is not None else None
        alerts = risk_alerts(latest_row) if latest_row is not None else ["No market data."]

        # ======== 左：配置柱状图；右：风险提示 ========
        c1, c2 = st.columns([2, 1])
        with c1:
            alloc_df = pd.DataFrame({"Symbol": list(alloc.keys()), "Weight": list(alloc.values())}).sort_values("Weight", ascending=False)
            fig_alloc = px.bar(alloc_df, x="Symbol", y="Weight", title="Optimal Allocation (by Risk)", text_auto=".0%")
            fig_alloc.update_yaxes(tickformat=".0%")
            fig_alloc.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=360)
            st.plotly_chart(fig_alloc, use_container_width=True)
        with c2:
            st.markdown("#### Risk Alerts (latest)")
            if latest_row is not None:
                st.write(f"VIX: {latest_row['VIX']:.1f} | FedFundsRate: {latest_row['FedFundsRate']:.2f}% | Inflation: {latest_row['InflationYoY']:.2f}%")
            for mmsg in alerts:
                st.info(mmsg)

        # ======== Tabs：回测 / 策略洞察 / 交易明细 ========
        tab1, tab2, tab3 = st.tabs(["Portfolio Backtest", "Strategy Insights", "Trades"])

        # --- Tab1: Backtest ---
        with tab1:
            if not port.empty:
                fig = px.line(port, x="date", y=["DailyPNL", "CumPNL"], title="Backtest: Daily & Cumulative PnL (Optimal Mix)")
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=400, legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

                # 用累计 PnL 构造“等价净值”以计算指标
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

                # 主图切换 + 滚动窗口
                toggle = st.radio("View", ["Cumulative Return", "Drawdown"], horizontal=True, key="mock_view")
                fig_main = go.Figure()
                if toggle == "Cumulative Return":
                    fig_main.add_trace(go.Scatter(x=cr.index, y=cr, mode="lines", name="CumReturn"))
                else:
                    fig_main.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown"))
                fig_main.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=320, legend_title_text="")
                st.plotly_chart(fig_main, use_container_width=True)

                roll_win = st.select_slider("Rolling window (days)", options=[20, 30, 60, 90], value=30, key="mock_roll")
                roll_series = rolling_stat(ret, roll_win, "sharpe")
                st.plotly_chart(
                    go.Figure(go.Scatter(x=roll_series.index, y=roll_series, mode="lines"))
                    .update_layout(margin=dict(l=0, r=0, t=10, b=0), height=200),
                    use_container_width=True
                )
                # 分布 & 回撤曲线
                st.subheader("Return Distribution and Drawdown")
                colL, colR = st.columns(2)
                with colL:
                    st.caption("Return Distribution (per period)")
                    hist = px.histogram(ret.dropna(), nbins=40, opacity=0.85)
                    hist.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
                    st.plotly_chart(hist, use_container_width=True)
                with colR:
                    st.caption("Drawdown Curve")
                    dd_fig = go.Figure(go.Scatter(x=dd.index, y=dd, mode="lines"))
                    dd_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
                    st.plotly_chart(dd_fig, use_container_width=True)
            else:
                st.info("Not enough data for the current filters.")

        # --- Tab2: Strategy Insights ---
        with tab2:
            # 策略近期贡献
            if not strat_daily.empty:
                recent_dates = sorted(strat_daily["date"].unique())
                keep_n = min(20, len(recent_dates))
                last_dates = set(recent_dates[-keep_n:])
                recent = strat_daily[strat_daily["date"].isin(last_dates)]
                g = recent.groupby("Strategy", as_index=False)["StratPNL"].sum().sort_values("StratPNL", ascending=False)
                fig = px.bar(g, x="Strategy", y="StratPNL", title="Recent Strategy Contribution (last ~20 days)", text_auto=".2s")
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy data available.")

            # Buy & Hold 基准（基于 equity_price）
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
                comp_df = pd.DataFrame([row_from_returns("Buy & Hold", pct_returns_from_price((port.set_index("date")["CumPNL"] - port["CumPNL"].min() + 100.0)))])
                if not comp_df.empty:
                    st.dataframe(
                        comp_df.assign(
                            TotalReturn=lambda d: (d["TotalReturn"]*100).round(2),
                            Vol=lambda d: (d["Vol"]*100).round(2),
                            Sharpe=lambda d: d["Sharpe"].round(2),
                            MaxDD=lambda d: (d["MaxDD"]*100).round(2)
                        ).rename(columns={"TotalReturn":"Return %","Vol":"Vol %","MaxDD":"MaxDD %"}),
                        use_container_width=True, height=220
                    )

            # 多资产比较 + 相关性
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
                    fig_cmp.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=340, legend_title_text="")
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    corr = ret_mat[pick].corr()
                    heat = go.Figure(data=go.Heatmap(
                        z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1
                    ))
                    heat.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
                    st.plotly_chart(heat, use_container_width=True)

        # --- Tab3: Trades 明细 ---
        with tab3:
            st.dataframe(f.sort_values("Open Time", ascending=False), use_container_width=True, height=420)
            st.download_button(
                "Download filtered trades (CSV)",
                data=f.to_csv(index=False).encode("utf-8"),
                file_name="filtered_trades.csv",
                mime="text/csv"
            )

        # ======== 资产贡献度（页面下方） ========
        if not sym_daily.empty:
            st.subheader("Asset Contribution")
            lookback = st.select_slider("Contribution window (days)", options=[20, 60, 90], value=20, key="contrib_win")
            last_dates = sorted(sym_daily["date"].unique())[-lookback:] if len(sym_daily) else []
            contrib = sym_daily[sym_daily["date"].isin(last_dates)].groupby("Symbol", as_index=False)["SymPNL"].sum()
            contrib = contrib.sort_values("SymPNL", ascending=False)
            fig_contrib = px.bar(contrib, x="Symbol", y="SymPNL",
                                 title=f"Recent Contribution (last ~{lookback} days)", text_auto=".2s")
            fig_contrib.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=300)
            st.plotly_chart(fig_contrib, use_container_width=True)

# ======== 页脚提示 ========
st.caption("Mock demo for UX exploration. Not investment advice.")

