
import os, math
import pandas as pd, numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Maxima Wealth - Investor Helper (Risk Slider)", layout="wide", page_icon="🧭")

@st.cache_data
def load_data(path="Investor_MockData.xlsx"):
    trades = pd.read_excel(path, sheet_name="Trades")
    mkt = pd.read_excel(path, sheet_name="MarketBackground")
    for c in ["Open Time","Close Time"]:
        trades[c] = pd.to_datetime(trades[c], errors="coerce")
    for c in ["Size","Open Price","Close Price","Commission","Swap","Profit","RecommendedPosition","RiskScore","InitCapital"]:
        if c in trades.columns:
            trades[c] = pd.to_numeric(trades[c], errors="coerce")
    mkt["Date"] = pd.to_datetime(mkt["Date"], errors="coerce").dt.date
    return trades, mkt

def compute_daily(trades: pd.DataFrame):
    t = trades.copy()
    t["date"] = t["Close Time"].dt.date
    daily_total = t.groupby("date", as_index=False)["Profit"].sum().rename(columns={"Profit": "DailyPNL"})
    sym = t.groupby(["date","Symbol"], as_index=False)["Profit"].sum().rename(columns={"Profit":"SymPNL"})
    strat = t.groupby(["date","Strategy"], as_index=False)["Profit"].sum().rename(columns={"Profit":"StratPNL"})
    return daily_total, sym, strat

def risk_to_allocation(risk: int, symbols: list):
    conservative = {"XAUUSD": 0.35, "US30": 0.30, "AAPL": 0.20, "EURUSD": 0.10, "BTCUSD": 0.05}
    balanced     = {"XAUUSD": 0.25, "US30": 0.25, "AAPL": 0.25, "EURUSD": 0.15, "BTCUSD": 0.10}
    aggressive   = {"XAUUSD": 0.10, "US30": 0.15, "AAPL": 0.30, "EURUSD": 0.10, "BTCUSD": 0.35}
    def blend(a, b, w):
        all_syms = set(a) | set(b) | set(symbols)
        out = {}
        for s in all_syms:
            out[s] = (a.get(s,0)*(1-w) + b.get(s,0)*w)
        ssum = sum(out.values()) or 1.0
        for s in out: out[s] = out[s]/ssum
        return out
    if risk <= 3:
        w = (risk-1)/2.0
        alloc = blend(conservative, balanced, w)
    elif risk <= 7:
        w = (risk-4)/3.0
        mid = blend(conservative, balanced, 1.0)
        alloc = blend(mid, aggressive, w)
    else:
        w = (risk-8)/2.0
        alloc = blend(balanced, aggressive, 0.5 + 0.5*w)
    alloc = {s: alloc.get(s,0.0) for s in symbols}
    ssum = sum(alloc.values()) or 1.0
    alloc = {k: v/ssum for k,v in alloc.items()}
    return alloc

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

def risk_alerts(mkt_row):
    msgs = []
    if mkt_row is None or len(mkt_row)==0:
        return ["No market data."]
    vix = float(mkt_row["VIX"])
    rate = float(mkt_row["FedFundsRate"])
    inf = float(mkt_row["InflationYoY"])
    if vix >= 25: msgs.append("High volatility (VIX >= 25): consider reducing leverage and tightening stops.")
    elif vix >= 18: msgs.append("Elevated volatility: watch position sizing and avoid over-trading.")
    else: msgs.append("Calm market regime: conditions favorable for trend-following.")
    if rate > 5.5: msgs.append("Rising rate environment may pressure growth stocks; favor quality & cash-flow assets.")
    if inf > 3.5: msgs.append("Sticky inflation: commodities/defensives may outperform; keep duration risk controlled.")
    return msgs

trades, mkt = load_data()

with st.sidebar:
    st.header("Investor Controls")
    risk = st.slider("Risk tolerance (1 = conservative, 10 = aggressive)", 1, 10, 5)
    syms = sorted(trades["Symbol"].dropna().unique().tolist())
    strats = sorted(trades["Strategy"].dropna().unique().tolist())
    accounts = sorted(trades["Account"].dropna().unique().tolist())
    sel_accounts = st.multiselect("Accounts", accounts, default=accounts)
    sel_strats = st.multiselect("Strategies", strats, default=strats)

f = trades.copy()
if sel_accounts: f = f[f["Account"].isin(sel_accounts)]
if sel_strats: f = f[f["Strategy"].isin(sel_strats)]

daily_total, sym_daily, strat_daily = compute_daily(f)
alloc = risk_to_allocation(risk, syms)
port = portfolio_curve_from_alloc(sym_daily, alloc)

latest_date = mkt["Date"].max() if not mkt.empty else None
latest_row = mkt[mkt["Date"]==latest_date].iloc[0] if latest_date is not None else None
alerts = risk_alerts(latest_row) if latest_row is not None else ["No market data."]

st.markdown("<h2 style='margin-bottom:0'>Investor Helper - Risk-aware Strategy Guide</h2>", unsafe_allow_html=True)
st.caption("Move the risk slider, get a recommended mix and see a simple backtest.")

c1, c2 = st.columns([2,1])
with c1:
    alloc_df = pd.DataFrame({"Symbol": list(alloc.keys()), "Weight": list(alloc.values())}).sort_values("Weight", ascending=False)
    fig_alloc = px.bar(alloc_df, x="Symbol", y="Weight", title="Recommended Allocation (by Risk)", text_auto=".0%")
    fig_alloc.update_yaxes(tickformat=".0%")
    fig_alloc.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=360)
    st.plotly_chart(fig_alloc, use_container_width=True)
with c2:
    st.markdown("#### Risk Alerts (latest)")
    if latest_row is not None:
        st.write(f"VIX: {latest_row['VIX']:.1f} | FedFundsRate: {latest_row['FedFundsRate']:.2f}% | Inflation: {latest_row['InflationYoY']:.2f}%")
    for m in alerts:
        st.info(m)

tab1, tab2, tab3 = st.tabs(["Portfolio Backtest", "Strategy Insights", "Trades"])

with tab1:
    if not port.empty:
        fig = px.line(port, x="date", y=["DailyPNL","CumPNL"], title="Backtest: Daily & Cumulative PnL (Recommended Mix)")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=400, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        total = float(port["DailyPNL"].sum())
        mdd = float((port["CumPNL"].cummax() - port["CumPNL"]).max()) if not port.empty else 0.0
        sharpe = float((port["DailyPNL"].mean() / (port["DailyPNL"].std(ddof=1) or 1e-9)) * np.sqrt(252)) if len(port) > 2 else 0.0
        k1,k2,k3 = st.columns(3)
        k1.metric("Total PnL (demo)", f"${total:,.0f}")
        k2.metric("Max Drawdown", f"${mdd:,.0f}")
        k3.metric("Sharpe (daily)", f"{sharpe:.2f}")
    else:
        st.info("Not enough data for the current filters.")

with tab2:
    if not strat_daily.empty:
        recent_dates = sorted(strat_daily["date"].unique())
        keep_n = min(20, len(recent_dates))
        last_dates = set(recent_dates[-keep_n:])
        recent = strat_daily[strat_daily["date"].isin(last_dates)]
        g = recent.groupby("Strategy", as_index=False)["StratPNL"].sum().sort_values("StratPNL", ascending=False)
        fig = px.bar(g, x="Strategy", y="StratPNL", title="Recent Strategy Contribution (last ~20 days)", text_auto=".2s")
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tip: Prefer strategies with consistent positive contribution under your chosen risk.")
    else:
        st.info("No strategy data available.")

with tab3:
    st.dataframe(f.sort_values("Open Time", ascending=False), use_container_width=True, height=420)
    st.download_button("Download filtered trades (CSV)", data=f.to_csv(index=False).encode("utf-8"), file_name="filtered_trades.csv", mime="text/csv")

st.caption("Mock demo for UX exploration. Not investment advice.")
