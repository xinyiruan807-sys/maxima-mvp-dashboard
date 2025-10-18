# app.py — Maxima Wealth Dashboard (Market-only, auto-updated CSV)
# Data source: maxima-mvp-risk-demo/data/market_latest.csv (updated by GitHub Actions)
# Requirements: streamlit, pandas, numpy, plotly

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------ Page config ------------------------
st.set_page_config(
    page_title="Maxima Wealth – Market Dashboard",
    layout="wide",
    page_icon="📈"
)

# ------------------------ Helpers ------------------------
def pct_returns_from_price(price: pd.Series) -> pd.Series:
    return price.sort_index().pct_change()

def cum_return(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return (1 + r).cumprod() - 1

def drawdown_curve(r: pd.Series) -> pd.Series:
    eq = (1 + r.fillna(0)).cumprod()
    peak = eq.cummax()
    return eq / peak - 1

def ann_vol(r: pd.Series, per_year: int = 252) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(per_year)) if len(r) > 1 else np.nan

def sharpe(r: pd.Series, per_year: int = 252) -> float:
    r = r.dropna()
    vol = r.std(ddof=1) * np.sqrt(per_year) if len(r) > 1 else np.nan
    return float((r.mean() * per_year) / vol) if (vol and vol > 0) else np.nan

@st.cache_data(show_spinner=False)
def load_market_csv() -> pd.DataFrame:
    """
    Load the auto-updated CSV with robust path resolution (works on Streamlit Cloud and local).
    Tries multiple candidate paths and shows which one is used.
    """
    base = Path(__file__).parent          # e.g., <repo>/maxima-mvp-risk-demo/
    cwd  = Path.cwd()                     # repo root on Streamlit Cloud

    candidates = [
        base / "data" / "market_latest.csv",                       # preferred
        cwd / "maxima-mvp-risk-demo" / "data" / "market_latest.csv",
        cwd / "data" / "market_latest.csv",                        # fallback if placed at repo root
    ]

    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break

    st.caption(f"CSV path resolved: {csv_path if csv_path else 'NOT FOUND'}")  # for debugging

    if not csv_path:
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # normalize
    colmap = {
        "date": "date", "Date": "date",
        "ticker": "symbol", "Symbol": "symbol",
        "open": "open", "Open": "open",
        "high": "high", "High": "high",
        "low": "low", "Low": "low",
        "close": "close", "Close": "close",
        "adj_close": "adj_close", "Adj Close": "adj_close",
        "volume": "volume", "Volume": "volume",
    }
    df = df.rename(columns=colmap)
    needed = {"date", "symbol", "close"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = pd.NA
    return df

def tiny_sparkline(x, y, title="", height=120):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    if title:
        fig.update_layout(title=title)
    return fig

# ------------------------ Load data ------------------------
df_mkt = load_market_csv()

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.header("Investor Controls")
    st.caption("Using auto-updated market data (data/market_latest.csv)")
    if df_mkt.empty:
        st.warning("No market data found. Ensure GitHub Actions has produced data/market_latest.csv")
        pick, dr, mode, w_text = [], None, "Equal", ""
    else:
        all_syms = sorted(df_mkt["symbol"].unique().tolist())
        default_syms = all_syms[: min(6, len(all_syms))]
        pick = st.multiselect("Symbols", all_syms, default=default_syms)

        dmin, dmax = df_mkt["date"].min(), df_mkt["date"].max()
        dr = st.slider(
            "Date range",
            min_value=dmin.to_pydatetime(),
            max_value=dmax.to_pydatetime(),
            value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
        )

        st.divider()
        st.subheader("Portfolio Weights")
        mode = st.radio("Weighting", ["Equal", "Custom JSON"], horizontal=True)
        w_text = st.text_input(
            'Custom weights JSON (e.g. {"AAPL":0.4,"MSFT":0.3,"GLD":0.3})',
            value=""
        )

# ------------------------ Main ------------------------
st.markdown("<h2 style='margin-bottom:0'>Market Overview</h2>", unsafe_allow_html=True)
if df_mkt.empty:
    st.info("Waiting for auto-updated CSV…")
    st.stop()

# filter view
pick = pick if pick else sorted(df_mkt["symbol"].unique().tolist())[:6]
view = df_mkt[
    (df_mkt["symbol"].isin(pick)) &
    (df_mkt["date"].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1])) if dr else True)
].copy()

if view.empty:
    st.info("No data for current filters.")
    st.stop()

# Price panel
st.subheader("Close Price (multi-asset)")
px = view.pivot_table(index="date", columns="symbol", values="close").sort_index()
st.line_chart(px, use_container_width=True)

# Quick metrics table
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
            "Vol (ann.)": f"{(ann_vol(r) * 100):.1f}%" if len(r.dropna()) else None,
            "Sharpe": f"{sharpe(r):.2f}" if len(r.dropna()) else None,
            "Max DD": f"{(dd.min() * 100):.1f}%" if len(dd.dropna()) else None,
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

# Single asset panel
sym0 = pick[0]
sub0 = view[view["symbol"] == sym0].set_index("date").sort_index()
r0 = pct_returns_from_price(sub0["close"])
cr0 = cum_return(r0)
dd0 = drawdown_curve(r0)

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Total Return", f"{(cr0.iloc[-1] if len(cr0) else 0)*100:.1f}%")
    st.plotly_chart(tiny_sparkline(cr0.index, cr0.values), use_container_width=True)
with colB:
    st.metric("Volatility (ann.)", f"{ann_vol(r0):.2%}")
    st.plotly_chart(tiny_sparkline(r0.index, r0.rolling(30).std(ddof=1)*np.sqrt(252)), use_container_width=True)
with colC:
    st.metric("Sharpe", f"{sharpe(r0):.2f}")
    st.plotly_chart(tiny_sparkline(r0.index, (r0.rolling(30).mean()/r0.rolling(30).std(ddof=1))*np.sqrt(252)), use_container_width=True)
with colD:
    st.metric("Max Drawdown", f"{dd0.min():.2%}")
    st.plotly_chart(tiny_sparkline(dd0.index, dd0.values), use_container_width=True)

st.subheader("Price / CumReturn / Drawdown")
toggle = st.radio("View", ["Price", "Cumulative Return", "Drawdown"], horizontal=True, key="main_view")
fig = go.Figure()
if toggle == "Price":
    fig.add_trace(go.Scatter(x=sub0.index, y=sub0["close"], mode="lines", name="Price"))
elif toggle == "Cumulative Return":
    fig.add_trace(go.Scatter(x=cr0.index, y=cr0, mode="lines", name="CumReturn"))
else:
    fig.add_trace(go.Scatter(x=dd0.index, y=dd0, mode="lines", name="Drawdown"))
fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=360, legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

# ------------------------ Portfolio backtest ------------------------
st.subheader("Portfolio Backtest (Equal / Custom Weights)")

def make_weights(symbols: list[str], mode: str, text: str) -> dict[str, float]:
    if mode == "Equal" or not text.strip():
        n = len(symbols)
        return {s: (1.0 / n if n > 0 else 0.0) for s in symbols}
    try:
        raw = json.loads(text)
        w = {s: float(raw.get(s, 0.0)) for s in symbols}
        ssum = sum(v for v in w.values() if v > 0)
        if ssum <= 0:
            raise ValueError("sum<=0")
        w = {k: max(0.0, v) / ssum for k, v in w.items()}
        return w
    except Exception:
        st.warning("Invalid JSON weights. Fallback to equal weight.")
        n = len(symbols)
        return {s: (1.0 / n if n > 0 else 0.0) for s in symbols}

px_clean = px.dropna(how="any")
if px_clean.shape[1] >= 1:
    symbols_in_view = list(px_clean.columns)
    weights = make_weights(symbols_in_view, mode, w_text if 'w_text' in locals() else "")
    w_vec = np.array([weights.get(s, 0.0) for s in symbols_in_view])
    if w_vec.sum() == 0:
        st.info("All weights are zero. Please adjust.")
    else:
        w_vec = w_vec / w_vec.sum()
        ret_mat = px_clean.pct_change().dropna()
        port_ret = pd.Series(ret_mat.values @ w_vec, index=ret_mat.index, name="port_ret")
        port_eq = (1 + port_ret).cumprod()
        port_dd = (port_eq / port_eq.cummax() - 1.0)

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Portfolio Return", f"{(port_eq.iloc[-1] - 1)*100:.1f}%")
        vol_ann = ann_vol(port_ret);  cB.metric("Volatility (ann.)", f"{(vol_ann*100):.1f}%" if vol_ann is not None else "–")
        shp = sharpe(port_ret);       cC.metric("Sharpe", f"{shp:.2f}" if shp is not None else "–")
        cD.metric("Max Drawdown", f"{port_dd.min()*100:.1f}%")

        fig_port = go.Figure(go.Scatter(x=port_eq.index, y=port_eq, mode="lines", name="Portfolio NAV"))
        fig_port.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=340, legend_title_text="")
        st.plotly_chart(fig_port, use_container_width=True)

        # table
        rows = []
        for sym in symbols_in_view:
            r = px_clean[sym].pct_change().dropna()
            eq = (1 + r).cumprod()
            dd_ = (eq / eq.cummax() - 1.0).min() if len(eq) else np.nan
            rows.append({
                "Symbol": sym,
                "Weight %": weights.get(sym, 0.0) * 100,
                "Return %": (eq.iloc[-1] - 1) * 100 if len(eq) else np.nan,
                "Vol %": ann_vol(r) * 100 if ann_vol(r) is not None else np.nan,
                "Sharpe": sharpe(r),
                "MaxDD %": dd_ * 100 if pd.notna(dd_) else np.nan
            })
        rows.append({
            "Symbol": "Portfolio",
            "Weight %": np.nan,
            "Return %": (port_eq.iloc[-1] - 1) * 100,
            "Vol %": vol_ann * 100 if vol_ann is not None else np.nan,
            "Sharpe": shp,
            "MaxDD %": port_dd.min() * 100
        })
        tbl = pd.DataFrame(rows).round({"Return %": 2, "Vol %": 2, "Sharpe": 2, "MaxDD %": 2, "Weight %": 1})
        st.dataframe(tbl, use_container_width=True, height=320)

        st.download_button(
            "Download filtered view (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="market_view_filtered.csv",
            mime="text/csv"
        )

st.caption("Data auto-updated daily via GitHub Actions → data/market_latest.csv")
