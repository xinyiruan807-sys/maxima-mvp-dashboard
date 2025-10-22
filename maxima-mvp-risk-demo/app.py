# app.py — Maxima Wealth Dashboard (Market-only, auto-updated CSV)
# Data source: maxima-mvp-risk-demo/data/market_latest.csv (updated by GitHub Actions)
# Requirements: streamlit, pandas, numpy, plotly

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

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

def weights_by_risk(px_df: pd.DataFrame, level: str) -> dict:
    """
    Risk-based weights using in-view assets:
    - Low: inverse volatility tilt (更偏低波动)
    - Medium: 50% equal + 50% inverse-vol
    - High: volatility tilt (更偏高波动)
    """
    symbols = list(px_df.columns)
    if len(symbols) == 0:
        return {}

    # compute annualized vol per symbol
    vols = {}
    for s in symbols:
        r = px_df[s].pct_change().dropna()
        v = ann_vol(r)
        vols[s] = float(v) if (v is not None and np.isfinite(v)) else np.nan

    # fallback: if too many NaN, use equal
    vol_series = pd.Series(vols).replace([np.inf, -np.inf], np.nan)
    if vol_series.isna().sum() >= len(vol_series) - 1:
        return {s: 1/len(symbols) for s in symbols}

    eps = 1e-9
    vol_filled = vol_series.fillna(vol_series.median())

    if level == "Low":
        raw = 1.0 / (vol_filled + eps)
    elif level == "High":
        raw = vol_filled.clip(lower=eps)
    else:  # Medium
        inv = 1.0 / (vol_filled + eps)
        eq  = pd.Series(1.0, index=vol_filled.index)
        raw = 0.5 * inv + 0.5 * eq

    w = (raw / raw.sum()).to_dict()
    return {k: float(v) for k, v in w.items()}

def build_portfolio_json(px_clean: pd.DataFrame,
                         weights: dict,
                         risk_level: str | None,
                         date_range: tuple[datetime, datetime],
                         port_ret: pd.Series,
                         extra_metrics: dict) -> dict:
    """Construct a clean JSON profile for AI & export."""
    assets = []
    for s in px_clean.columns:
        r = px_clean[s].pct_change().dropna()
        eq = (1 + r).cumprod()
        asset = {
            "symbol": s,
            "weight": round(float(weights.get(s, 0.0)), 6),
            "last_price": float(px_clean[s].dropna().iloc[-1]),
            "total_return": round(float(eq.iloc[-1] - 1) if len(eq) else float("nan"), 6),
            "vol_ann": round(float(ann_vol(r)) if len(r) else float("nan"), 6),
            "sharpe": round(float(sharpe(r)) if len(r) else float("nan"), 6)
        }
        assets.append(asset)

    port_eq = (1 + port_ret).cumprod()
    port_dd = (port_eq / port_eq.cummax() - 1.0)

    profile = {
        "portfolio_name": f"{risk_level} Risk Portfolio" if risk_level else "Custom Portfolio",
        "risk_level": risk_level,
        "date_range": {
            "start": pd.to_datetime(date_range[0]).strftime("%Y-%m-%d"),
            "end": pd.to_datetime(date_range[1]).strftime("%Y-%m-%d")
        },
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assets": assets,
        "metrics": {
            "portfolio_total_return": round(float(port_eq.iloc[-1] - 1), 6) if len(port_eq) else None,
            "portfolio_vol_ann": round(float(ann_vol(port_ret)), 6) if len(port_ret) else None,
            "portfolio_sharpe": round(float(sharpe(port_ret)), 6) if len(port_ret) else None,
            "portfolio_max_drawdown": round(float(port_dd.min()), 6) if len(port_dd) else None
        },
        "extra": extra_metrics
    }
    return profile

def explain_portfolio(profile: dict) -> str:
    """Generate a human-readable explanation based on portfolio JSON (mock GPT)."""
    risk = profile.get("risk_level") or "Custom"
    assets = profile.get("assets", [])
    metrics = profile.get("metrics", {})
    # Top-3 by weight
    top = sorted(assets, key=lambda x: x.get("weight", 0), reverse=True)[:3]
    top_str = ", ".join([f'{a["symbol"]} {a["weight"]*100:.1f}%' for a in top])

    ret = metrics.get("portfolio_total_return")
    vol = metrics.get("portfolio_vol_ann")
    shp = metrics.get("portfolio_sharpe")
    dd  = metrics.get("portfolio_max_drawdown")

    parts = []
    parts.append(f"This is a **{risk.lower()}-risk** portfolio built from the symbols you selected.")
    if top:
        parts.append(f"Top allocations: {top_str}.")
    if ret is not None:
        parts.append(f"Portfolio total return over the selected period is **{ret*100:.1f}%**.")
    if vol is not None:
        parts.append(f"Annualized volatility is **{vol*100:.1f}%**, with a Sharpe ratio of **{shp:.2f}**." if shp is not None else
                     f"Annualized volatility is **{vol*100:.1f}%**.")
    if dd is not None:
        parts.append(f"Maximum drawdown reached **{dd*100:.1f}%**.")
    parts.append("Weights were assigned automatically based on measured volatility under the chosen risk level "
                 "(low = inverse-volatility tilt; high = volatility tilt; medium = blended).")
    return " ".join(parts)

# ------------------------ Load data ------------------------
df_mkt = load_market_csv()

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.header("Investor Controls")
    st.caption("Using auto-updated market data (data/market_latest.csv)")
    if df_mkt.empty:
        st.warning("No market data found. Ensure GitHub Actions has produced data/market_latest.csv")
        pick, dr, mode, w_text, risk_level = [], None, "Equal", "", None
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
        mode = st.radio("Weighting", ["Equal", "Risk Levels", "Custom JSON"], horizontal=True)
        risk_level = None
        if mode == "Risk Levels":
            risk_level = st.selectbox("Select Risk Level", ["Low", "Medium", "High"], index=1)
        w_text = st.text_input(
            'Custom weights JSON (e.g. {"AAPL":0.4,"MSFT":0.3,"GLD":0.3})',
            value="" if mode != "Custom JSON" else ""
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
st.subheader("Portfolio Backtest (Equal / Risk Levels / Custom)")

def make_weights(symbols: list[str], mode: str, px_df: pd.DataFrame, risk_level: str | None, text: str) -> dict[str, float]:
    if len(symbols) == 0:
        return {}
    if mode == "Equal":
        return {s: 1.0/len(symbols) for s in symbols}
    if mode == "Risk Levels":
        w = weights_by_risk(px_df, risk_level or "Medium")
        # ensure keys limited to current symbols
        w = {s: w.get(s, 0.0) for s in symbols}
        ssum = sum(w.values())
        return {k: v/ssum for k, v in w.items()} if ssum > 0 else {s: 1.0/len(symbols) for s in symbols}
    # Custom JSON
    try:
        raw = json.loads(text) if text and text.strip() else {}
        w = {s: float(raw.get(s, 0.0)) for s in symbols}
        ssum = sum(v for v in w.values() if v > 0)
        if ssum <= 0:
            raise ValueError("sum<=0")
        w = {k: max(0.0, v) / ssum for k, v in w.items()}
        return w
    except Exception:
        st.warning("Invalid JSON weights. Fallback to equal weight.")
        return {s: 1.0/len(symbols) for s in symbols}

px_clean = px.dropna(how="any")
if px_clean.shape[1] >= 1:
    symbols_in_view = list(px_clean.columns)
    weights = make_weights(symbols_in_view, mode, px_clean, risk_level, w_text if 'w_text' in locals() else "")
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

        # component table
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

        # ---------- JSON Export + Explain ----------
        st.divider()
        st.subheader("AI Profile Export & Explanation")

        # extra metadata for profile
        extra = {
            "weighting_mode": mode,
            "symbols_in_view": symbols_in_view,
            "notes": "Weights computed via inverse-volatility/volatility tilt for risk levels; equal or custom otherwise."
        }
        profile = build_portfolio_json(
            px_clean=px_clean[symbols_in_view],
            weights=weights,
            risk_level=risk_level if mode == "Risk Levels" else None,
            date_range=dr if dr else (px_clean.index.min(), px_clean.index.max()),
            port_ret=port_ret,
            extra_metrics=extra
        )

        # download JSON
        json_bytes = json.dumps(profile, indent=2).encode("utf-8")
        st.download_button(
            "Download portfolio profile (JSON)",
            data=json_bytes,
            file_name="portfolio_profile.json",
            mime="application/json",
            help="JSON schema for AI integration (Clarity API / Gemini / Grok, etc.)"
        )

        # show JSON preview (collapsed)
        with st.expander("Preview JSON profile"):
            st.json(profile, expanded=False)

        # Explain button (mock GPT)
        if st.button("Explain this portfolio"):
            explanation = explain_portfolio(profile)
            st.markdown(explanation)

        # also keep CSV download for filtered view
        st.download_button(
            "Download filtered market view (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="market_view_filtered.csv",
            mime="text/csv"
        )

st.caption("Data auto-updated daily via GitHub Actions → data/market_latest.csv")
