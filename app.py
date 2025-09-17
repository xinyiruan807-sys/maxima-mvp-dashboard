
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Maxima Wealth Dashboard – MVP Pro", layout="wide")

# -------------------- Data loading --------------------
@st.cache_data
def load_excel(path: str | None, uploaded=None, sheet="MockData"):
    if uploaded is not None:
        df = pd.read_excel(uploaded, sheet_name=sheet)
    elif path and os.path.exists(path):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        raise FileNotFoundError("No Excel file provided and default path not found.")
    # Normalize types
    for c in ["Open Time","Close Time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["Size","Open Price","Close Price","Commission","Swap","Profit"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Strategy field unify
    if "Strategy" not in df.columns and "Comment" in df.columns:
        df["Strategy"] = df["Comment"]
    return df

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Data Source")
    default_path = "Clean_MockData.xlsx"  # you can change to 'MT4_MT5 Data.xlsx'
    uploaded = st.file_uploader("Upload Excel (sheet: MockData)", type=["xlsx"])
    sheet_name = st.text_input("Sheet name", value="MockData")
    st.caption("If no upload, the app will try to read the file in the working folder.")

    # Load data (with message)
    df = None
    try:
        df = load_excel(default_path, uploaded, sheet=sheet_name)
        st.success("Data loaded.")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    st.header("Filters")
    symbols = sorted(df["Symbol"].dropna().unique().tolist()) if "Symbol" in df.columns else []
    strategies = sorted(df["Strategy"].dropna().unique().tolist()) if "Strategy" in df.columns else []

    sel_symbols = st.multiselect("Symbol", options=symbols, default=symbols)
    sel_strategies = st.multiselect("Strategy", options=strategies, default=strategies)
    date_from = st.date_input("From (Open Time)", value=None)
    date_to = st.date_input("To (Open Time)", value=None)

    st.divider()
    st.subheader("Downloads")
    auto_download = st.checkbox("Auto-refresh downloads on filter", value=True)

# -------------------- Filtering --------------------
fdf = df.copy()
if sel_symbols:
    fdf = fdf[fdf["Symbol"].isin(sel_symbols)]
if sel_strategies:
    fdf = fdf[fdf["Strategy"].isin(sel_strategies)]
if date_from:
    fdf = fdf[fdf["Open Time"] >= pd.to_datetime(date_from)]
if date_to:
    fdf = fdf[fdf["Open Time"] <= pd.to_datetime(date_to)]

# -------------------- KPIs --------------------
st.title("📊 Maxima Wealth Dashboard – MVP (Pro)")
st.caption("Web-like interactive prototype with sidebar filters, buttons, and downloads")

total_profit = float(np.nansum(fdf["Profit"])) if "Profit" in fdf else 0.0
trades = len(fdf)
win_rate = (fdf["Profit"] > 0).mean() * 100 if trades else 0.0
avg_hold = (fdf["Close Time"] - fdf["Open Time"]).dt.total_seconds().mean() / 3600 if trades else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Profit", f"${total_profit:,.2f}")
c2.metric("Total Trades", str(trades))
c3.metric("Win Rate", f"{win_rate:.1f}%")
c4.metric("Avg Holding Time", f"{avg_hold:.2f} h")

st.divider()

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["📈 PnL", "🧪 Distributions", "📋 Trades Table"])

png_cache = {}

with tab1:
    if not fdf.empty:
        cur = fdf.sort_values("Close Time").copy()
        cur["CumProfit"] = cur["Profit"].cumsum()
        fig_pnl = px.line(cur, x="Close Time", y="CumProfit", title="Cumulative PnL")
        fig_pnl.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=420)
        st.plotly_chart(fig_pnl, use_container_width=True)

        # PNG download for PnL
        try:
            buf = fig_pnl.to_image(format="png")
        except Exception:
            # Fallback to HTML download if no kaleido installed
            buf = None
        colA, colB = st.columns([1,4])
        if buf is not None:
            st.download_button("Download PnL as PNG", data=buf, file_name="pnl_curve.png", mime="image/png", use_container_width=False)
        else:
            html_bytes = fig_pnl.to_html(full_html=False).encode("utf-8")
            st.download_button("Download PnL as HTML", data=html_bytes, file_name="pnl_curve.html", mime="text/html", use_container_width=False)
    else:
        st.info("No data for PnL.")

with tab2:
    if not fdf.empty:
        # Strategy PnL
        g_strat = fdf.groupby("Strategy", as_index=False)["Profit"].sum()
        fig_strat = px.bar(g_strat, x="Strategy", y="Profit", title="Strategy PnL", text_auto=".2s")
        fig_strat.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=360)
        st.plotly_chart(fig_strat, use_container_width=True)

        # Symbol mix & PnL
        g_sym = fdf.groupby("Symbol", as_index=False).agg(Count=("Ticket","count"), Profit=("Profit","sum"))
        col1, col2 = st.columns(2)
        with col1:
            fig_mix = px.pie(g_sym, names="Symbol", values="Count", title="Symbol Mix (Trades)")
            fig_mix.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=320)
            st.plotly_chart(fig_mix, use_container_width=True)
        with col2:
            fig_sympnl = px.bar(g_sym, x="Symbol", y="Profit", title="Symbol PnL", text_auto=".2s")
            fig_sympnl.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=320)
            st.plotly_chart(fig_sympnl, use_container_width=True)

        # Download combined charts as HTML
        html_parts = [fig_strat.to_html(full_html=False), fig_mix.to_html(full_html=False), fig_sympnl.to_html(full_html=False)]
        html_bundle = "<hr/>".join(html_parts).encode("utf-8")
        st.download_button("Download Distributions (HTML)", data=html_bundle, file_name="distributions.html", mime="text/html")
    else:
        st.info("No data for distributions.")

with tab3:
    st.markdown("### Filtered Trades")
    st.dataframe(fdf.sort_values("Open Time", ascending=False), use_container_width=True, height=420)

    # CSV download of filtered data
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="filtered_trades.csv", mime="text/csv")

st.caption("Tip: Upload a different Excel, tweak filters, and export charts/CSV for quick stakeholder reviews.")
