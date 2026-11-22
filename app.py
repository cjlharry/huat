from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from backtester import fetch_market_data as _fetch_market_data
from backtester import run_backtest

st.set_page_config(
    page_title="Quantitative Strategy Backtester",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    return _fetch_market_data(ticker, start_date, end_date)


# --- Custom CSS for terminal aesthetics ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 31, 38, 0.45);
        border: 1px solid rgba(128, 134, 145, 0.22);
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        opacity: 0.85;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### Market & Timeframe")
    preset_cols = st.columns(4)
    presets = ["SPY", "QQQ", "D05.SI", "BTC-USD"]
    for idx, sym in enumerate(presets):
        if preset_cols[idx].button(sym, use_container_width=True):
            st.session_state["ticker_input"] = sym

    ticker = st.text_input(
        "Yahoo Finance Ticker",
        value=st.session_state.get("ticker_input", "SPY"),
        help="Enter any Yahoo Finance symbol (e.g. SPY, NVDA, D05.SI, 9988.HK, BTC-USD)",
    )

    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 3)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=default_start)
    with col_d2:
        end_date = st.date_input("End Date", value=default_end)

    chart_style = st.radio("Price Chart Style", options=["Candlestick", "Line"], horizontal=True)

    st.divider()
    st.markdown("### Strategy Engine")
    strategy = st.selectbox(
        "Strategy Model",
        options=["SMA Crossover", "RSI Mean Reversion", "Bollinger Band Breakout"],
    )

    params = {}
    if strategy == "SMA Crossover":
        params["fast_window"] = st.slider("Fast SMA (days)", min_value=5, max_value=100, value=20, step=1)
        params["slow_window"] = st.slider("Slow SMA (days)", min_value=20, max_value=300, value=50, step=5)
    elif strategy == "RSI Mean Reversion":
        params["rsi_period"] = st.slider("RSI Period (days)", min_value=5, max_value=30, value=14, step=1)
        params["oversold"] = st.slider("Oversold Buy Level", min_value=10, max_value=45, value=30, step=1)
        params["overbought"] = st.slider("Overbought Sell Level", min_value=55, max_value=90, value=70, step=1)
    elif strategy == "Bollinger Band Breakout":
        params["bb_window"] = st.slider("Band Window (days)", min_value=10, max_value=100, value=20, step=1)
        params["bb_std"] = st.slider("Std Dev Multiplier", min_value=1.0, max_value=3.5, value=2.0, step=0.1)

    st.divider()
    st.markdown("### Capital & Risk Controls")
    initial_capital = st.number_input(
        "Starting Capital", min_value=100.0, max_value=10_000_000.0, value=10000.0, step=1000.0
    )
    commission_pct = st.number_input(
        "Commission (%)", min_value=0.0, max_value=5.0, value=0.10, step=0.05, format="%.2f"
    )
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        stop_loss_pct = st.number_input(
            "Stop-Loss (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, format="%.1f"
        )
    with col_r2:
        take_profit_pct = st.number_input(
            "Take-Profit (%)", min_value=0.0, max_value=200.0, value=15.0, step=1.0, format="%.1f"
        )

# --- Validation & Execution ---
if start_date >= end_date:
    st.error("Start Date must be earlier than End Date.")
    st.stop()

try:
    with st.spinner(f"Fetching market data and simulating {ticker.upper()}..."):
        raw_df = fetch_market_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        results = run_backtest(
            df=raw_df,
            strategy=strategy,
            params=params,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
except ValueError as err:
    st.error(str(err))
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while running backtest: {exc}")
    st.stop()

df = results["data"]
trades_df = results["trades"]
metrics = results["metrics"]
open_position = results["open_position"]

# --- Top Header Bar ---
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(f"## {ticker.upper()} | {strategy}")
    st.caption(
        f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}  |  "
        f"Next-Open Execution  |  Comm: {commission_pct:.2f}%  |  "
        f"SL: {stop_loss_pct:.1f}%  |  TP: {take_profit_pct:.1f}%"
    )
with header_right:
    latest_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    daily_chg = ((latest_close - prev_close) / prev_close) * 100.0
    st.metric(
        label="Latest Close",
        value=f"{latest_close:,.2f}",
        delta=f"{daily_chg:+.2f}% (1D)",
    )

# --- KPI Cards Grid ---
excess_return = metrics["net_return_pct"] - metrics["bh_return_pct"]
pf_display = "Inf" if metrics["profit_factor"] == float("inf") else f"{metrics['profit_factor']:.2f}"

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    label="Strategy Net Return",
    value=f"{metrics['net_return_pct']:+.2f}%",
    delta=f"{excess_return:+.2f}% vs B&H ({metrics['bh_return_pct']:+.2f}%)",
)
k2.metric(
    label="Final Portfolio Value",
    value=f"{metrics['final_equity']:,.2f}",
    delta=f"{metrics['final_equity'] - initial_capital:+,.2f}",
)
k3.metric(
    label="CAGR / Sharpe Ratio",
    value=f"{metrics['cagr_pct']:.2f}%",
    delta=f"Sharpe: {metrics['sharpe_ratio']:.2f}",
    delta_color="off",
)
k4.metric(
    label="Max Drawdown",
    value=f"{metrics['max_drawdown_pct']:.2f}%",
    delta=f"{metrics['total_trades']} Trades | Win: {metrics['win_rate_pct']:.1f}% | PF: {pf_display}",
    delta_color="off",
)

if open_position is not None:
    st.info(
        f"**Active Open Position:** Entered on {open_position['entry_date']} at "
        f"{open_position['entry_price']:,.2f} ({open_position['shares']:.4f} units)  |  "
        f"Current: {open_position['current_price']:,.2f}  |  "
        f"Unrealised P&L: {open_position['unrealised_pnl']:+,.2f} ({open_position['unrealised_return_pct']:+.2f}%)"
    )

# --- Main Workspace Tabs ---
tab_charts, tab_trades = st.tabs(["Performance & Technical Charts", f"Trade Ledger ({len(trades_df)})"])

with tab_charts:
    # Row 1: Equity Curve (Left) + Underwater Drawdown (Right)
    col_eq, col_dd = st.columns([1.65, 1.0])

    with col_eq:
        fig_equity = go.Figure()
        fig_equity.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Strategy_Equity"],
                mode="lines",
                name="Strategy Equity",
                line=dict(color="#00b4d8", width=2.4),
                fill="tozeroy",
                fillcolor="rgba(0, 180, 216, 0.08)",
            )
        )
        fig_equity.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Benchmark_Equity"],
                mode="lines",
                name="Buy & Hold",
                line=dict(color="#9aa5b1", width=1.6, dash="dot"),
            )
        )
        fig_equity.update_layout(
            title=dict(text="Equity Curve vs. Buy & Hold", font=dict(size=15)),
            height=340,
            margin=dict(l=15, r=15, t=42, b=15),
            hovermode="x unified",
            yaxis_title="Equity",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_equity.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
        fig_equity.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
        st.plotly_chart(fig_equity, use_container_width=True)

    with col_dd:
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Drawdown_Pct"],
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#ef476f", width=1.6),
                fillcolor="rgba(239, 71, 111, 0.22)",
            )
        )
        fig_dd.update_layout(
            title=dict(text="Underwater Drawdown (%)", font=dict(size=15)),
            height=340,
            margin=dict(l=15, r=15, t=42, b=15),
            hovermode="x unified",
            yaxis_title="Drawdown (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_dd.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
        fig_dd.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
        st.plotly_chart(fig_dd, use_container_width=True)

    # Row 2: Synchronised Two-Panel Price + Signals + Sub-Indicator Chart
    sub_title = "RSI Oscillator" if strategy == "RSI Mean Reversion" else "Trading Volume"
    fig_price = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.74, 0.26],
        subplot_titles=(f"{ticker.upper()} Price Action & Executions", sub_title),
    )

    if chart_style == "Candlestick":
        fig_price.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color="#06d6a0",
                decreasing_line_color="#ef476f",
            ),
            row=1,
            col=1,
        )
    else:
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color="#e2e8f0", width=1.8),
            ),
            row=1,
            col=1,
        )

    # Strategy Overlays on Top Panel
    if strategy == "SMA Crossover":
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_Fast"],
                mode="lines",
                name=f"SMA {params['fast_window']}",
                line=dict(color="#ffd166", width=1.4),
            ),
            row=1,
            col=1,
        )
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_Slow"],
                mode="lines",
                name=f"SMA {params['slow_window']}",
                line=dict(color="#118ab2", width=1.4),
            ),
            row=1,
            col=1,
        )
    elif strategy == "Bollinger Band Breakout":
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                mode="lines",
                name="Upper Band",
                line=dict(color="rgba(6, 214, 160, 0.5)", width=1.1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                mode="lines",
                name="Lower Band",
                fill="tonexty",
                fillcolor="rgba(0, 180, 216, 0.06)",
                line=dict(color="rgba(239, 71, 111, 0.5)", width=1.1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Mid"],
                mode="lines",
                name="Mid Band",
                line=dict(color="#ffd166", width=1.1, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Entry & Exit Markers
    buy_dates, buy_prices = [], []
    if not trades_df.empty:
        buy_dates.extend(pd.to_datetime(trades_df["Entry Date"]).tolist())
        buy_prices.extend(trades_df["Entry Price"].tolist())
    if open_position is not None:
        buy_dates.append(pd.to_datetime(open_position["entry_date"]))
        buy_prices.append(open_position["entry_price"])

    if buy_dates:
        fig_price.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="Buy Entry",
                marker=dict(symbol="triangle-up", size=12, color="#06d6a0", line=dict(width=1.2, color="#073b4c")),
            ),
            row=1,
            col=1,
        )

    if not trades_df.empty:
        fig_price.add_trace(
            go.Scatter(
                x=pd.to_datetime(trades_df["Exit Date"]),
                y=trades_df["Exit Price"],
                mode="markers",
                name="Sell Exit",
                marker=dict(symbol="triangle-down", size=12, color="#ef476f", line=dict(width=1.2, color="#073b4c")),
            ),
            row=1,
            col=1,
        )

    # Bottom Panel: RSI or Volume
    if strategy == "RSI Mean Reversion" and "RSI" in df.columns:
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig_price.add_hline(y=params["overbought"], line_dash="dash", line_color="#ef476f", row=2, col=1)
        fig_price.add_hline(y=params["oversold"], line_dash="dash", line_color="#06d6a0", row=2, col=1)
        fig_price.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)
    else:
        vol_colors = [
            "rgba(6, 214, 160, 0.45)" if c >= o else "rgba(239, 71, 111, 0.45)"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig_price.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig_price.update_yaxes(title_text="Volume", row=2, col=1)

    fig_price.update_layout(
        height=540,
        margin=dict(l=15, r=15, t=35, b=15),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_price.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    fig_price.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    fig_price.update_yaxes(title_text="Price", row=1, col=1)
    st.plotly_chart(fig_price, use_container_width=True)

with tab_trades:
    if trades_df.empty:
        st.info("No closed trades were recorded for the selected parameters and date range.")
    else:
        col_search, col_filter, col_export = st.columns([2.5, 1.5, 1.0])
        with col_search:
            search_term = st.text_input(
                "Search Dates or Values",
                placeholder="e.g. 2024-08",
            )
        with col_filter:
            reasons = ["All"] + sorted(trades_df["Exit Reason"].unique().tolist())
            selected_reason = st.selectbox("Filter by Exit Reason", options=reasons)

        filtered_trades = trades_df.copy()
        if selected_reason != "All":
            filtered_trades = filtered_trades[filtered_trades["Exit Reason"] == selected_reason]
        if search_term.strip():
            mask = filtered_trades.apply(
                lambda row: search_term.lower() in " ".join(row.astype(str)).lower(), axis=1
            )
            filtered_trades = filtered_trades[mask]

        with col_export:
            st.write("")
            st.write("")
            csv_bytes = filtered_trades.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export CSV",
                data=csv_bytes,
                file_name=f"{ticker.upper()}_{strategy.lower().replace(' ', '_')}_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(
            filtered_trades,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Entry Price": st.column_config.NumberColumn(format="%.2f"),
                "Exit Price": st.column_config.NumberColumn(format="%.2f"),
                "Shares": st.column_config.NumberColumn(format="%.4f"),
                "Net P&L": st.column_config.NumberColumn(format="%+.2f"),
                "Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )
