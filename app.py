from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import fetch_market_data, run_backtest

st.set_page_config(
    page_title="Algorithmic Strategy Backtester",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Algorithmic Strategy Backtesting Dashboard")
st.caption(
    "Event-driven backtesting engine with next-open execution, transaction costs, and intraday risk controls."
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Market & Date Range")
    ticker = st.text_input("Yahoo Finance Ticker", value="SPY", help="Examples: SPY, AAPL, D05.SI, BTC-USD")

    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 3)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=default_start)
    with col_d2:
        end_date = st.date_input("End Date", value=default_end)

    st.divider()
    st.header("Strategy Parameters")
    strategy = st.selectbox(
        "Select Strategy",
        options=["SMA Crossover", "RSI Mean Reversion", "Bollinger Band Breakout"],
    )

    params = {}
    if strategy == "SMA Crossover":
        params["fast_window"] = st.slider("Fast SMA Window (days)", min_value=5, max_value=100, value=20, step=1)
        params["slow_window"] = st.slider("Slow SMA Window (days)", min_value=20, max_value=300, value=50, step=5)
    elif strategy == "RSI Mean Reversion":
        params["rsi_period"] = st.slider("RSI Period (days)", min_value=5, max_value=30, value=14, step=1)
        params["oversold"] = st.slider("Oversold Buy Threshold", min_value=10, max_value=45, value=30, step=1)
        params["overbought"] = st.slider("Overbought Sell Threshold", min_value=55, max_value=90, value=70, step=1)
    elif strategy == "Bollinger Band Breakout":
        params["bb_window"] = st.slider("Bollinger Window (days)", min_value=10, max_value=100, value=20, step=1)
        params["bb_std"] = st.slider(
            "Standard Deviation Multiplier", min_value=1.0, max_value=3.5, value=2.0, step=0.1
        )

    st.divider()
    st.header("Capital & Risk Controls")
    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=100.0, max_value=10_000_000.0, value=10000.0, step=1000.0
    )
    commission_pct = st.number_input(
        "Commission per Trade (%)", min_value=0.0, max_value=5.0, value=0.10, step=0.05, format="%.2f"
    )
    stop_loss_pct = st.number_input(
        "Stop-Loss (%, 0 = Disabled)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, format="%.1f"
    )
    take_profit_pct = st.number_input(
        "Take-Profit (%, 0 = Disabled)", min_value=0.0, max_value=200.0, value=15.0, step=1.0, format="%.1f"
    )

# --- Validation & Execution ---
if start_date >= end_date:
    st.error("Start Date must be earlier than End Date.")
    st.stop()

try:
    with st.spinner(f"Loading historical data for {ticker.upper()}..."):
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

# --- KPI Metrics Row ---
st.subheader(f"Performance Summary: {ticker.upper()} ({strategy})")

excess_return = metrics["net_return_pct"] - metrics["bh_return_pct"]
pf_display = "Inf" if metrics["profit_factor"] == float("inf") else f"{metrics['profit_factor']:.2f}"

row1_cols = st.columns(4)
row1_cols[0].metric(
    label="Net Return vs. Buy & Hold",
    value=f"{metrics['net_return_pct']:.2f}%",
    delta=f"{excess_return:+.2f}% vs B&H ({metrics['bh_return_pct']:.2f}%)",
)
row1_cols[1].metric(label="CAGR", value=f"{metrics['cagr_pct']:.2f}%")
row1_cols[2].metric(label="Sharpe Ratio", value=f"{metrics['sharpe_ratio']:.2f}")
row1_cols[3].metric(label="Max Drawdown", value=f"{metrics['max_drawdown_pct']:.2f}%")

row2_cols = st.columns(4)
row2_cols[0].metric(label="Total Trades", value=f"{metrics['total_trades']}")
row2_cols[1].metric(label="Win Rate", value=f"{metrics['win_rate_pct']:.1f}%")
row2_cols[2].metric(label="Profit Factor", value=pf_display)
row2_cols[3].metric(label="Final Equity", value=f"${metrics['final_equity']:,.2f}")

st.divider()

# --- Chart 1: Equity Curve ---
st.subheader("Portfolio Equity Curve vs. Buy & Hold")
fig_equity = go.Figure()
fig_equity.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Strategy_Equity"],
        mode="lines",
        name="Strategy Equity",
        line=dict(color="#1f77b4", width=2.2),
    )
)
fig_equity.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Benchmark_Equity"],
        mode="lines",
        name="Buy & Hold Benchmark",
        line=dict(color="#7f7f7f", width=1.8, dash="dash"),
    )
)
fig_equity.update_layout(
    height=380,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified",
    yaxis_title="Portfolio Value ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_equity, use_container_width=True)

# --- Chart 2: Price Action & Trade Signals ---
st.subheader("Asset Close Price & Executed Trade Markers")
fig_price = go.Figure()
fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#2c3e50", width=1.6),
    )
)

# Overlay strategy-specific indicator bands/lines
if strategy == "SMA Crossover":
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_Fast"],
            mode="lines",
            name=f"SMA {params['fast_window']}",
            line=dict(color="#ff7f0e", width=1.2),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA_Slow"],
            mode="lines",
            name=f"SMA {params['slow_window']}",
            line=dict(color="#9467bd", width=1.2),
        )
    )
elif strategy == "Bollinger Band Breakout":
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Upper"],
            mode="lines",
            name="Upper Band",
            line=dict(color="rgba(44, 160, 44, 0.45)", width=1, dash="dot"),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Mid"],
            mode="lines",
            name="Middle Band",
            line=dict(color="rgba(127, 127, 127, 0.5)", width=1, dash="dash"),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Lower"],
            mode="lines",
            name="Lower Band",
            line=dict(color="rgba(214, 39, 40, 0.45)", width=1, dash="dot"),
        )
    )

if not trades_df.empty:
    fig_price.add_trace(
        go.Scatter(
            x=pd.to_datetime(trades_df["Entry Date"]),
            y=trades_df["Entry Price"],
            mode="markers",
            name="Buy Execution",
            marker=dict(symbol="triangle-up", size=11, color="#00c853", line=dict(width=1, color="#004d40")),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=pd.to_datetime(trades_df["Exit Date"]),
            y=trades_df["Exit Price"],
            mode="markers",
            name="Sell Execution",
            marker=dict(symbol="triangle-down", size=11, color="#d50000", line=dict(width=1, color="#4a0000")),
        )
    )

fig_price.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified",
    yaxis_title="Price ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_price, use_container_width=True)

# --- Chart 3: Underwater Drawdown ---
st.subheader("Underwater Drawdown (%)")
fig_dd = go.Figure()
fig_dd.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Drawdown_Pct"],
        mode="lines",
        fill="tozeroy",
        name="Drawdown (%)",
        line=dict(color="#d62728", width=1.4),
        fillcolor="rgba(214, 39, 40, 0.25)",
    )
)
fig_dd.update_layout(
    height=260,
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode="x unified",
    yaxis_title="Drawdown (%)",
)
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# --- Trade Log & CSV Export ---
st.subheader("Executed Trade Log")
if trades_df.empty:
    st.info("No trades were triggered for the selected parameters and date range.")
else:
    col_search, col_export = st.columns([3, 1])
    with col_search:
        search_term = st.text_input(
            "Filter Trade Log (by date or exit reason)",
            placeholder="e.g. Stop-Loss, Signal, 2024-05",
        )
    filtered_trades = trades_df.copy()
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
            label="Download Trade Log (CSV)",
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
            "Entry Price": st.column_config.NumberColumn(format="$%.2f"),
            "Exit Price": st.column_config.NumberColumn(format="$%.2f"),
            "Shares": st.column_config.NumberColumn(format="%.4f"),
            "Net P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
