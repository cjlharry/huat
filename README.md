# Algorithmic Stock Trading & Strategy Backtesting Dashboard

An open-source interactive financial backtesting web application built with **Streamlit**, **yfinance**, **Pandas**, **NumPy**, and **Plotly**. Designed to run locally or deploy for free on **Streamlit Community Cloud**.

## Features

- **Live Market Data**: Fetches adjusted historical OHLCV data from Yahoo Finance (`yfinance`) with 1-hour caching (`@st.cache_data(ttl=3600)`), inclusive date bounds, and timezone-normalised MultiIndex handling.
- **Lookahead-Free Execution**: Technical signals generated on bar $t$ close are shifted by 1 trading day to execute at bar $t+1$ open.
- **Configurable Strategies**:
  - **SMA Crossover**: Adjustable fast and slow moving average windows.
  - **RSI Mean Reversion**: Wilder's smoothed RSI with custom period, oversold buy threshold, and overbought sell threshold.
  - **Bollinger Band Breakout**: Volatility breakout entries above the upper band with middle-band exits.
- **Risk & Cost Modelling**: Supports initial capital sizing, percentage commissions per trade, intraday stop-loss, and intraday take-profit orders.
- **Fair-Test Tools**: Test-period presets (10Y, 20Y, 3Y, 2007-09 crisis, 2022 bear market), a stop-loss/take-profit switch, and Buy & Hold comparatives for return, CAGR and drawdown. Time in market, win rate, profit factor and average win/loss are reported, with a warning when fewer than 10 trades make the statistics unreliable.
- **Interactive Analytics**: Compare portfolio equity against Buy & Hold, inspect executed entry/exit markers on price charts, monitor open unrealised positions, analyse underwater drawdowns, and export filtered trade logs to CSV.

## Repository Structure

```text
├── app.py              # Streamlit dashboard UI and Plotly visualisations
├── backtester.py       # Data ingestion, indicator math, event-driven backtest engine, and metrics
├── requirements.txt    # Pinned Python dependencies
└── README.md           # Setup and deployment guide

## Disclaimer

Educational use only; not investment advice. Backtests are hypothetical and exclude slippage, taxes and survivorship bias. Past performance does not predict future results.
