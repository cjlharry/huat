import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical OHLCV data from Yahoo Finance and normalise columns."""
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Please enter a valid ticker symbol.")

    # Make end_date inclusive since yfinance treats end as exclusive
    end_inclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        clean_ticker,
        start=start_date,
        end=end_inclusive,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"No market data returned for '{clean_ticker}' between {start_date} and {end_date}. "
            "Please check the ticker symbol and date range."
        )

    required_cols = {"Open", "High", "Low", "Close", "Volume"}

    # Safely flatten MultiIndex columns regardless of level ordering
    if isinstance(df.columns, pd.MultiIndex):
        level_0 = set(df.columns.get_level_values(0))
        level_1 = set(df.columns.get_level_values(1))
        if required_cols.issubset(level_0):
            df.columns = df.columns.get_level_values(0)
        elif required_cols.issubset(level_1):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()].copy()

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required market data columns: {', '.join(sorted(missing_cols))}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Normalise DatetimeIndex to timezone-naive timestamps
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx
    df.sort_index(inplace=True)

    if len(df) < 5:
        raise ValueError(
            f"Insufficient data points ({len(df)} rows) for '{clean_ticker}'. "
            "Please select a wider date range."
        )

    return df


def generate_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """Compute technical indicators and shift raw signals by 1 bar to avoid lookahead bias."""
    data = df.copy()
    close = data["Close"]

    if strategy == "SMA Crossover":
        fast_window = int(params["fast_window"])
        slow_window = int(params["slow_window"])
        if fast_window >= slow_window:
            raise ValueError("Fast SMA window must be strictly smaller than Slow SMA window.")

        data["SMA_Fast"] = close.rolling(window=fast_window).mean()
        data["SMA_Slow"] = close.rolling(window=slow_window).mean()

        cross_above = (data["SMA_Fast"] > data["SMA_Slow"]) & (
            data["SMA_Fast"].shift(1) <= data["SMA_Slow"].shift(1)
        )
        cross_below = (data["SMA_Fast"] < data["SMA_Slow"]) & (
            data["SMA_Fast"].shift(1) >= data["SMA_Slow"].shift(1)
        )

        raw_signal = pd.Series(0, index=data.index)
        raw_signal[cross_above] = 1
        raw_signal[cross_below] = -1

    elif strategy == "RSI Mean Reversion":
        period = int(params["rsi_period"])
        oversold = float(params["oversold"])
        overbought = float(params["overbought"])
        if oversold >= overbought:
            raise ValueError("Oversold threshold must be lower than Overbought threshold.")

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Preserve warmup NaNs while handling flat or zero-loss windows accurately
        both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
        only_loss_zero = (avg_loss == 0.0) & (avg_gain > 0.0)
        rsi = np.where(both_zero, 50.0, np.where(only_loss_zero, 100.0, rsi))
        data["RSI"] = pd.Series(rsi, index=data.index)

        buy_cond = (data["RSI"] < oversold) & (data["RSI"].shift(1) >= oversold)
        sell_cond = (data["RSI"] > overbought) & (data["RSI"].shift(1) <= overbought)

        raw_signal = pd.Series(0, index=data.index)
        raw_signal[buy_cond] = 1
        raw_signal[sell_cond] = -1

    elif strategy == "Bollinger Band Breakout":
        window = int(params["bb_window"])
        num_std = float(params["bb_std"])

        mid = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        data["BB_Mid"] = mid
        data["BB_Upper"] = mid + num_std * std
        data["BB_Lower"] = mid - num_std * std

        breakout_up = (close > data["BB_Upper"]) & (close.shift(1) <= data["BB_Upper"].shift(1))
        breakdown_mid = (close < data["BB_Mid"]) & (close.shift(1) >= data["BB_Mid"].shift(1))

        raw_signal = pd.Series(0, index=data.index)
        raw_signal[breakout_up] = 1
        raw_signal[breakdown_mid] = -1

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    # Shift signals by 1 bar so decisions made at bar t close execute at bar t+1 open
    data["Signal"] = raw_signal.shift(1).fillna(0).astype(int)
    return data


def run_backtest(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
) -> dict:
    """Run an event-driven long-only backtest with next-open execution and risk controls."""
    data = generate_signals(df, strategy, params)

    comm_rate = commission_pct / 100.0
    sl_rate = stop_loss_pct / 100.0
    tp_rate = take_profit_pct / 100.0

    cash = float(initial_capital)
    shares = 0.0
    entry_price = 0.0
    entry_date = None
    entry_cost_total = 0.0

    equity_curve = []
    trades = []

    dates = data.index
    opens = data["Open"].to_numpy()
    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()
    closes = data["Close"].to_numpy()
    signals = data["Signal"].to_numpy()

    for i in range(len(data)):
        date = dates[i]
        open_p = float(opens[i])
        high_p = float(highs[i])
        low_p = float(lows[i])
        close_p = float(closes[i])
        sig = int(signals[i])

        # 1. Execute scheduled open orders from prior bar's signal
        if shares == 0.0 and sig == 1 and open_p > 0.0:
            max_investable = cash / (1.0 + comm_rate)
            if max_investable > 0.0:
                shares = max_investable / open_p
                entry_commission = shares * open_p * comm_rate
                entry_cost_total = shares * open_p + entry_commission
                cash -= entry_cost_total
                entry_price = open_p
                entry_date = date

        elif shares > 0.0 and sig == -1 and open_p > 0.0:
            gross_proceeds = shares * open_p
            exit_commission = gross_proceeds * comm_rate
            net_proceeds = gross_proceeds - exit_commission
            cash += net_proceeds

            net_pnl = net_proceeds - entry_cost_total
            ret_pct = (net_pnl / entry_cost_total) * 100.0
            trades.append(
                {
                    "Entry Date": entry_date.strftime("%Y-%m-%d"),
                    "Exit Date": date.strftime("%Y-%m-%d"),
                    "Entry Price": round(entry_price, 4),
                    "Exit Price": round(open_p, 4),
                    "Shares": round(shares, 4),
                    "Net P&L": round(net_pnl, 2),
                    "Return (%)": round(ret_pct, 2),
                    "Exit Reason": "Signal",
                }
            )
            shares = 0.0
            entry_price = 0.0
            entry_date = None
            entry_cost_total = 0.0

        # 2. Check intraday Stop-Loss and Take-Profit while holding a position
        if shares > 0.0:
            sl_price = entry_price * (1.0 - sl_rate) if sl_rate > 0.0 else -np.inf
            tp_price = entry_price * (1.0 + tp_rate) if tp_rate > 0.0 else np.inf

            hit_sl = sl_rate > 0.0 and low_p <= sl_price
            hit_tp = tp_rate > 0.0 and high_p >= tp_price

            if hit_sl or hit_tp:
                # Conservative assumption: evaluate Stop-Loss first if both breach intraday
                if hit_sl:
                    exec_price = min(open_p, sl_price)
                    reason = "Stop-Loss"
                else:
                    exec_price = max(open_p, tp_price)
                    reason = "Take-Profit"

                gross_proceeds = shares * exec_price
                exit_commission = gross_proceeds * comm_rate
                net_proceeds = gross_proceeds - exit_commission
                cash += net_proceeds

                net_pnl = net_proceeds - entry_cost_total
                ret_pct = (net_pnl / entry_cost_total) * 100.0
                trades.append(
                    {
                        "Entry Date": entry_date.strftime("%Y-%m-%d"),
                        "Exit Date": date.strftime("%Y-%m-%d"),
                        "Entry Price": round(entry_price, 4),
                        "Exit Price": round(exec_price, 4),
                        "Shares": round(shares, 4),
                        "Net P&L": round(net_pnl, 2),
                        "Return (%)": round(ret_pct, 2),
                        "Exit Reason": reason,
                    }
                )
                shares = 0.0
                entry_price = 0.0
                entry_date = None
                entry_cost_total = 0.0

        # 3. Record end-of-day mark-to-market equity
        daily_equity = cash + shares * close_p
        equity_curve.append(daily_equity)

    # Track any open position separately without inflating closed trade statistics
    open_position = None
    if shares > 0.0:
        final_close = float(closes[-1])
        market_value = shares * final_close
        unrealised_pnl = market_value - entry_cost_total
        open_position = {
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 4),
            "current_price": round(final_close, 4),
            "shares": round(shares, 4),
            "unrealised_pnl": round(unrealised_pnl, 2),
            "unrealised_return_pct": round((unrealised_pnl / entry_cost_total) * 100.0, 2),
        }

    data["Strategy_Equity"] = equity_curve

    # Buy & Hold benchmark (buy at first bar's Open with commission)
    first_open = float(opens[0])
    if first_open > 0.0:
        bh_shares = (initial_capital / (1.0 + comm_rate)) / first_open
        data["Benchmark_Equity"] = bh_shares * data["Close"]
    else:
        data["Benchmark_Equity"] = float(initial_capital)

    # Drawdown series (%)
    rolling_peak = data["Strategy_Equity"].cummax()
    data["Drawdown_Pct"] = ((data["Strategy_Equity"] - rolling_peak) / rolling_peak) * 100.0

    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(data, trades_df, initial_capital)

    return {
        "data": data,
        "trades": trades_df,
        "metrics": metrics,
        "open_position": open_position,
    }


def calculate_metrics(data: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float) -> dict:
    """Compute portfolio and closed trade-level performance statistics."""
    final_equity = max(float(data["Strategy_Equity"].iloc[-1]), 0.0)
    bh_final = max(float(data["Benchmark_Equity"].iloc[-1]), 0.0)

    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0
    bh_return_pct = ((bh_final - initial_capital) / initial_capital) * 100.0

    days = (data.index[-1] - data.index[0]).days
    years = max(days / 365.25, 1.0 / 252.0)
    cagr_pct = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    daily_returns = data["Strategy_Equity"].pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0.0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    max_drawdown_pct = float(data["Drawdown_Pct"].min())

    total_trades = len(trades_df)
    if total_trades > 0:
        wins = trades_df[trades_df["Net P&L"] > 0]
        losses = trades_df[trades_df["Net P&L"] < 0]
        win_rate_pct = (len(wins) / total_trades) * 100.0

        gross_profit = float(wins["Net P&L"].sum())
        gross_loss = abs(float(losses["Net P&L"].sum()))
        if gross_loss > 0.0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0

    return {
        "final_equity": final_equity,
        "net_return_pct": net_return_pct,
        "bh_return_pct": bh_return_pct,
        "cagr_pct": cagr_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
    }
