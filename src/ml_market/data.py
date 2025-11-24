"""Data fetching module for stock market OHLCV data."""

import pandas as pd
import yfinance as yf
import numpy as np


def fetch_ohlcv(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for given tickers.

    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        start: Start date in 'YYYY-MM-DD' format (optional)
        end: End date in 'YYYY-MM-DD' format (optional)
        period: Period to fetch if start/end not provided (default: '1y')
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume

    Example:
        >>> df = fetch_ohlcv(['AAPL', 'MSFT'], period='1mo')
        >>> df.head()
    """
    if not tickers:
        raise ValueError("At least one ticker symbol must be provided")

    all_data = []

    for ticker in tickers:
        # Download data for single ticker
        ticker_obj = yf.Ticker(ticker)

        if start and end:
            df = ticker_obj.history(start=start, end=end)
        else:
            df = ticker_obj.history(period=period)

        if df.empty:
            continue

        # Reset index to make date a column
        df = df.reset_index()

        # Add ticker column
        df["ticker"] = ticker

        # Rename columns to lowercase
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Select only the columns we need
        df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]

        all_data.append(df)

    if not all_data:
        raise ValueError(f"No data found for tickers: {tickers}")

    # Concatenate all dataframes
    result = pd.concat(all_data, ignore_index=True)

    # Sort by date and ticker
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)

    return result  # type: ignore[no-any-return]


# ================================================================
# PULL & COMPUTE SECTOR DATA (XLK)
# ================================================================
def load_sector_data(start="2010-01-01", end=None):
    ticker = "XLK"
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns.values]

    df = data.reset_index()  # Make date a column
    df.rename(columns=str.lower, inplace=True)

    # Robust handling of close column naming
    close_col_candidates = [c for c in df.columns if "close" in c]
    if len(close_col_candidates) == 0:
        raise ValueError("No close price column found in sector download")
    close_col = close_col_candidates[0]  # pick first match

    df.rename(columns={close_col: "close"}, inplace=True)

    df = df[["date", "close"]].sort_values("date")

    df["xlk_ret_1d"] = df["close"].pct_change()
    df["xlk_vol_20"] = df["xlk_ret_1d"].rolling(20).std()
    df["xlk_mom_10"] = df["close"].pct_change(10)

    return df


# ================================================================
# PULL & COMPUTE MACRO / MARKET CONTEXT DATA
# ================================================================
def load_macro_data(start="2010-01-01", end=None):
    symbols = ["SPY", "QQQ", "TLT", "DX-Y.NYB", "^VIX"]
    rename_map = {"DX-Y.NYB": "DXY", "^VIX": "VIX"}

    frames = []

    for s in symbols:
        raw = yf.download(s, start=start, end=end, auto_adjust=True)

        # flatten multiindex columns if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(col).strip() for col in raw.columns.values]

        df = raw.reset_index()
        df.rename(columns=str.lower, inplace=True)

        # find whichever column contains 'close'

        close_candidates = [c for c in df.columns if "close" in c]

        if len(close_candidates) == 0:
            raise ValueError(f"No close-like column found for {s}")
        close_col = close_candidates[0]

        label = rename_map.get(s, s).lower()  # use short clean name
        df.rename(columns={close_col: f"{label}_close"}, inplace=True)

        df = df[["date", f"{label}_close"]].sort_values("date")

        df[f"{label}_ret_1d"] = df[f"{label}_close"].pct_change()
        df[f"{label}_vol_20"] = df[f"{label}_ret_1d"].rolling(20).std()
        df[f"{label}_mom_10"] = df[f"{label}_close"].pct_change(10)

        frames.append(df)

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="date", how="outer")

    return macro.sort_values("date")
