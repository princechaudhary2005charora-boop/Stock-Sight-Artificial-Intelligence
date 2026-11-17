"""
Smart data fetcher.

Priority order:
  1. Neon DB  — if data is present and up-to-date for the requested range
  2. yfinance — for any missing date range; result is persisted to DB

Graceful fallback: if the DB is unreachable, fetches directly from yfinance
so the app still works without a database connection.
"""
import datetime
import warnings

import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# Lazy DB init — only runs once per Streamlit server process
@st.cache_resource
def _init_db():
    # Database intentionally disabled for cloud demo.
    # Stock data will be fetched directly from Yahoo Finance.
    return False



def _yfinance_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical OHLCV data with a fallback method."""

    df = pd.DataFrame()

    # Method 1: Yahoo Finance download
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
            timeout=20,
        )
    except Exception as e:
        print(f"Download failed for {ticker}: {e}")

    # Method 2: Retry using Ticker.history()
    if df.empty:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start,
                end=end,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"History fallback failed for {ticker}: {e}")

    if df.empty:
        return pd.DataFrame()

    # Handle different yfinance MultiIndex column formats
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = set(df.columns.get_level_values(level))
            if {"Open", "High", "Low", "Close"}.issubset(values):
                df.columns = df.columns.get_level_values(level)
                break

    # Standardize column names
    df.columns = [str(col).title() for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Remove timezone information if present
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"

    return df


def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Return a OHLCV DataFrame indexed by Date for the given ticker and range.

    Serves from Neon DB when possible. Fetches from yfinance only for
    data that isn't already stored, then persists it for future calls.
    Falls back silently to yfinance-only if DB is unavailable.
    """
    db_ok = _init_db()

    today = datetime.date.today()
    start_dt = datetime.date.fromisoformat(start)
    end_dt   = min(datetime.date.fromisoformat(end), today)

    if not db_ok:
        return _yfinance_fetch(ticker, start, str(end_dt))

    try:
        from db import get_sync_info, upsert_ohlcv, query_ohlcv

        sync = get_sync_info(ticker)

        if sync:
            last_synced, db_from, db_to = sync
            synced_today = (last_synced.date() == today)
            covers_range = (db_from <= start_dt) and (db_to >= end_dt)

            if synced_today and covers_range:
                # DB is fresh and covers the full requested range — no API call
                return query_ohlcv(ticker, start, str(end_dt))

            # DB has older data; only fetch the tail we're missing
            if db_to is not None and db_from <= start_dt:
                fetch_start = str(db_to + datetime.timedelta(days=1))
            else:
                fetch_start = start
        else:
            fetch_start = start

        # Fetch only the missing window from yfinance
        raw = _yfinance_fetch(ticker, fetch_start, str(today + datetime.timedelta(days=1)))
        if not raw.empty:
            upsert_ohlcv(ticker, raw)

        return query_ohlcv(ticker, start, str(end_dt))

    except Exception as e:
        # DB unreachable mid-session — fall back to live yfinance
        st.warning(f"⚠️ DB unavailable, using live data: {e}")
        return _yfinance_fetch(ticker, start, str(end_dt))
