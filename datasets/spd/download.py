"""Download stock price data via yfinance with parquet caching."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "PG"]
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2025-12-31"
CACHE_DIR = Path("data/cache")


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    """Deterministic cache key from arguments."""
    key = f"{sorted(tickers)}_{start}_{end}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def download_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | Path = CACHE_DIR,
    force_refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Download adjusted close prices via yfinance.

    Caches to parquet keyed by hash(tickers + dates).

    Args:
        tickers: list of stock tickers.
        start: start date string.
        end: end date string.
        cache_dir: directory for cached parquet files.
        force_refresh: if True, re-download even if cached.

    Returns:
        prices: (T, N) array of adjusted close prices.
        dates: (T,) array of datetime64 dates.
        tickers: list of ticker names in column order.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(tickers, start, end)}.parquet"

    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading cached prices from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        logger.info(f"Downloading prices for {tickers} from {start} to {end}")
        df = yf.download(tickers, start=start, end=end, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        else:
            df = df[["Close"]]
            df.columns = tickers

        df = df[tickers]

        n_missing = df.isna().sum().sum()
        if n_missing > 0:
            logger.warning(f"Found {n_missing} missing values, forward/back-filling")
            df = df.ffill().bfill()

        df.to_parquet(cache_path)
        logger.info(f"Cached prices to {cache_path}")

    dates = df.index.values.astype("datetime64[D]")
    prices = df.values.astype(np.float64)

    logger.info(f"Loaded {len(dates)} trading days for {len(tickers)} stocks")
    return prices, dates, list(tickers)
