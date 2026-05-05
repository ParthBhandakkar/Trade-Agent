"""
Data Loader for Strategy 09 Backtester
=======================================

Loads historical OHLCV data from local CSV files exported from Exness via
Google Drive.  CSV structure (per Drive):

    time_utc, Day_IST, Time_IST, time, open, high, low, close,
    tick_volume, spread, real_volume

Directory layout expected:

    backtest/data/<SYMBOL>/<timeframe>.csv
    e.g.  backtest/data/USDJPY/4h.csv

The loader:
  1. Reads the CSV
  2. Parses ``time_utc`` as the datetime index (UTC-aware)
  3. Filters to the requested date range
  4. Returns a clean DataFrame with columns: open, high, low, close, volume

**Parity with live MT5:** backtest signals only match ``auto_trader`` when these OHLC rows are
the **same aggregation** as the terminal that feeds the bot (same server session / 4H bar
opens). If CSVs use a different UTC phase (e.g. bars at 00/04/08/… vs 01/05/09/… server time),
liquidity sweeps and MSS timestamps shift — no amount of replay logic fixes wrong input bars.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

UTC = pytz.UTC
IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

# Map timeframe strings to canonical folder/file names
TIMEFRAME_MAP = {
    "4h": "4h",
    "1h": "1h",
    "15m": "15m",
    "5m": "5m",
}

# Auto-detect market from symbol name
def detect_market(symbol: str) -> str:
    """Return 'CRYPTO' or 'FOREX' based on the symbol name."""
    sym = symbol.upper()
    if sym.endswith("USDT") or sym.endswith("USD") and len(sym) > 6:
        return "CRYPTO"
    return "FOREX"


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

def _find_csv(symbol: str, timeframe: str, data_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate the CSV file for a given symbol and timeframe.

    Tries several naming conventions:
      1. <data_dir>/<SYMBOL>/<tf>.csv          (canonical)
      2. <data_dir>/<SYMBOL>/<SYMBOL>_<tf>*.csv  (Drive export naming)
    """
    base = (data_dir or DATA_DIR) / symbol.upper()
    if not base.exists():
        return None

    tf = TIMEFRAME_MAP.get(timeframe, timeframe)

    # Try canonical name first
    canonical = base / f"{tf}.csv"
    if canonical.exists():
        return canonical

    # Try Drive export pattern: USDJPY_4h_2021-03-11_2026-04-22.csv
    for f in sorted(base.glob(f"{symbol.upper()}_{tf}_*.csv")):
        return f  # return first match

    # Fallback: any csv containing the timeframe
    for f in sorted(base.glob(f"*_{tf}*.csv")):
        return f

    return None


def load_ohlcv(
    symbol: str,
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    data_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load OHLCV data for *symbol* / *timeframe* from CSV.

    Returns a DataFrame indexed by UTC datetime with columns:
        open, high, low, close, volume

    Returns None if no data file is found.
    """
    csv_path = _find_csv(symbol, timeframe, data_dir)
    if csv_path is None:
        logger.warning(f"No CSV found for {symbol}/{timeframe} in {data_dir or DATA_DIR}")
        return None

    logger.info(f"Loading {symbol}/{timeframe} from {csv_path.name}")

    df = pd.read_csv(csv_path)

    # ── Parse datetime ────────────────────────────────────────────
    if "time_utc" in df.columns:
        df["datetime"] = pd.to_datetime(df["time_utc"], utc=True)
    elif "time" in df.columns:
        # Unix timestamp fallback
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        raise ValueError(
            f"Cannot find datetime column in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )

    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # ── Rename volume column ──────────────────────────────────────
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
    if "volume" not in df.columns:
        df["volume"] = 0

    # ── Keep only OHLCV ───────────────────────────────────────────
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    # ── Date range filter ─────────────────────────────────────────
    if start is not None:
        if start.tzinfo is None:
            start = UTC.localize(start)
        df = df[df.index >= start]
    if end is not None:
        if end.tzinfo is None:
            end = UTC.localize(end)
        df = df[df.index <= end]

    if df.empty:
        logger.warning(f"No data for {symbol}/{timeframe} in range {start}–{end}")
        return None

    logger.info(
        f"  Loaded {len(df)} bars  "
        f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})"
    )
    return df


def resample_htf_from_5m(
    df_5m: pd.DataFrame,
    htf_hours: int,
    server_utc_offset: int,
) -> pd.DataFrame:
    """Rebuild higher-timeframe OHLCV from 5m bars using the MT5 server offset.

    MT5 4H candles open at ``00:00, 04:00, 08:00, …`` **server local time**.
    In UTC that becomes ``(24 - offset) % 24, (28 - offset) % 24, …``.
    For ``server_utc_offset=3`` and ``htf_hours=4``, the 4H UTC grid is
    ``01, 05, 09, 13, 17, 21``.

    Parameters
    ----------
    df_5m : DataFrame with UTC-aware DatetimeIndex and OHLCV columns.
    htf_hours : target bar size in hours (typically 4).
    server_utc_offset : MT5 server hours ahead of UTC (e.g. 2 or 3).

    Returns
    -------
    DataFrame with the same OHLCV columns, indexed by bar-open in UTC.
    """
    pandas_offset = f"{(24 - server_utc_offset) % htf_hours}h"
    freq = f"{htf_hours}h"

    vol_col = "volume" if "volume" in df_5m.columns else "tick_volume"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if vol_col in df_5m.columns:
        agg[vol_col] = "sum"

    resampled = df_5m.resample(freq, offset=pandas_offset).agg(agg)
    resampled.dropna(subset=["open"], inplace=True)

    if vol_col != "volume" and vol_col in resampled.columns:
        resampled.rename(columns={vol_col: "volume"}, inplace=True)
    if "volume" not in resampled.columns:
        resampled["volume"] = 0

    return resampled


def load_all_timeframes(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    data_dir: Optional[Path] = None,
    server_utc_offset: Optional[int] = None,
) -> dict[str, Optional[pd.DataFrame]]:
    """Load 4h, 1h, 15m, 5m data for a symbol.

    Parameters
    ----------
    server_utc_offset : int, optional
        MT5 server UTC offset (e.g. 3 for Exness summer, 2 for winter).
        When set, **4H bars are rebuilt from 5m** so their open timestamps
        match the live terminal's grid (01/05/09/13/17/21 UTC for offset=3).
        1H/15m/5m are kept from CSV (1H alignment is hour-boundary regardless
        of server offset).

    Returns dict:  {"4h": df, "1h": df, "15m": df, "5m": df}
    """
    result = {}
    for tf in ("4h", "1h", "15m", "5m"):
        # For higher timeframes, load extra history for EMA warm-up
        tf_start = start
        if start is not None:
            if tf == "4h":
                tf_start = start - pd.Timedelta(days=60)
            elif tf == "1h":
                tf_start = start - pd.Timedelta(days=30)
            elif tf == "15m":
                tf_start = start - pd.Timedelta(days=14)
            elif tf == "5m":
                if server_utc_offset is not None:
                    tf_start = start - pd.Timedelta(days=60)
                else:
                    tf_start = start - pd.Timedelta(days=7)
        result[tf] = load_ohlcv(symbol, tf, tf_start, end, data_dir)

    if server_utc_offset is not None and result["5m"] is not None:
        result["4h"] = resample_htf_from_5m(
            result["5m"], 4, server_utc_offset,
        )
        logger.info(
            f"  Resampled 4H from 5m with server UTC+{server_utc_offset} "
            f"→ {len(result['4h'])} bars"
        )

        if result["1h"] is None:
            result["1h"] = resample_htf_from_5m(
                result["5m"], 1, server_utc_offset,
            )
            logger.info(
                f"  Resampled 1H from 5m (CSV missing) → {len(result['1h'])} bars"
            )

    return result


# ---------------------------------------------------------------------------
# Utility: list available symbols
# ---------------------------------------------------------------------------

def list_available_symbols(data_dir: Optional[Path] = None) -> list[str]:
    """Return list of symbol names that have data directories.

    Filters out internal directories (starting with _ or .) and
    directories that don't look like valid trading symbols.
    """
    base = data_dir or DATA_DIR
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and not d.name.startswith("_")
        and d.name.isalpha()
        and d.name == d.name.upper()
    )
