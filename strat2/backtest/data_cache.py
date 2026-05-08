from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

BACKTEST_DIR = Path(__file__).resolve().parent
STRAT2_DIR = BACKTEST_DIR.parent
REPO_ROOT = STRAT2_DIR.parent
DATA_DIR = BACKTEST_DIR / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(STRAT2_DIR / ".env")
except Exception:
    pass

try:
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MetaTrader5 is required: pip install MetaTrader5") from exc


TIMEFRAMES: Dict[str, int] = {
    "D1": mt5.TIMEFRAME_D1,
    "H4": mt5.TIMEFRAME_H4,
    "H1": mt5.TIMEFRAME_H1,
    "M15": mt5.TIMEFRAME_M15,
    "M5": mt5.TIMEFRAME_M5,
}

TIMEFRAME_MINUTES: Dict[str, int] = {
    "D1": 1440,
    "H4": 240,
    "H1": 60,
    "M15": 15,
    "M5": 5,
}


@dataclass
class CacheInfo:
    symbol: str
    mt5_symbol: str
    timeframe: str
    path: Path
    source: str
    rows: int
    start_utc: Optional[str]
    end_utc: Optional[str]


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clean_symbol(symbol: str) -> str:
    value = (symbol or "").upper().strip()
    for suffix in (".M", "M"):
        if value.endswith(suffix) and len(value) > 6:
            return value[: -len(suffix)]
    return value


def cache_path(symbol: str, timeframe: str) -> Path:
    base = clean_symbol(symbol)
    return DATA_DIR / base / f"{base}_{timeframe}.csv"


class MT5HistoryCache:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.connected = False
        self.symbol_cache: Dict[str, str] = {}

    def connect(self) -> bool:
        if self.connected and mt5.account_info() is not None:
            return True
        if not mt5.initialize():
            return False
        login = (os.getenv("MT5_LOGIN") or os.getenv("XM_MT5_LOGIN") or "").strip()
        password = (os.getenv("MT5_PASSWORD") or os.getenv("XM_MT5_PASSWORD") or "").strip()
        server = (os.getenv("MT5_SERVER") or os.getenv("XM_MT5_SERVER") or "").strip()
        if login and password:
            if not mt5.login(int(login), password=password, server=server or None):
                return False
        self.connected = mt5.account_info() is not None
        return self.connected

    def shutdown(self) -> None:
        try:
            mt5.shutdown()
        finally:
            self.connected = False

    def resolve_symbol(self, requested: str) -> Optional[str]:
        key = clean_symbol(requested)
        if key in self.symbol_cache:
            return self.symbol_cache[key]
        candidates = [requested, key, f"{key}m", f"{key}.m"]
        for candidate in candidates:
            if candidate and mt5.symbol_select(candidate, True):
                self.symbol_cache[key] = candidate
                return candidate
        for item in mt5.symbols_get(f"*{key}*") or []:
            name = getattr(item, "name", "")
            if name and mt5.symbol_select(name, True):
                self.symbol_cache[key] = name
                return name
        return None

    def symbol_info(self, symbol: str):
        mt5_symbol = self.resolve_symbol(symbol)
        if not mt5_symbol:
            return None
        return mt5.symbol_info(mt5_symbol)

    def read_cached(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = cache_path(symbol, timeframe)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if df.empty or "datetime" not in df.columns:
            return None
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        return df.sort_index()

    def write_cached(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        path = cache_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy().sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out.insert(0, "datetime", out.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
        out.to_csv(path, index=False)
        return path

    def fetch_mt5(self, mt5_symbol: str, timeframe: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        start_utc = ensure_utc(start_utc)
        end_utc = ensure_utc(end_utc)
        rates = mt5.copy_rates_range(mt5_symbol, TIMEFRAMES[timeframe], start_utc, end_utc)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0
        if "real_volume" not in df.columns:
            df["real_volume"] = 0
        if "spread" not in df.columns:
            df["spread"] = 0
        df["volume"] = df["tick_volume"]
        return df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume", "volume"]].sort_index()

    def ensure_history(
        self,
        symbol: str,
        timeframe: str,
        start_utc: datetime,
        end_utc: datetime,
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, CacheInfo]:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        start_utc = ensure_utc(start_utc)
        end_utc = ensure_utc(end_utc)
        path = cache_path(symbol, timeframe)
        cached = None if force_refresh else self.read_cached(symbol, timeframe)
        source = "offline"

        covers_range = False
        if cached is not None and not cached.empty:
            covers_range = cached.index.min() <= start_utc and cached.index.max() >= end_utc - timedelta(minutes=TIMEFRAME_MINUTES[timeframe])

        if cached is None or cached.empty or not covers_range:
            if not self.connect():
                if cached is None or cached.empty:
                    raise RuntimeError(f"MT5 is unavailable and no offline cache exists for {symbol} {timeframe}")
            mt5_symbol = self.resolve_symbol(symbol)
            if not mt5_symbol:
                raise RuntimeError(f"Could not resolve {symbol} in MT5")
            fetched = self.fetch_mt5(mt5_symbol, timeframe, start_utc, end_utc)
            if fetched.empty and (cached is None or cached.empty):
                raise RuntimeError(f"No MT5 data returned for {symbol} {timeframe}")
            merged = fetched if cached is None or cached.empty else pd.concat([cached, fetched]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            self.write_cached(symbol, timeframe, merged)
            cached = merged
            source = "mt5"

        mt5_symbol = self.symbol_cache.get(clean_symbol(symbol), clean_symbol(symbol))
        sliced = cached[(cached.index >= start_utc) & (cached.index <= end_utc)].copy()
        info = CacheInfo(
            symbol=clean_symbol(symbol),
            mt5_symbol=mt5_symbol,
            timeframe=timeframe,
            path=path,
            source=source,
            rows=len(sliced),
            start_utc=None if sliced.empty else sliced.index.min().strftime("%Y-%m-%d %H:%M:%S UTC"),
            end_utc=None if sliced.empty else sliced.index.max().strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        return sliced, info

