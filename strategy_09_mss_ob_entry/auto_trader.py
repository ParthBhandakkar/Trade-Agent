"""
Strategy 09 — MSS + Order Block: AUTO TRADER (Crypto + Forex)
=============================================================

Scans for fresh Strategy 09 signals and **automatically executes trades**:
  • Crypto → Binance Futures (Testnet by default)
  • Forex  → MT5 (Exness when MT5_* credentials are configured)

Stop-Loss Logic
~~~~~~~~~~~~~~~
1. Start with the Order Block extreme:
     - Bearish trade → SL = OB.top  (high of the OB candle)
     - Bullish trade → SL = OB.bottom (low of the OB candle)
2. Scan 5M candles between OB time and tap candle for a worse extreme:
     - Bearish: if any 5M candle between OB and tap has a higher high,
       use that as SL instead.
     - Bullish: if any 5M candle between OB and tap has a lower low,
       use that as SL instead.
3. Add spread buffer:
     - Crypto: +/- 0.05% of price
     - Forex:  +/- 15 pips (10 pips for JPY pairs → 0.15 / 0.015)

Take-Profit: Single TP at 1.5R risk-reward.

Credentials
~~~~~~~~~~~
  • Binance Testnet keys: from project root ``.env``
  • MT5 / Exness: from project root ``.env`` (or the already logged-in terminal)
  • Email config: from project root ``.env``

Run
---
    .venv\\Scripts\\python.exe strategy_09_mss_ob_entry\\auto_trader.py
    .venv\\Scripts\\python.exe strategy_09_mss_ob_entry\\auto_trader.py --dry-run
    .venv\\Scripts\\python.exe strategy_09_mss_ob_entry\\auto_trader.py --crypto-only
    .venv\\Scripts\\python.exe strategy_09_mss_ob_entry\\auto_trader.py --forex-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Load credentials — the repository root .env
from scripts.utils.env_loader import load_root_env               # noqa: E402

load_root_env(REPO_ROOT)

from strategy import (                                        # noqa: E402
    MSSOrderBlockStrategy, MSSOB_Signal, DailyBias,
    MSSConfirmation, OBEntrySetup, BiasType,
    format_signal_for_jsonl,
)
from email_notifier import EmailNotifier                       # noqa: E402
from phase_logger import PhaseLogger                           # noqa: E402
from scripts.utils.indicators import Direction, OrderBlock     # noqa: E402
from entry_filters import check_killzone, FilterResult, calculate_ema  # noqa: E402

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

FRESH_WINDOW_MINUTES = 15         # Tap must be ≤15 min old

# Position sizing
CRYPTO_TRADE_USDT = 10.0          # USDT margin per crypto trade
FOREX_MARGIN_INR  = 1000          # ₹1000 margin per forex trade
INR_USD_RATE      = 84.0           # Approximate INR per USD (for lot calc)

# Hard loss cap — if estimated SL loss > this, tighten SL at open time
HARD_LOSS_CAP_ENABLED = True
HARD_LOSS_CAP_AMOUNT  = 3000      # ₹3000 max loss per trade
HARD_LOSS_CAP_COMMISSION_BUFFER = 500  # ₹500 extra buffer for commission

# London Open manipulation block (07:00-08:00 UTC)
# Optimization across 90 combos: 07:00-08:00 is the sweet spot.
# Tighter than 08:30 — saves the 08:17-08:27 winners while still
# blocking the 07:xx losers.
LONDON_OPEN_BLOCK_START_UTC = 7 * 60       # 07:00 UTC in minutes
LONDON_OPEN_BLOCK_END_UTC   = 8 * 60       # 08:00 UTC in minutes

# SELL filter: DISABLED after real-data analysis proved it was curve-fitted.
# The OB strategy enters on pullbacks, so 1H EMA alignment is inherently
# violated at entry time.  See generic_filter_analysis.md for details.

# Spread buffer for SL
CRYPTO_SL_BUFFER_PCT = 0.35       # 0.35% buffer for crypto (wick room)
FOREX_SL_BUFFER_PIPS = 15         # 15 pips buffer for forex (10 for JPY)

# Max concurrent positions
MAX_CRYPTO_POSITIONS = 5
MAX_FOREX_POSITIONS = 5

# MT5 resilience
MT5_RECONNECT_RETRIES = 3
MT5_RETRY_DELAY_SEC = 2.0
MT5_KEEPALIVE_INTERVAL_SEC = 30

# Daily loss limit (USDT) — stop opening new crypto trades after this
CRYPTO_DAILY_LOSS_LIMIT = 15.0

# Max hold time (seconds) — force-close crypto positions held longer than this
MAX_HOLD_SECONDS = 24 * 3600   # 24 hours

# Trailing SL — dynamic from the margin used on each trade.
# First trigger at 1.2× margin profit locks 1.0× margin.
# After that, every additional 0.5× margin profit locks another 0.5× margin.
# Override the base step with --trailing-sl-step.
TRAILING_SL_STEP_INR = 0        # 0 = auto (use margin as step)

# Crypto symbol blacklist — consistently losing symbols
CRYPTO_SYMBOL_BLACKLIST = {
    "AVAXUSDT", "LTCUSDT", "BTCUSDT", "ETCUSDT",
    "XLMUSDT", "NEARUSDT", "XRPUSDT", "ARBUSDT", "LINKUSDT",
}

# Magic number for MT5 (identifies our trades)
MT5_MAGIC = 909009

# Binance pair configs (price & qty precision)
# NOTE: Blacklisted symbols kept here for precision lookups (e.g. force-close)
#       but filtered out at signal time by CRYPTO_SYMBOL_BLACKLIST.
BINANCE_PAIR_CONFIG = {
    "BTCUSDT":   {"pp": 2, "qp": 3, "mq": 0.001},
    "ETHUSDT":   {"pp": 2, "qp": 3, "mq": 0.001},
    "BNBUSDT":   {"pp": 2, "qp": 2, "mq": 0.01},
    "SOLUSDT":   {"pp": 3, "qp": 0, "mq": 1},
    "XRPUSDT":   {"pp": 4, "qp": 1, "mq": 0.1},
    "ADAUSDT":   {"pp": 5, "qp": 0, "mq": 1},
    "DOGEUSDT":  {"pp": 5, "qp": 0, "mq": 1},
    "AVAXUSDT":  {"pp": 3, "qp": 1, "mq": 0.1},
    "DOTUSDT":   {"pp": 3, "qp": 1, "mq": 0.1},
    "LINKUSDT":  {"pp": 3, "qp": 1, "mq": 0.1},
    "LTCUSDT":   {"pp": 2, "qp": 2, "mq": 0.01},
    "ATOMUSDT":  {"pp": 3, "qp": 1, "mq": 0.1},
    "UNIUSDT":   {"pp": 3, "qp": 1, "mq": 0.1},
    "ETCUSDT":   {"pp": 3, "qp": 1, "mq": 0.1},
    "XLMUSDT":   {"pp": 5, "qp": 0, "mq": 1},
    "NEARUSDT":  {"pp": 3, "qp": 0, "mq": 1},
    "AAVEUSDT":  {"pp": 2, "qp": 2, "mq": 0.01},
    "FILUSDT":   {"pp": 3, "qp": 1, "mq": 0.1},
    "APTUSDT":   {"pp": 3, "qp": 1, "mq": 0.1},
    "INJUSDT":   {"pp": 3, "qp": 1, "mq": 0.1},
    "OPUSDT":    {"pp": 4, "qp": 0, "mq": 1},
}

CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT", "XLMUSDT",
    "NEARUSDT", "AAVEUSDT", "FILUSDT", "APTUSDT",
    "INJUSDT", "OPUSDT",
]

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDJPY", "NZDCAD", "NZDCHF",
    "CADJPY", "CADCHF",
    "CHFJPY",
    "XAUUSD",
]


# ============================================================================
# FOREX SESSION FILTER
# ============================================================================

def _is_forex_session(now_ist: datetime) -> bool:
    wd = now_ist.weekday()
    t = now_ist.hour * 60 + now_ist.minute
    if wd == 6:
        return False
    if wd == 5:
        return t < 90
    if wd == 0:
        return t >= 90
    return True


# ============================================================================
# LONDON OPEN MANIPULATION BLOCK
# ============================================================================

def _is_london_open_block(now_utc: datetime) -> bool:
    """Block entries during London Open first hour (07:00-08:00 UTC).

    Real-data optimization across 90 filter combos: 07:00-08:00 is the
    sweet spot.  Banks sweep liquidity in the first hour, faking out
    OB entries.  After 08:00, the real directional move starts.
    """
    t = now_utc.hour * 60 + now_utc.minute
    return LONDON_OPEN_BLOCK_START_UTC <= t <= LONDON_OPEN_BLOCK_END_UTC


# ============================================================================
# CURRENCY CORRELATION GUARD
# ============================================================================

def _get_exposed_currencies(
    mt5_exec: Optional["MT5Executor"],
    binance: Optional["BinanceExecutor"],
) -> Set[str]:
    """Return set of currencies already exposed via open positions.

    Prevents correlated-pair blowups like the Apr 7 EURUSD + GBPUSD
    disaster where 4 positions on 2 correlated pairs lost ₹8,491 in 6 min.
    """
    currencies: Set[str] = set()

    if mt5_exec and mt5_exec.connected:
        for pos in mt5_exec.get_open_positions():
            sym = pos.symbol.upper().replace("M", "")
            if len(sym) >= 6:
                currencies.add(sym[:3])
                currencies.add(sym[3:6])

    if binance and binance.connected:
        for pos in binance.get_open_positions():
            sym = pos['symbol'].upper()
            # Crypto pairs are like ETHUSDT
            if sym.endswith("USDT"):
                currencies.add(sym[:-4])
                currencies.add("USDT")

    return currencies


def _check_correlation_guard(
    symbol: str, market: str, exposed: Set[str],
) -> Tuple[bool, str]:
    """Return (passed, reason).  Fails if base or quote is already exposed."""
    sym = symbol.upper().replace("M", "")
    if market == "CRYPTO":
        if sym.endswith("USDT"):
            base = sym[:-4]
        else:
            base = sym[:3]
        if base in exposed:
            return False, f"{base} already exposed in another position"
    else:
        if len(sym) >= 6:
            base, quote = sym[:3], sym[3:6]
            if base in exposed:
                return False, f"{base} already exposed in another position"
            if quote in exposed:
                return False, f"{quote} already exposed in another position"
    return True, "No currency overlap"


# ============================================================================
# SELL DIRECTION EXTRA CONFIRMATION — DEPRECATED
# ============================================================================
# This filter was removed after real-data analysis proved it was curve-fitted.
# The OB strategy enters on pullbacks into Order Blocks — by definition, at
# entry time the 1H close is on the "wrong" side of EMA21.  Any 1H EMA filter
# fights the core strategy logic.  Tested on 42 trades: it blocked 24 of 28
# winners.  See results/generic_filter_analysis.md for full details.
# The function is preserved below for reference only (not called anywhere).


# ============================================================================
# BIAS TRACKING
# ============================================================================

@dataclass
class BiasTrack:
    key: str
    bias: DailyBias
    market: str
    created_utc: datetime
    expires_utc: datetime
    mss: Optional[MSSConfirmation] = None
    ob_entry: Optional[OBEntrySetup] = None
    alerted: bool = False
    completed: bool = False
    last_rejected_tap: Optional[datetime] = None  # skip taps at/before this time


def _bias_key(bias: DailyBias) -> str:
    ts = bias.sweep_timestamp
    return f"{bias.direction.value}_{ts.isoformat() if ts is not None else 'none'}"


# ============================================================================
# CYCLE SCHEDULING
# ============================================================================

def _floor_5m(dt: datetime) -> datetime:
    dt0 = dt.replace(second=0, microsecond=0)
    return dt0.replace(minute=dt0.minute - (dt0.minute % 5))


def _due_5m_cycle(
    now_ist: datetime, poll_seconds: int, last_key: Optional[datetime],
) -> Optional[datetime]:
    trigger = _floor_5m(now_ist - timedelta(minutes=1)) + timedelta(minutes=1)
    if last_key is not None and trigger <= last_key:
        return None
    lag = (now_ist - trigger).total_seconds()
    if 0 <= lag <= max(90, poll_seconds + 5):
        return trigger
    return None


def _is_due_4h(c: datetime) -> bool:
    return c.minute == 31 and (c.hour % 4) == 1

def _is_due_1h(c: datetime) -> bool:
    return c.minute == 31

def _is_due_15m(c: datetime) -> bool:
    return c.minute in (1, 16, 31, 46)


# ============================================================================
# HELPERS
# ============================================================================

def _drop_incomplete(
    df: pd.DataFrame, interval_min: int, now_utc: datetime, safety_s: int = 60,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    if out.index.tz is None:
        out = out.copy()
        out.index = out.index.tz_localize(UTC)
    cutoff = now_utc - timedelta(seconds=safety_s)
    while len(out) > 0:
        last_close = out.index[-1] + timedelta(minutes=interval_min)
        if last_close <= cutoff:
            break
        out = out.iloc[:-1]
    return out


def _signal_id(sig_json: Dict[str, Any]) -> str:
    return "|".join([
        str(sig_json.get("symbol", "")),
        str(sig_json.get("direction", "")),
        str(sig_json.get("signal_datetime_ist", "")),
    ])


def _is_fresh(tap_time, now_utc: datetime, window_min: int) -> bool:
    if tap_time is None:
        return False
    tt = tap_time
    if hasattr(tt, 'to_pydatetime'):
        tt = tt.to_pydatetime()
    if tt.tzinfo is None:
        tt = UTC.localize(tt)
    age = (now_utc - tt).total_seconds() / 60.0
    return age <= window_min


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _is_jpy_pair(symbol: str) -> bool:
    """Check if symbol is a JPY cross (pips = 0.01 instead of 0.0001)."""
    return "JPY" in symbol.upper()


def _get_mt5_credentials() -> Tuple[int, str, str]:
    """Prefer generic Exness MT5 credentials, fallback to legacy XM keys."""
    login_raw = (
        os.getenv("MT5_LOGIN", "").strip()
        or os.getenv("XM_MT5_LOGIN", "0").strip()
    )
    password = (
        os.getenv("MT5_PASSWORD", "").strip()
        or os.getenv("XM_MT5_PASSWORD", "").strip()
    )
    server = (
        os.getenv("MT5_SERVER", "").strip()
        or os.getenv("XM_MT5_SERVER", "XMGlobal-MT5 3").strip()
    )
    try:
        login = int(login_raw) if login_raw else 0
    except ValueError:
        login = 0
    return login, password, server


# ============================================================================
# MT5 MARKET DATA FETCHER
# ============================================================================

class MT5DataFetcher:
    """Fetch OHLCV candles directly from the logged-in MT5 terminal."""

    _INTERVALS = {
        "1m": "TIMEFRAME_M1",
        "3m": "TIMEFRAME_M3",
        "5m": "TIMEFRAME_M5",
        "15m": "TIMEFRAME_M15",
        "1h": "TIMEFRAME_H1",
        "4h": "TIMEFRAME_H4",
    }

    def __init__(self, market: str = "FOREX"):
        self.market = market
        self.mt5 = None
        self.connected = False
        self.login, self.password, self.server = _get_mt5_credentials()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._symbol_cache: Dict[str, str] = {}
        self.connect()

    def connect(self) -> bool:
        self.connected = False
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            if not mt5.initialize():
                logger.error(f"MT5 data init failed: {mt5.last_error()}")
                return False
            if self.login and self.password:
                if not mt5.login(login=self.login, password=self.password, server=self.server):
                    logger.error(f"MT5 data login failed: {mt5.last_error()}")
                    return False
            info = mt5.account_info()
            if info is None:
                logger.error(f"MT5 data account unavailable: {mt5.last_error()}")
                return False
            self.connected = True
            logger.info(
                f"MT5 data source connected: {info.server} | "
                f"{'Demo' if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else 'LIVE'}"
            )
            return True
        except ImportError:
            logger.error("MetaTrader5 library not installed - MT5 data disabled")
            return False
        except Exception as e:
            logger.error(f"MT5 data connect failed: {e}")
            return False

    def ensure_connected(self) -> bool:
        if self.connected and self.mt5 is not None:
            try:
                if self.mt5.account_info() is not None:
                    return True
            except Exception:
                pass
        return self.connect()

    def _candidate_symbols(self, symbol: str) -> List[str]:
        sym = symbol.upper()
        candidates = [sym]
        if sym.endswith("USDT"):
            candidates.append(sym[:-4] + "USD")
        candidates.extend([f"{c}m" for c in list(candidates)])
        candidates.extend([f"{c}.m" for c in list(candidates) if not c.endswith(".m")])
        return list(dict.fromkeys(candidates))

    def _resolve_symbol(self, symbol: str) -> Optional[str]:
        key = symbol.upper()
        if key in self._symbol_cache:
            return self._symbol_cache[key]
        mt5 = self.mt5
        if mt5 is None:
            return None

        for candidate in self._candidate_symbols(symbol):
            if mt5.symbol_select(candidate, True):
                self._symbol_cache[key] = candidate
                return candidate

        search_key = key[:-4] + "USD" if key.endswith("USDT") else key
        matches = mt5.symbols_get(f"*{search_key}*") or []
        for item in matches:
            name = getattr(item, "name", "")
            if name and mt5.symbol_select(name, True):
                self._symbol_cache[key] = name
                return name

        logger.warning(f"MT5 symbol not found for {symbol}")
        return None

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        n_bars: int = 5000,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{interval}_{n_bars}"
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]
        if not self.ensure_connected() or self.mt5 is None:
            return None

        tf_attr = self._INTERVALS.get(interval.lower())
        if tf_attr is None:
            logger.error(f"Unknown interval: {interval}")
            return None
        timeframe = getattr(self.mt5, tf_attr, None)
        if timeframe is None:
            logger.error(f"MT5 timeframe unavailable: {tf_attr}")
            return None

        mt5_symbol = self._resolve_symbol(symbol)
        if mt5_symbol is None:
            return None

        try:
            bars = max(1, int(n_bars))
            logger.info(f"    {interval} ({bars} bars, MT5 {mt5_symbol})...")
            rates = self.mt5.copy_rates_from_pos(mt5_symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                logger.warning(f"No MT5 rates for {symbol} {interval}: {self.mt5.last_error()}")
                return None

            df = pd.DataFrame(rates)
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("datetime", inplace=True)
            volume_col = "tick_volume" if "tick_volume" in df.columns else "real_volume"
            df["volume"] = df[volume_col] if volume_col in df.columns else 0
            df = df[["open", "high", "low", "close", "volume"]].sort_index()
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"MT5 fetch error {symbol} {interval}: {e}")
            return None

    def clear_cache(self):
        self._cache.clear()


# ============================================================================
# STRICT 4H EMA TREND FILTER
# ============================================================================

_MIN_4H_BARS_FOR_EMA = 50  # Minimum closed 4H candles required


def _check_strict_ema(
    df_4h: pd.DataFrame,
    direction: str,
) -> Tuple[bool, str]:
    """Strict 4H EMA alignment check evaluated at tap time.

    Bullish:  4H close > EMA21 > EMA50
    Bearish:  4H close < EMA21 < EMA50

    Uses only closed 4H candles (caller must already drop incomplete).
    Requires at least 50 closed 4H bars.

    Returns:
        (passed: bool, detail: str)
    """
    if df_4h is None or df_4h.empty:
        return False, "No 4H data available — skipping trade"

    if len(df_4h) < _MIN_4H_BARS_FOR_EMA:
        return False, (
            f"Insufficient 4H history ({len(df_4h)} bars < "
            f"{_MIN_4H_BARS_FOR_EMA} required) — skipping trade"
        )

    close = df_4h["close"]
    ema21 = calculate_ema(close, 21)
    ema50 = calculate_ema(close, 50)

    last_close = float(close.iloc[-1])
    last_ema21 = float(ema21.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])

    is_bull = direction.lower() in ("bullish", "long")

    if is_bull:
        passed = last_close > last_ema21 > last_ema50
        if passed:
            detail = (
                f"ALIGNED: Close({last_close:.5f}) > "
                f"EMA21({last_ema21:.5f}) > EMA50({last_ema50:.5f})"
            )
        else:
            # Diagnose why it failed
            if last_ema21 <= last_ema50:
                detail = (
                    f"COUNTER-TREND: Bullish but EMA21({last_ema21:.5f}) "
                    f"<= EMA50({last_ema50:.5f}) — trend is bearish"
                )
            else:
                detail = (
                    f"WEAK ALIGNMENT: EMA21({last_ema21:.5f}) > "
                    f"EMA50({last_ema50:.5f}) but Close({last_close:.5f}) "
                    f"< EMA21 — price not confirming trend"
                )
    else:
        passed = last_close < last_ema21 < last_ema50
        if passed:
            detail = (
                f"ALIGNED: Close({last_close:.5f}) < "
                f"EMA21({last_ema21:.5f}) < EMA50({last_ema50:.5f})"
            )
        else:
            if last_ema21 >= last_ema50:
                detail = (
                    f"COUNTER-TREND: Bearish but EMA21({last_ema21:.5f}) "
                    f">= EMA50({last_ema50:.5f}) — trend is bullish"
                )
            else:
                detail = (
                    f"WEAK ALIGNMENT: EMA21({last_ema21:.5f}) < "
                    f"EMA50({last_ema50:.5f}) but Close({last_close:.5f}) "
                    f"> EMA21 — price not confirming trend"
                )

    return passed, detail


# ============================================================================
# SMART STOP LOSS CALCULATION
# ============================================================================

def compute_smart_sl(
    ob: OrderBlock,
    bias_direction: BiasType,
    df_5m: pd.DataFrame,
    tap_time: pd.Timestamp,
    entry_price: float,
    market: str,
    symbol: str,
) -> Tuple[float, str]:
    """
    Compute the smart stop-loss:

    1. Base SL = OB extreme (top for bearish, bottom for bullish).
    2. Scan 5M candles between OB time and tap candle:
       - Bearish: find highest high → use if higher than OB top.
       - Bullish: find lowest low → use if lower than OB bottom.
    3. Add spread buffer.
    4. Return (sl_price, reason_string).
    """
    # ── Step 1: Base SL at OB extreme ─────────────────────────────
    if bias_direction == BiasType.BEARISH:
        base_sl = ob.top
        reason = f"OB high ({base_sl:.6f})"
    else:
        base_sl = ob.bottom
        reason = f"OB low ({base_sl:.6f})"

    # ── Step 2: Scan for worse extreme between OB and tap ─────────
    ob_time = ob.datetime
    if hasattr(ob_time, 'to_pydatetime'):
        ob_time = ob_time.to_pydatetime()
    if ob_time.tzinfo is None:
        ob_time = UTC.localize(ob_time)

    tt = tap_time
    if hasattr(tt, 'to_pydatetime'):
        tt = tt.to_pydatetime()
    if tt.tzinfo is None:
        tt = UTC.localize(tt)

    # Filter 5M data between OB candle and tap candle (inclusive)
    if df_5m is not None and not df_5m.empty:
        mask = (df_5m.index >= ob_time) & (df_5m.index <= tt)
        segment = df_5m.loc[mask]

        if len(segment) > 0:
            if bias_direction == BiasType.BEARISH:
                swing_high = segment['high'].max()
                if swing_high > base_sl:
                    reason = (
                        f"Swing high ({swing_high:.6f}) between OB and tap "
                        f"(was OB high {base_sl:.6f})"
                    )
                    base_sl = swing_high
            else:
                swing_low = segment['low'].min()
                if swing_low < base_sl:
                    reason = (
                        f"Swing low ({swing_low:.6f}) between OB and tap "
                        f"(was OB low {base_sl:.6f})"
                    )
                    base_sl = swing_low

    # ── Step 3: Add spread buffer ─────────────────────────────────
    if market == "CRYPTO":
        buffer = base_sl * (CRYPTO_SL_BUFFER_PCT / 100.0)
    else:
        # Forex: pips
        if _is_jpy_pair(symbol):
            buffer = FOREX_SL_BUFFER_PIPS * 0.01   # 1 pip = 0.01 for JPY
        else:
            buffer = FOREX_SL_BUFFER_PIPS * 0.0001  # 1 pip = 0.0001

    if bias_direction == BiasType.BEARISH:
        sl = base_sl + buffer
    else:
        sl = base_sl - buffer

    reason += f" + {FOREX_SL_BUFFER_PIPS}pip buffer" if market == "FOREX" else f" + {CRYPTO_SL_BUFFER_PCT}% buffer"

    return sl, reason


def compute_tp(entry_price: float, sl_price: float, rr: float = 1.5) -> float:
    """TP at *rr* × risk from entry (default 1.5R)."""
    risk = abs(entry_price - sl_price)
    if entry_price > sl_price:
        return entry_price + rr * risk
    else:
        return entry_price - rr * risk


# ============================================================================
# BINANCE EXECUTOR
# ============================================================================

class BinanceExecutor:
    """Thin wrapper around python-binance for Futures testnet."""

    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        if testnet:
            self.api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        else:
            self.api_key = os.getenv("BINANCE_API_KEY", "")
            self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.client = None
        self.connected = False
        self._leverage_cache: Dict[str, int] = {}  # symbol → max leverage

    def connect(self) -> bool:
        try:
            from binance.client import Client
            self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
            self.client.futures_account()
            self.connected = True
            logger.info("✅ Binance Futures connected (Testnet)" if self.testnet else "✅ Binance LIVE")
            return True
        except Exception as e:
            logger.error(f"Binance connect failed: {e}")
            return False

    def get_open_position_count(self) -> int:
        if not self.connected:
            return 0
        try:
            positions = self.client.futures_position_information()
            return sum(1 for p in positions if float(p['positionAmt']) != 0)
        except Exception:
            return 0

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return list of open positions with symbol, side, qty, entry time."""
        if not self.connected:
            return []
        try:
            positions = self.client.futures_position_information()
            result = []
            for p in positions:
                amt = float(p.get('positionAmt', 0))
                if amt == 0:
                    continue
                result.append({
                    'symbol': p['symbol'],
                    'positionAmt': amt,
                    'entryPrice': float(p.get('entryPrice', 0)),
                    'unRealizedProfit': float(p.get('unRealizedProfit', 0)),
                    'updateTime': int(p.get('updateTime', 0)),
                })
            return result
        except Exception as e:
            logger.debug(f"get_open_positions error: {e}")
            return []

    def force_close_position(self, symbol: str, position_amt: float) -> bool:
        """Market-close an open position. Returns True on success."""
        from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
        try:
            # If positionAmt > 0 it's a long → sell to close.
            # If positionAmt < 0 it's a short → buy to close.
            close_side = SIDE_SELL if position_amt > 0 else SIDE_BUY
            qty = abs(position_amt)
            self.client.futures_create_order(
                symbol=symbol, side=close_side,
                type=ORDER_TYPE_MARKET, quantity=qty,
                reduceOnly=True,
            )
            logger.info(f"  🔒 Force-closed {symbol} (qty={qty})")
            # Cancel lingering SL/TP orders
            try:
                self.client.futures_cancel_all_open_orders(symbol=symbol)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"  Force-close {symbol} failed: {e}")
            return False

    def modify_sl(self, symbol: str, new_sl: float, tp: float, is_long: bool) -> bool:
        """Cancel existing SL/TP orders and re-place with new SL (keep TP)."""
        from binance.enums import (
            SIDE_BUY, SIDE_SELL,
            FUTURE_ORDER_TYPE_STOP_MARKET,
            FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
        )
        if not self.connected:
            return False
        try:
            cfg = BINANCE_PAIR_CONFIG.get(symbol, {"pp": 4, "qp": 2, "mq": 0.01})
            pp = cfg["pp"]
            close_side = SIDE_SELL if is_long else SIDE_BUY

            # Cancel all existing orders for this symbol
            self.client.futures_cancel_all_open_orders(symbol=symbol)

            sl_r = round(new_sl, pp)
            tp_r = round(tp, pp)

            # Re-place SL
            self.client.futures_create_order(
                symbol=symbol, side=close_side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_r, closePosition=True,
                workingType='MARK_PRICE',
            )
            # Re-place TP
            self.client.futures_create_order(
                symbol=symbol, side=close_side,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_r, closePosition=True,
                workingType='MARK_PRICE',
            )
            logger.info(f"  📈 Trailing SL: {symbol} new SL={sl_r} (TP={tp_r} preserved)")
            return True
        except Exception as e:
            logger.error(f"  Trailing SL modify failed for {symbol}: {e}")
            return False

    def get_today_realized_pnl(self) -> float:
        """Query Binance for today's total realized P/L (from income history)."""
        if not self.connected:
            return 0.0
        try:
            import calendar
            from datetime import timezone as tz
            now = datetime.now(tz.utc)
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_ms = int(start_of_day.timestamp() * 1000)
            income = self.client.futures_income_history(
                incomeType="REALIZED_PNL", startTime=start_ms, limit=1000,
            )
            total = sum(float(i.get("income", 0)) for i in income)
            return total
        except Exception as e:
            logger.debug(f"get_today_realized_pnl error: {e}")
            return 0.0

    def ensure_connected(self) -> bool:
        """Re-connect if the connection was lost."""
        if self.connected:
            try:
                self.client.futures_account()
                return True
            except Exception:
                self.connected = False
        logger.info("Reconnecting to Binance...")
        return self.connect()

    # ── Leverage helpers ─────────────────────────────────────────

    def _get_leverage_brackets(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch leverage brackets for a symbol from Binance Futures API."""
        if not self.connected or not self.client:
            return []
        try:
            data = self.client.futures_leverage_bracket(symbol=symbol)
            if isinstance(data, list) and data:
                brackets = data[0].get("brackets", [])
                if isinstance(brackets, list):
                    return brackets
        except Exception as e:
            logger.debug(f"  Could not fetch leverage brackets for {symbol}: {e}")
        return []

    def _get_max_leverage(self, symbol: str, target_margin_usdt: float) -> int:
        """Query Binance for the max leverage allowed for *symbol* given our margin."""
        if symbol in self._leverage_cache:
            return self._leverage_cache[symbol]

        brackets = self._get_leverage_brackets(symbol)
        if not brackets:
            # Fallback — try setting a high value; Binance will cap it
            self._leverage_cache[symbol] = 125
            return 125

        def _lev(b: Dict[str, Any]) -> int:
            try:
                return int(b.get("initialLeverage", 0))
            except Exception:
                return 0

        # Sort highest leverage first
        sorted_brackets = sorted(brackets, key=_lev, reverse=True)
        global_max = max((_lev(b) for b in sorted_brackets), default=125)

        # Pick the highest bracket whose notional cap fits our notional
        best = global_max
        for b in sorted_brackets:
            lev = _lev(b)
            if lev <= 0:
                continue
            try:
                cap = float(b.get("notionalCap", float("inf")))
            except Exception:
                continue
            required_notional = target_margin_usdt * lev
            if required_notional <= cap:
                best = lev
                break

        self._leverage_cache[symbol] = best
        return best

    def set_leverage(self, symbol: str, leverage: int) -> int:
        """Set leverage for symbol; returns actual leverage applied."""
        try:
            result = self.client.futures_change_leverage(
                symbol=symbol, leverage=leverage,
            )
            actual = int(result.get("leverage", leverage))
            logger.info(f"  Leverage {symbol}: requested {leverage}x → applied {actual}x")
            return actual
        except Exception as e:
            logger.warning(f"  Set leverage failed for {symbol}: {e}")
            return leverage

    # ── Trade execution ──────────────────────────────────────────

    def execute(
        self,
        symbol: str,
        direction: str,        # "bullish" / "bearish"
        sl: float,
        tp: float,
        margin_usdt: float = CRYPTO_TRADE_USDT,
    ) -> Dict[str, Any]:
        """Execute a futures trade with MAX leverage.

        Places one market entry, one SL (closePosition), and one TP
        order at 1.5R (reduceOnly, closes full position).
        """
        from binance.enums import (
            SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET,
            FUTURE_ORDER_TYPE_STOP_MARKET,
            FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
        )

        result = {"success": False, "symbol": symbol, "direction": direction}
        if not self.connected:
            result["error"] = "Not connected"
            return result

        try:
            cfg = BINANCE_PAIR_CONFIG.get(symbol, {"pp": 4, "qp": 2, "mq": 0.01})
            pp = cfg["pp"]
            qp = cfg["qp"]
            mq = cfg["mq"]

            # ── Query & apply MAX leverage for this symbol ────────
            max_lev = self._get_max_leverage(symbol, margin_usdt)
            actual_lev = self.set_leverage(symbol, max_lev)

            # Get current price
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])

            # Compute quantity from margin × max leverage
            notional = margin_usdt * actual_lev
            quantity = notional / price
            mult = 10 ** qp
            quantity = math.ceil((quantity - 1e-12) * mult) / mult
            if qp == 0:
                quantity = max(int(quantity), 1)
            quantity = max(quantity, mq)

            is_long = direction.lower() in ("bullish", "long")
            side = SIDE_BUY if is_long else SIDE_SELL
            close_side = SIDE_SELL if is_long else SIDE_BUY

            # ── Entry order: prefer LIMIT for lower fees ─────────
            # Use a limit order at the current price. If it doesn't fill
            # within ~5 s, fall back to a market order.
            from binance.enums import ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTC

            order = None
            try:
                limit_price = round(price, pp)
                order = self.client.futures_create_order(
                    symbol=symbol, side=side, type=ORDER_TYPE_LIMIT,
                    quantity=quantity, price=limit_price,
                    timeInForce=TIME_IN_FORCE_GTC,
                )
                limit_order_id = order['orderId']
                logger.info(f"  Limit order placed for {symbol} @ {limit_price}, waiting for fill...")

                # Wait up to 5 seconds for fill
                _filled = False
                for _wait in range(10):
                    time.sleep(0.5)
                    status = self.client.futures_get_order(
                        symbol=symbol, orderId=limit_order_id,
                    )
                    if status['status'] == 'FILLED':
                        _filled = True
                        order = status
                        logger.info(f"  ✅ Limit filled for {symbol}")
                        break
                    elif status['status'] in ('CANCELED', 'EXPIRED', 'REJECTED'):
                        break

                if not _filled:
                    # Cancel unfilled limit and fall back to market
                    try:
                        self.client.futures_cancel_order(
                            symbol=symbol, orderId=limit_order_id,
                        )
                    except Exception:
                        pass
                    logger.info(f"  Limit not filled, falling back to MARKET for {symbol}")
                    order = self.client.futures_create_order(
                        symbol=symbol, side=side, type=ORDER_TYPE_MARKET,
                        quantity=quantity,
                    )
            except Exception as limit_err:
                logger.warning(f"  Limit order failed ({limit_err}), using MARKET")
                order = self.client.futures_create_order(
                    symbol=symbol, side=side, type=ORDER_TYPE_MARKET,
                    quantity=quantity,
                )
            order_id = str(order['orderId'])
            exec_price = float(order.get('avgPrice', 0))
            if exec_price == 0:
                trades = self.client.futures_account_trades(symbol=symbol, limit=1)
                if trades:
                    exec_price = float(trades[0]['price'])

            # ── Cancel stale open orders AFTER market entry succeeds ─
            # Doing this AFTER ensures existing positions keep their
            # SL/TP protection if the market order were to fail.
            try:
                self.client.futures_cancel_all_open_orders(symbol=symbol)
                logger.info(f"  Cancelled old open orders for {symbol}")
            except Exception:
                pass  # No open orders is fine

            sl_r = round(sl, pp)
            tp_r = round(tp, pp)

            # SL order — closes full position
            sl_ok = False
            try:
                self.client.futures_create_order(
                    symbol=symbol, side=close_side,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    stopPrice=sl_r, closePosition=True,
                    workingType='MARK_PRICE',
                )
                sl_ok = True
            except Exception as e:
                logger.error(f"  ⚠ SL ORDER FAILED for {symbol}: {e}")

            # TP order — closes full position at 1.5R
            tp_ok = False
            try:
                self.client.futures_create_order(
                    symbol=symbol, side=close_side,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    stopPrice=tp_r, closePosition=True,
                    workingType='MARK_PRICE',
                )
                tp_ok = True
            except Exception as e:
                logger.error(f"  ⚠ TP ORDER FAILED for {symbol}: {e}")

            if not sl_ok or not tp_ok:
                logger.error(
                    f"  🚨 CRITICAL: {symbol} position OPEN without "
                    f"{'SL' if not sl_ok else ''}"
                    f"{' and ' if not sl_ok and not tp_ok else ''}"
                    f"{'TP' if not tp_ok else ''}! "
                    f"Place manually on Binance NOW!"
                )

            result.update({
                "success": True,
                "order_id": order_id,
                "exec_price": exec_price,
                "quantity": quantity,
                "sl": sl_r,
                "tp": tp_r,
                "leverage": actual_lev,
            })
            logger.info(
                f"✅ BINANCE {'LONG' if is_long else 'SHORT'} {symbol} "
                f"qty={quantity} @ {exec_price:.{pp}f}  "
                f"SL={sl_r}  TP={tp_r}  LEV={actual_lev}x"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Binance execute error: {e}")

        return result


# ============================================================================
# MT5 EXECUTOR
# ============================================================================

class MT5Executor:
    """Thin wrapper around MetaTrader5 for forex trade execution."""

    def __init__(self):
        self.mt5 = None
        self.login, self.password, self.server = _get_mt5_credentials()
        self.connected = False
        self.account_leverage: int = 1  # Queried on connect
        self._last_keepalive_ts: float = 0.0
        self._symbol_cache: Dict[str, str] = {}

    def connect(self) -> bool:
        self.connected = False
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            if not mt5.initialize():
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            if self.login and self.password:
                if not mt5.login(login=self.login, password=self.password, server=self.server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
            self.connected = True
            info = mt5.account_info()
            self.account_leverage = info.leverage
            logger.info(
                f"✅ MT5 connected: {info.name} | "
                f"Balance: {info.balance} {info.currency} | "
                f"Leverage: 1:{self.account_leverage} | "
                f"{'Demo' if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else 'LIVE'}"
            )
            return True
        except ImportError:
            logger.warning("MetaTrader5 library not installed — forex trading disabled")
            return False
        except Exception as e:
            logger.error(f"MT5 connect failed: {e}")
            return False

    def get_open_position_count(self) -> int:
        if not self.connected or self.mt5 is None:
            return 0
        positions = self.mt5.positions_get()
        return len(positions) if positions else 0

    def ensure_connected(self) -> bool:
        """Re-connect if the connection was lost."""
        if self.connected and self.mt5 is not None:
            try:
                info = self.mt5.account_info()
                if info is not None:
                    return True
            except Exception:
                pass
            self.connected = False
            try:
                self.mt5.shutdown()
            except Exception:
                pass
        else:
            self.connected = False
        logger.info("Reconnecting to MT5...")
        return self.connect()

    def keep_alive(self, interval_sec: int = MT5_KEEPALIVE_INTERVAL_SEC) -> bool:
        """Periodic MT5 health check to reduce silent logouts."""
        now_ts = time.time()
        if (now_ts - self._last_keepalive_ts) < interval_sec:
            return self.connected and self.mt5 is not None
        self._last_keepalive_ts = now_ts
        return self.ensure_connected()

    def get_open_positions(self):
        """Return current MT5 positions or an empty list if unavailable."""
        if not self.connected or self.mt5 is None:
            return []
        try:
            positions = self.mt5.positions_get()
            return positions or []
        except Exception as e:
            logger.warning(f"MT5 positions_get error: {e}")
            return []

    def _normalize_symbol(self, symbol: str) -> str:
        """Resolve strategy symbols to broker MT5 symbols, including suffixes."""
        requested = symbol.upper()
        if requested in self._symbol_cache:
            return self._symbol_cache[requested]
        mt5 = self.mt5
        if mt5 is None:
            return requested

        candidates = [requested]
        if requested.endswith("USDT"):
            candidates.append(requested[:-4] + "USD")
        candidates.extend([f"{c}m" for c in list(candidates)])
        candidates.extend([f"{c}.m" for c in list(candidates) if not c.endswith(".m")])

        for candidate in list(dict.fromkeys(candidates)):
            if mt5.symbol_select(candidate, True):
                self._symbol_cache[requested] = candidate
                return candidate

        search_key = requested[:-4] + "USD" if requested.endswith("USDT") else requested
        matches = mt5.symbols_get(f"*{search_key}*") or []
        for item in matches:
            name = getattr(item, "name", "")
            if name and mt5.symbol_select(name, True):
                self._symbol_cache[requested] = name
                return name

        return requested

    def _get_direct_fx_rate(self, from_ccy: str, to_ccy: str) -> Optional[float]:
        """Direct/inverse MT5 symbol lookup only (no triangulation).

        Falls back to hardcoded INR_USD_RATE for USD↔INR since MT5
        brokers typically don't list INR pairs.
        """
        f = (from_ccy or "").upper()
        t = (to_ccy or "").upper()
        if not f or not t:
            return None
        if f == t:
            return 1.0

        if f == "USD" and t == "INR":
            return INR_USD_RATE
        if f == "INR" and t == "USD":
            return 1.0 / INR_USD_RATE

        mt5 = self.mt5
        if mt5 is None:
            return None
        direct = f"{f}{t}"
        if mt5.symbol_select(direct, True):
            tick = mt5.symbol_info_tick(direct)
            if tick is not None and getattr(tick, "bid", 0) > 0:
                return float(tick.bid)

        inverse = f"{t}{f}"
        if mt5.symbol_select(inverse, True):
            tick = mt5.symbol_info_tick(inverse)
            if tick is not None and getattr(tick, "ask", 0) > 0:
                return 1.0 / float(tick.ask)

        return None

    def _get_fx_rate(self, from_ccy: str, to_ccy: str) -> Optional[float]:
        """Return conversion rate to convert 1 unit of from_ccy into to_ccy.

        Supports:
          1. Direct MT5 symbol (FROMTO) or inverse (TOFROM)
          2. Triangulation via USD  (e.g. JPY→USD→INR)
          3. Hardcoded INR_USD_RATE for the USD↔INR leg
        """
        f = (from_ccy or "").upper()
        t = (to_ccy or "").upper()
        if not f or not t:
            return None
        if f == t:
            return 1.0

        rate = self._get_direct_fx_rate(f, t)
        if rate is not None:
            return rate

        if f != "USD" and t != "USD":
            f_to_usd = self._get_direct_fx_rate(f, "USD")
            usd_to_t = self._get_direct_fx_rate("USD", t)
            if f_to_usd is not None and usd_to_t is not None:
                logger.debug(
                    f"  FX triangulation: {f}→USD={f_to_usd:.6f} × "
                    f"USD→{t}={usd_to_t:.6f} = {f_to_usd * usd_to_t:.6f}"
                )
                return f_to_usd * usd_to_t

        return None

    def _compute_lot_from_margin(
        self, symbol: str, direction: str,
        margin_inr: float = FOREX_MARGIN_INR,
    ) -> float:
        """Compute lot size so that used margin ≈ margin_inr.

        Uses MT5's order_calc_margin() for the actual per-lot margin
        (returned in account currency = INR), then scales to reach the target.
        If the initial API call returns None or an unreasonable value, falls
        back to querying margin for a small lot and scaling up.
        """
        mt5 = self.mt5

        sym = self._normalize_symbol(symbol)
        info = mt5.symbol_info(sym)
        if info is None:
            logger.warning(f"[{sym}] symbol_info returned None, using 0.01")
            return 0.01

        is_long = direction.lower() in ("bullish", "long")
        mt5_type = mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            logger.warning(f"[{sym}] symbol_info_tick returned None, using volume_min")
            return info.volume_min
        ref_price = tick.ask if is_long else tick.bid

        # ── Primary: ask MT5 for margin of 1.0 lot ───────────────
        margin_1lot = mt5.order_calc_margin(mt5_type, sym, 1.0, ref_price)
        logger.info(
            f"[{sym}] order_calc_margin(1.0 lot, price={ref_price}) = {margin_1lot}"
        )

        # ── Fallback 1: query margin for volume_min and scale up ──
        if margin_1lot is None or margin_1lot <= 0:
            logger.warning(
                f"[{sym}] order_calc_margin returned {margin_1lot} for 1.0 lot, "
                f"trying with volume_min={info.volume_min}"
            )
            margin_min = mt5.order_calc_margin(
                mt5_type, sym, info.volume_min, ref_price,
            )
            if margin_min is not None and margin_min > 0:
                margin_1lot = margin_min / info.volume_min
                logger.info(
                    f"[{sym}] Scaled from volume_min: "
                    f"margin({info.volume_min})={margin_min} → margin/lot={margin_1lot:.2f}"
                )

        # ── Fallback 2: use order_check to probe max lot ─────────
        if margin_1lot is None or margin_1lot <= 0:
            logger.warning(f"[{sym}] All margin queries failed, probing via order_check")
            # Try a test order with a reasonable lot to get the margin
            test_lot = 0.1
            test_req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": test_lot,
                "type": mt5_type,
                "price": ref_price,
            }
            check = mt5.order_check(test_req)
            if check is not None and hasattr(check, 'margin') and check.margin > 0:
                margin_1lot = check.margin / test_lot
                logger.info(
                    f"[{sym}] order_check probe: "
                    f"margin({test_lot})={check.margin} → margin/lot={margin_1lot:.2f}"
                )
            else:
                # Last resort: manual calc
                contract = info.trade_contract_size
                lev = self.account_leverage or 2000
                notional_quote = (contract * ref_price) / lev

                account_info = mt5.account_info()
                account_ccy = (account_info.currency if account_info else "INR").upper()
                quote_ccy = sym[3:6] if len(sym) >= 6 and sym[:6].isalpha() else "USD"

                conv = self._get_fx_rate(quote_ccy, account_ccy)
                if conv is not None and conv > 0:
                    margin_1lot = notional_quote * conv
                    logger.warning(
                        f"[{sym}] Using manual fallback with FX conversion: "
                        f"contract={contract} price={ref_price} lev={lev} "
                        f"{quote_ccy}->{account_ccy}={conv:.6f} "
                        f"→ margin/lot={margin_1lot:.2f}"
                    )
                else:
                    # Convert base currency to account currency instead
                    # (avoids the old bug of treating all quote ccys as USD)
                    base_ccy = sym[:3] if len(sym) >= 6 and sym[:6].isalpha() else "USD"
                    base_margin = contract / lev  # base-ccy units per lot
                    base_conv = self._get_fx_rate(base_ccy, account_ccy)
                    if base_conv is not None and base_conv > 0:
                        margin_1lot = base_margin * base_conv
                        logger.warning(
                            f"[{sym}] Manual fallback via base ccy: "
                            f"contract={contract} lev={lev} "
                            f"{base_ccy}->{account_ccy}={base_conv:.4f} "
                            f"→ margin/lot={margin_1lot:.2f}"
                        )
                    else:
                        margin_1lot = base_margin * INR_USD_RATE
                        logger.error(
                            f"[{sym}] LAST RESORT fallback (no FX rates found): "
                            f"treating base {base_ccy} as USD — "
                            f"contract={contract} lev={lev} "
                            f"→ margin/lot={margin_1lot:.2f}"
                        )

        if margin_1lot is None or margin_1lot <= 0:
            logger.error(
                f"[{sym}] Cannot compute margin per lot "
                f"(got {margin_1lot}), using volume_min"
            )
            return info.volume_min

        account_info = mt5.account_info()
        account_ccy = (account_info.currency if account_info else "INR").upper()
        target_margin = margin_inr
        if account_ccy != "INR":
            conv = self._get_fx_rate("INR", account_ccy)
            if conv is not None and conv > 0:
                target_margin = margin_inr * conv
            else:
                logger.warning(
                    f"[{sym}] Could not convert target margin INR->{account_ccy}; "
                    f"using raw target value {margin_inr}"
                )

        # order_calc_margin returns account currency, so size against that.
        raw_lot = target_margin / margin_1lot

        # Round down to nearest volume step
        vol_step = info.volume_step
        vol_decimals = len(str(vol_step).rstrip('0').split('.')[-1])
        lot = round(
            max(info.volume_min,
                (raw_lot // vol_step) * vol_step),
            vol_decimals,
        )
        lot = min(lot, info.volume_max)

        logger.info(
            f"[{sym}] margin target ₹{margin_inr} ({target_margin:.2f} {account_ccy}) "
            f"| margin/lot {margin_1lot:.2f} {account_ccy} | raw_lot={raw_lot:.4f} "
            f"| vol_step={vol_step} | final lot={lot}"
        )

        if lot <= info.volume_min and raw_lot < info.volume_min:
            logger.warning(
                f"[{sym}] ⚠ LOT CLAMPED to volume_min ({info.volume_min})! "
                f"margin/lot={margin_1lot:.2f} {account_ccy} is too high for "
                f"₹{margin_inr} target. "
                f"This means the position is much smaller than intended. "
                f"Possible causes: pair-specific leverage restriction, "
                f"wrong FX conversion, or order_calc_margin failure."
            )

        return lot

    def execute(
        self,
        symbol: str,
        direction: str,
        sl: float,
        tp: float,
        margin_inr: float = FOREX_MARGIN_INR,
    ) -> Dict[str, Any]:
        """Place a single MT5 order with TP at 1.5R.
        Lot size is computed dynamically to use ~margin_inr of margin."""
        mt5 = self.mt5
        result = {"success": False, "symbol": symbol, "direction": direction}
        if not self.connected:
            result["error"] = "Not connected"
            return result

        try:
            sym = self._normalize_symbol(symbol)
            if not mt5.symbol_select(sym, True):
                result["error"] = f"Symbol {sym} not available"
                return result

            info = mt5.symbol_info(sym)
            if info is None:
                result["error"] = f"No info for {sym}"
                return result

            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                result["error"] = "No tick data"
                return result

            # Compute lot size from margin target
            lot_size = self._compute_lot_from_margin(symbol, direction, margin_inr)
            logger.info(f"[{sym}] Using lot size {lot_size} for ₹{margin_inr} margin")

            digits = info.digits
            sl_r = round(sl, digits)
            tp_r = round(tp, digits)

            is_long = direction.lower() in ("bullish", "long")
            mt5_type = mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL
            price = tick.ask if is_long else tick.bid

            # Single order with full lot size
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": lot_size,
                "type": mt5_type,
                "price": price,
                "sl": sl_r,
                "tp": tp_r,
                "deviation": 20,
                "magic": MT5_MAGIC,
                "comment": "S09",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            account_info = mt5.account_info()
            check = mt5.order_check(req)
            if account_info is not None:
                result.update({
                    "account_currency": getattr(account_info, "currency", None),
                    "account_balance": getattr(account_info, "balance", None),
                    "account_equity": getattr(account_info, "equity", None),
                    "account_margin_free": getattr(account_info, "margin_free", None),
                    "account_leverage": getattr(account_info, "leverage", None),
                })
            result.update({
                "lot": lot_size,
                "sl": sl_r,
                "tp": tp_r,
            })
            if check is not None:
                result["order_check"] = {
                    "retcode": getattr(check, "retcode", None),
                    "comment": getattr(check, "comment", None),
                    "margin": getattr(check, "margin", None),
                    "margin_free": getattr(check, "margin_free", None),
                }
                logger.info(
                    f"[{sym}] order_check retcode={getattr(check, 'retcode', None)} "
                    f"comment={getattr(check, 'comment', None)} "
                    f"margin={getattr(check, 'margin', None)} "
                    f"free_after={getattr(check, 'margin_free', None)}"
                )
            res = mt5.order_send(req)
            if res is None:
                result["error"] = f"order_send None: {mt5.last_error()}"
                return result
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                result["error"] = f"Rejected: {res.comment} (code {res.retcode})"
                return result

            result.update({
                "success": True,
                "order_id": str(res.order),
                "exec_price": res.price,
                "lot": lot_size,
                "sl": sl_r,
                "tp": tp_r,
            })
            logger.info(
                f"✅ MT5 {'BUY' if is_long else 'SELL'} {sym}  "
                f"lot={lot_size}  TP={tp_r}  "
                f"@ {res.price:.{digits}f}  SL={sl_r}"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"MT5 execute error: {e}")

        return result

    def execute_with_retries(
        self,
        symbol: str,
        direction: str,
        sl: float,
        tp: float,
        margin_inr: float = FOREX_MARGIN_INR,
        retries: int = MT5_RECONNECT_RETRIES,
        retry_delay_sec: float = MT5_RETRY_DELAY_SEC,
    ) -> Dict[str, Any]:
        """Execute MT5 order with reconnect/retry on transient terminal disconnects."""
        last_result: Dict[str, Any] = {"success": False, "error": "MT5 retry not attempted"}

        transient_markers = (
            "not connected",
            "order_send none",
            "no tick data",
            "no info for",
            "trade context busy",
            "off quotes",
            "requote",
            "no ipc connection",
            "connection",
        )

        for attempt in range(1, retries + 2):
            if not self.ensure_connected():
                last_result = {"success": False, "error": "MT5 reconnect failed"}
            else:
                last_result = self.execute(symbol, direction, sl, tp, margin_inr)

            if last_result.get("success"):
                return last_result

            err = str(last_result.get("error", "")).lower()
            is_transient = any(marker in err for marker in transient_markers)
            if attempt <= retries and is_transient:
                logger.warning(
                    f"[{symbol}] MT5 transient failure: {last_result.get('error')} | "
                    f"retry {attempt}/{retries} in {retry_delay_sec:.1f}s"
                )
                time.sleep(retry_delay_sec)
                continue

            return last_result

        return last_result

    # ── Hard loss cap helpers ─────────────────────────────────
    def estimate_sl_loss(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        entry_price: float,
        sl_price: float,
    ) -> Optional[float]:
        """
        Estimate the loss (in account currency, e.g. INR) if SL is hit.
        Returns a positive number representing the loss amount, or None on error.
        """
        mt5 = self.mt5
        if not self.connected or mt5 is None:
            return None
        try:
            sym = self._normalize_symbol(symbol)
            is_long = direction.lower() in ("bullish", "long")
            action = mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL

            profit = mt5.order_calc_profit(action, sym, lot_size, entry_price, sl_price)
            if profit is None:
                return None
            # profit is negative when loss; return as positive loss amount
            return abs(profit) if profit < 0 else -profit
        except Exception as e:
            logger.warning(f"estimate_sl_loss error: {e}")
            return None

    def calculate_capped_sl(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        entry_price: float,
        original_sl: float,
        max_loss: float,
    ) -> float:
        """
        Binary search for a tighter SL that keeps loss <= max_loss.
        Returns the new (tighter) SL price, or original_sl if no cap needed.
        """
        mt5 = self.mt5
        if not self.connected or mt5 is None:
            return original_sl

        sym = self._normalize_symbol(symbol)
        info = mt5.symbol_info(sym)
        if info is None:
            return original_sl

        is_long = direction.lower() in ("bullish", "long")
        point = info.point
        digits = info.digits

        # Check if cap is even needed
        est_loss = self.estimate_sl_loss(symbol, direction, lot_size, entry_price, original_sl)
        if est_loss is None or est_loss <= max_loss:
            return original_sl

        logger.info(
            f"[{sym}] SL loss ₹{est_loss:.0f} > cap ₹{max_loss:.0f} — tightening SL"
        )

        # Binary search: narrow the SL until loss <= max_loss
        # For LONG: SL is below entry. Tight SL = closer to entry (higher).
        # For SHORT: SL is above entry. Tight SL = closer to entry (lower).
        if is_long:
            lo, hi = original_sl, entry_price - point
        else:
            lo, hi = entry_price + point, original_sl

        for _ in range(50):  # max 50 iterations
            mid = round((lo + hi) / 2, digits)
            loss_at_mid = self.estimate_sl_loss(symbol, direction, lot_size, entry_price, mid)
            if loss_at_mid is None:
                break

            if loss_at_mid <= max_loss:
                # Can afford this SL, try to widen (give more room)
                if is_long:
                    hi_new = mid
                    if hi_new == hi:
                        break
                    hi = hi_new
                else:
                    lo_new = mid
                    if lo_new == lo:
                        break
                    lo = lo_new
            else:
                # Too expensive, need tighter SL
                if is_long:
                    lo_new = mid
                    if lo_new == lo:
                        break
                    lo = lo_new
                else:
                    hi_new = mid
                    if hi_new == hi:
                        break
                    hi = hi_new

        # Return the tightest acceptable SL
        capped_sl = round(hi if is_long else lo, digits)
        verify_loss = self.estimate_sl_loss(symbol, direction, lot_size, entry_price, capped_sl)
        verify_loss_str = f"{verify_loss:.0f}" if verify_loss is not None else "?"
        logger.info(
            f"[{sym}] Capped SL: {original_sl} → {capped_sl}  "
            f"(est loss ₹{verify_loss_str})"
        )
        return capped_sl


    def modify_sl(self, symbol: str, new_sl: float, ticket: int = 0) -> bool:
        """Modify SL of an open MT5 position (keep existing TP)."""
        mt5 = self.mt5
        if not self.connected or mt5 is None:
            return False
        try:
            sym = self._normalize_symbol(symbol)
            positions = mt5.positions_get(symbol=sym)
            if not positions:
                return False

            position = None
            if ticket:
                for p in positions:
                    if p.ticket == ticket:
                        position = p
                        break
            # Fallback to first position with our magic
            if position is None:
                for p in positions:
                    if p.magic == MT5_MAGIC:
                        position = p
                        break
            if position is None:
                position = positions[0]

            info = mt5.symbol_info(sym)
            if info is None:
                return False
            digits = info.digits
            sl_r = round(new_sl, digits)

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": sym,
                "position": position.ticket,
                "sl": sl_r,
                "tp": position.tp,  # Keep existing TP
                "magic": MT5_MAGIC,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"  📈 Trailing SL: {sym} ticket={position.ticket} "
                    f"new SL={sl_r}"
                )
                return True
            else:
                comment = res.comment if res else "None"
                logger.warning(f"  Trailing SL modify failed for {sym}: {comment}")
                return False
        except Exception as e:
            logger.error(f"  Trailing SL modify error for {symbol}: {e}")
            return False


# ============================================================================
# TRAILING STOP-LOSS HELPER
# ============================================================================

def _get_trailing_target(
    profit_inr: float,
    step_inr: float,
) -> Optional[Tuple[int, float, float]]:
    """Return trailing ladder info for the current unrealized profit.

    The ladder is dynamic from the initial margin used on the trade:
      • First trigger at 1.2×margin profit → SL locks 1.0×margin
      • Then every +0.5×margin profit → SL locks +0.5×margin more

    Example for ₹1000 step:
      • Profit ₹1200 → lock ₹1000
      • Profit ₹1700 → lock ₹1500
      • Profit ₹2200 → lock ₹2000
    """
    if step_inr <= 0 or profit_inr <= 0:
        return None

    first_trigger_inr = step_inr * 1.2
    if profit_inr < first_trigger_inr:
        return None

    ladder_increment_inr = step_inr * 0.5
    if ladder_increment_inr <= 0:
        return None

    # Small epsilon avoids floating-point edge misses at exact milestones.
    extra_levels = int(math.floor(((profit_inr - first_trigger_inr) / ladder_increment_inr) + 1e-9))
    trail_level = 1 + max(0, extra_levels)
    trigger_inr = first_trigger_inr + extra_levels * ladder_increment_inr
    locked_inr = step_inr + extra_levels * ladder_increment_inr
    return trail_level, trigger_inr, locked_inr


def _check_trailing_sl(
    binance: Optional["BinanceExecutor"],
    mt5_exec: Optional["MT5Executor"],
    tracker: Dict[str, Dict[str, Any]],
    trades_log: Path,
    now_ist: datetime,
    crypto_step_inr: float = 0,
    forex_step_inr: float = 0,
) -> None:
    """Move SL once dynamic profit milestones are crossed.

    The ladder is based on the trade's margin step:
      • First trigger at 1.2×step profit → SL locks 1.0×step
      • Then every +0.5×step profit → SL locks +0.5×step
    """
    # Fallback if caller didn't provide steps
    if crypto_step_inr <= 0:
        crypto_step_inr = CRYPTO_TRADE_USDT * INR_USD_RATE  # e.g. 10*84 = ₹840
    if forex_step_inr <= 0:
        forex_step_inr = FOREX_MARGIN_INR                    # e.g. ₹1000

    # ── Crypto (Binance) ──────────────────────────────────────────
    market_step = crypto_step_inr
    if binance and binance.connected:
        try:
            open_syms = set()
            for pos in binance.get_open_positions():
                sym = pos['symbol']
                open_syms.add(sym)

                # Auto-discover positions not yet tracked (e.g. on restart)
                if sym not in tracker:
                    amt = pos['positionAmt']
                    _dir = 'bullish' if amt > 0 else 'bearish'
                    entry_p = pos['entryPrice']
                    # Read SL/TP from existing open orders
                    _tp_val = 0.0
                    _sl_val = 0.0
                    try:
                        oo = binance.client.futures_get_open_orders(symbol=sym)
                        for o in oo:
                            ot = o.get('type', '')
                            if 'TAKE_PROFIT' in ot:
                                _tp_val = float(o.get('stopPrice', 0))
                            elif 'STOP' in ot and 'TAKE_PROFIT' not in ot:
                                _sl_val = float(o.get('stopPrice', 0))
                    except Exception:
                        pass
                    if _sl_val == 0:
                        _sl_val = entry_p  # fallback
                    tracker[sym] = {
                        'entry_price': entry_p,
                        'current_sl': _sl_val,
                        'tp': _tp_val,
                        'direction': _dir,
                        'last_trail_level': 0,
                        'trail_step_inr': market_step,
                        'market': 'CRYPTO',
                    }
                    logger.info(
                        f"  🔍 Auto-discovered open crypto position: {sym} "
                        f"{_dir} entry={entry_p} SL={_sl_val} TP={_tp_val}"
                    )

                t = tracker[sym]
                step = float(t.get('trail_step_inr') or market_step)
                profit_usdt = pos['unRealizedProfit']
                profit_inr = profit_usdt * INR_USD_RATE

                trail_target = _get_trailing_target(profit_inr, step)
                if trail_target is None:
                    continue

                trail_level, trigger_inr, locked_inr = trail_target
                if trail_level <= t['last_trail_level']:
                    continue

                entry = t['entry_price']
                qty = abs(pos['positionAmt'])
                is_long = t['direction'].lower() in ('bullish', 'long')

                lock_usdt = locked_inr / INR_USD_RATE
                if qty > 0:
                    price_shift = lock_usdt / qty
                else:
                    continue

                new_sl = (entry + price_shift) if is_long else (entry - price_shift)

                # Only move if new SL is *better* (closer to current price)
                current_sl = t['current_sl']
                if is_long and new_sl <= current_sl:
                    continue
                if not is_long and new_sl >= current_sl:
                    continue

                success = binance.modify_sl(sym, new_sl, t['tp'], is_long)
                if success:
                    old_sl = t['current_sl']
                    t['current_sl'] = new_sl
                    t['last_trail_level'] = trail_level
                    logger.info(
                        f"  📈 {sym}: Profit ₹{profit_inr:.0f} crossed "
                        f"₹{trigger_inr:.0f} → SL {old_sl} → {new_sl} "
                        f"(locking ₹{locked_inr:.0f})"
                    )
                    print(
                        f"  📈 TRAILING SL: {sym} profit ₹{profit_inr:.0f} "
                        f"→ SL: {old_sl} → {new_sl} "
                        f"(locking ₹{locked_inr:.0f})"
                    )
                    _append_jsonl(trades_log, {
                        "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                        "market": "CRYPTO",
                        "symbol": sym,
                        "action": "TRAILING_SL",
                        "profit_inr": round(profit_inr, 2),
                        "old_sl": old_sl,
                        "new_sl": new_sl,
                        "trail_level": trail_level,
                    })

            # Clean up tracker for closed positions
            closed = [
                s for s in list(tracker)
                if tracker[s]['market'] == 'CRYPTO' and s not in open_syms
            ]
            for s in closed:
                del tracker[s]
        except Exception as e:
            logger.warning(f"Trailing SL crypto check error: {e}")
            logger.debug(traceback.format_exc())

    # ── Forex (MT5) ───────────────────────────────────────────────
    market_step = forex_step_inr
    # NOTE: We trail ALL open positions (not just magic=909009) because
    #       positions can be merged/replaced in MT5 netting mode, changing
    #       the magic number, or the user may have manually placed trades
    #       they also want trailed.
    if mt5_exec:
        try:
            positions = mt5_exec.get_open_positions()
            open_syms = set()
            if positions:
                for pos in positions:
                    sym = pos.symbol
                    open_syms.add(sym)

                    # Auto-discover forex positions not yet tracked
                    if sym not in tracker:
                        _dir = 'bullish' if pos.type == 0 else 'bearish'  # 0=BUY, 1=SELL
                        tracker[sym] = {
                            'entry_price': pos.price_open,
                            'current_sl': pos.sl,
                            'tp': pos.tp,
                            'direction': _dir,
                            'last_trail_level': 0,
                            'trail_step_inr': market_step,
                            'market': 'FOREX',
                        }
                        logger.info(
                            f"  🔍 Auto-discovered open forex position: {sym} "
                            f"{_dir} entry={pos.price_open} SL={pos.sl} TP={pos.tp}"
                        )

                    t = tracker[sym]
                    step = float(t.get('trail_step_inr') or market_step)
                    profit_inr = pos.profit  # Already in account currency (INR)

                    trail_target = _get_trailing_target(profit_inr, step)
                    if trail_target is None:
                        continue

                    trail_level, trigger_inr, locked_inr = trail_target
                    if trail_level <= t['last_trail_level']:
                        continue

                    entry = pos.price_open
                    current_price = pos.price_current
                    is_long = t['direction'].lower() in ('bullish', 'long')

                    # Derive price movement per INR of profit
                    price_diff = abs(current_price - entry)
                    if price_diff <= 0 or profit_inr <= 0:
                        continue
                    price_per_inr = price_diff / profit_inr

                    new_sl = (
                        entry + locked_inr * price_per_inr
                        if is_long
                        else entry - locked_inr * price_per_inr
                    )

                    current_sl = t['current_sl']
                    if is_long and new_sl <= current_sl:
                        continue
                    if not is_long and new_sl >= current_sl:
                        continue

                    success = mt5_exec.modify_sl(sym, new_sl, pos.ticket)
                    if success:
                        old_sl = t['current_sl']
                        t['current_sl'] = new_sl
                        t['last_trail_level'] = trail_level
                        logger.info(
                            f"  📈 {sym}: Profit ₹{profit_inr:.0f} crossed "
                            f"₹{trigger_inr:.0f} → SL {old_sl} → {new_sl} "
                            f"(locking ₹{locked_inr:.0f})"
                        )
                        print(
                            f"  📈 TRAILING SL: {sym} profit ₹{profit_inr:.0f} "
                            f"→ SL: {old_sl} → {new_sl} "
                            f"(locking ₹{locked_inr:.0f})"
                        )
                        _append_jsonl(trades_log, {
                            "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                            "market": "FOREX",
                            "symbol": sym,
                            "action": "TRAILING_SL",
                            "profit_inr": round(profit_inr, 2),
                            "old_sl": old_sl,
                            "new_sl": new_sl,
                            "trail_level": trail_level,
                        })

            # Clean up tracker for closed forex positions
            closed = [
                s for s in list(tracker)
                if tracker[s]['market'] == 'FOREX' and s not in open_syms
            ]
            for s in closed:
                del tracker[s]
        except Exception as e:
            logger.warning(f"Trailing SL forex check error: {e}")
            logger.debug(traceback.format_exc())


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Auto Trader — Strategy 09 (MSS+OB) — Crypto + Forex",
    )
    ap.add_argument("--crypto-pairs", nargs="+", default=CRYPTO_PAIRS)
    ap.add_argument("--forex-pairs", nargs="+", default=FOREX_PAIRS)
    ap.add_argument("--crypto-only", action="store_true")
    ap.add_argument("--forex-only", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--min-quality", type=int, default=50)
    ap.add_argument("--fresh-window", type=int, default=FRESH_WINDOW_MINUTES)
    ap.add_argument("--bias-ttl-hours", type=float, default=16.0)
    ap.add_argument("--bias-lookback-hours", type=int, default=72)
    ap.add_argument("--state-file",
                    default=str(THIS_DIR / "results" / "auto_trader_sent.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan and log signals but do NOT execute trades")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--skip-session-filter", action="store_true")
    ap.add_argument("--crypto-margin", type=float, default=CRYPTO_TRADE_USDT,
                    help="USDT margin per crypto trade (default 10)")
    ap.add_argument("--forex-margin", type=float, default=FOREX_MARGIN_INR,
                    help="INR margin per forex trade (default 1000)")
    # ── Killzone filter ───────────────────────────────────────────
    ap.add_argument("--no-killzone", action="store_true",
                    help="Allow trades outside London/NY killzones")
    ap.add_argument("--allow-asian", action="store_true",
                    help="Allow Asian session trades (default: blocked for forex)")
    # ── Strict EMA trend filter ───────────────────────────────────
    ap.add_argument("--no-ema-filter", action="store_true",
                    help="Disable strict 4H EMA trend alignment filter")
    # ── New crypto improvement flags ─────────────────────────────
    ap.add_argument("--no-blacklist", action="store_true",
                    help="Disable crypto symbol blacklist")
    ap.add_argument("--no-daily-limit", action="store_true",
                    help="Disable daily loss limit for crypto")
    ap.add_argument("--daily-loss-limit", type=float, default=CRYPTO_DAILY_LOSS_LIMIT,
                    help=f"USDT daily loss limit for crypto (default {CRYPTO_DAILY_LOSS_LIMIT})")
    ap.add_argument("--no-max-hold", action="store_true",
                    help="Disable max-hold-time force-close")
    ap.add_argument("--max-hold-hours", type=float, default=MAX_HOLD_SECONDS / 3600,
                    help=f"Max hold time in hours before force-close (default {MAX_HOLD_SECONDS/3600:.0f})")
    ap.add_argument("--no-trailing-sl", action="store_true",
                    help="Disable trailing SL")
    ap.add_argument("--trailing-sl-step", type=float, default=0,
                    help="INR profit step for trailing SL (default: auto = margin used)")
    # ── London open block ─────────────────────────────────────────
    ap.add_argument("--no-london-block", action="store_true",
                    help="Disable London Open manipulation block (07:00-08:00 UTC)")
    # ── Correlation guard (OFF by default — optimization shows net negative)
    ap.add_argument("--correlation-guard", action="store_true",
                    help="Enable currency correlation guard (OFF by default, costs more winners than it saves)")
    # SELL extra confirmation: REMOVED (curve-fitted, not generic)
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    scan_crypto = not args.forex_only
    scan_forex = not args.crypto_only

    # ── Strategy + Data Fetchers ──────────────────────────────────
    strategy = MSSOrderBlockStrategy()
    crypto_fetcher = MT5DataFetcher("CRYPTO") if scan_crypto else None
    forex_fetcher = MT5DataFetcher("FOREX") if scan_forex else None
    notifier = EmailNotifier()
    phase_log = PhaseLogger(THIS_DIR / "phase_logs" / "auto_trader")

    # ── Executors ─────────────────────────────────────────────────
    binance: Optional[BinanceExecutor] = None
    mt5_exec: Optional[MT5Executor] = None

    if scan_crypto and not args.dry_run:
        binance = BinanceExecutor(testnet=True)
        if not binance.connect():
            logger.warning("⚠ Binance connection failed — crypto trades will be skipped")
            binance = None

    if scan_forex and not args.dry_run:
        mt5_exec = MT5Executor()
        if not mt5_exec.connect():
            logger.warning("⚠ MT5 connection failed — forex trades will be skipped")
            mt5_exec = None

    # ── Log / state files ─────────────────────────────────────────
    log_dir = THIS_DIR / "phase_logs" / "auto_trader"
    log_dir.mkdir(parents=True, exist_ok=True)
    trades_log = log_dir / "trades.jsonl"
    cycle_log = log_dir / "cycle_summary.jsonl"

    state_path = Path(args.state_file)
    sent: Set[str] = set()
    if state_path.exists():
        try:
            sent = set(json.loads(state_path.read_text("utf-8")))
        except Exception:
            sent = set()

    # ── Build pair list ───────────────────────────────────────────
    pairs: List[tuple] = []
    if scan_crypto:
        for s in args.crypto_pairs:
            pairs.append((s, "CRYPTO", crypto_fetcher))
    if scan_forex:
        for s in args.forex_pairs:
            pairs.append((s, "FOREX", forex_fetcher))

    active: Dict[str, Dict[str, BiasTrack]] = {}
    last_cycle_key: Optional[datetime] = None
    first_run = True

    # ── Banner ────────────────────────────────────────────────────
    n_c = len(args.crypto_pairs) if scan_crypto else 0
    n_f = len(args.forex_pairs) if scan_forex else 0
    print("═" * 62)
    print("  STRATEGY 09 — MSS + OB  |  AUTO TRADER")
    print("═" * 62)
    print(f"  Crypto:         {n_c} pairs  (Binance {'Testnet' if binance else 'OFF'})")
    print(f"  Forex:          {n_f} pairs  (MT5 {'ON' if mt5_exec else 'OFF'})")
    print(f"  Fresh Window:   {args.fresh_window} min")
    print(f"  Data Source:    MT5 / Exness")
    print(f"  Crypto Margin:  ${args.crypto_margin} × MAX leverage (per symbol)")
    print(f"  Forex Margin:   ₹{args.forex_margin}  (account lev: 1:{mt5_exec.account_leverage if mt5_exec else 'N/A'})")
    print(f"  Min Quality:    {args.min_quality}")
    print(f"  Email:          {'ON' if notifier.enabled else 'OFF'}")
    print(f"  Dry Run:        {args.dry_run}")
    print(f"  Killzone:       {'OFF' if args.no_killzone else 'London/NY only'}")
    print(f"  EMA Filter:     {'OFF' if args.no_ema_filter else 'Strict 4H (Close>EMA21>EMA50)'}")
    print(f"  Blacklist:      {'OFF' if args.no_blacklist else f'{len(CRYPTO_SYMBOL_BLACKLIST)} symbols'}")
    print(f"  Daily Limit:    {'OFF' if args.no_daily_limit else f'${args.daily_loss_limit:.0f} USDT'}")
    print(f"  Max Hold:       {'OFF' if args.no_max_hold else f'{args.max_hold_hours:.0f}h → force-close'}")
    if args.no_trailing_sl:
        _tsl_label = "OFF"
    elif args.trailing_sl_step > 0:
        _tsl_label = f"Every ₹{args.trailing_sl_step:.0f} (manual)"
    else:
        _tsl_label = f"Dynamic (Crypto: ₹{args.crypto_margin * INR_USD_RATE:.0f}, Forex: ₹{args.forex_margin:.0f})"
    print(f"  Trailing SL:    {_tsl_label}")
    print(f"  London Block:   {'OFF' if args.no_london_block else '07:00-08:00 UTC blocked'}")
    print(f"  Correl Guard:   {'Max 1 pos per currency' if args.correlation_guard else 'OFF'}")
    print("═" * 62 + "\n")

    # ── Daily P/L tracker (crypto) ────────────────────────────────
    daily_crypto_pnl: Dict[str, float] = {}  # date_str → cumulative realized P/L

    # ── Trailing SL tracker ───────────────────────────────────────
    # symbol → {entry_price, current_sl, tp, direction, last_trail_level, trail_step_inr, market}
    trailing_sl_tracker: Dict[str, Dict[str, Any]] = {}

    # ── Main Loop ─────────────────────────────────────────────────
    while True:
        try:
            now_ist = datetime.now(IST)

            # Keep sessions warm so transient MT5 logout does not skip trades
            if scan_forex and mt5_exec and not args.dry_run:
                if not mt5_exec.keep_alive():
                    logger.warning("⚠ MT5 keep-alive failed; will retry on trade execution")

            # ── Trailing SL: runs EVERY poll (~60s) for fast reaction ──
            if not args.dry_run and not args.no_trailing_sl:
                # Step = margin used (dynamic), or manual override
                _c_step = args.trailing_sl_step if args.trailing_sl_step > 0 else args.crypto_margin * INR_USD_RATE
                _f_step = args.trailing_sl_step if args.trailing_sl_step > 0 else args.forex_margin
                _check_trailing_sl(
                    binance, mt5_exec, trailing_sl_tracker,
                    trades_log, now_ist,
                    crypto_step_inr=_c_step,
                    forex_step_inr=_f_step,
                )

            cycle_key = _due_5m_cycle(now_ist, args.poll_seconds, last_cycle_key)
            if cycle_key is None:
                time.sleep(min(5, args.poll_seconds))
                continue

            last_cycle_key = cycle_key
            loop_start = now_ist
            cycle_str = cycle_key.strftime("%H:%M")

            check_4h = _is_due_4h(cycle_key) or first_run
            check_1h = _is_due_1h(cycle_key) or first_run
            check_15m = _is_due_15m(cycle_key) or first_run

            phases = []
            if check_4h:  phases.append("4H")
            if check_1h:  phases.append("1H")
            if check_15m: phases.append("15M")
            phases.append("5M")
            label = "INIT" if first_run else ", ".join(phases)
            print(f"\n[{cycle_str}] ═══ Cycle: {label} ═══")

            refresh = {"5m", "15m"}
            if check_4h: refresh.add("4h")
            if check_1h: refresh.add("1h")

            forex_open = args.skip_session_filter or _is_forex_session(now_ist)
            cycle_trades = 0

            # ── Max-hold check: force-close stale crypto positions ──
            if scan_crypto and binance and not args.dry_run and not args.no_max_hold:
                try:
                    max_hold_ms = int(args.max_hold_hours * 3600 * 1000)
                    now_ms = int(time.time() * 1000)
                    for pos in binance.get_open_positions():
                        age_ms = now_ms - pos['updateTime']
                        if age_ms > max_hold_ms:
                            sym_name = pos['symbol']
                            age_h = age_ms / 3600000
                            logger.info(
                                f"  ⏰ {sym_name} held {age_h:.1f}h > "
                                f"{args.max_hold_hours:.0f}h limit — force-closing"
                            )
                            print(
                                f"  ⏰ Force-closing {sym_name} "
                                f"(held {age_h:.1f}h, limit {args.max_hold_hours:.0f}h)"
                            )
                            closed = binance.force_close_position(
                                sym_name, pos['positionAmt'],
                            )
                            if closed:
                                _append_jsonl(trades_log, {
                                    "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                                    "market": "CRYPTO",
                                    "symbol": sym_name,
                                    "action": "FORCE_CLOSE",
                                    "reason": f"Max hold {args.max_hold_hours:.0f}h exceeded ({age_h:.1f}h)",
                                    "unrealized_pnl": pos['unRealizedProfit'],
                                })
                except Exception as e:
                    logger.debug(f"Max-hold check error: {e}")

            # (Trailing SL already checked above, before cycle gate)

            # ── Daily loss limit check (crypto) ───────────────────
            today_str = now_ist.strftime("%Y-%m-%d")
            crypto_daily_stopped = False
            if not args.no_daily_limit and scan_crypto and binance:
                # Get actual realized P/L from Binance for today
                today_pnl = binance.get_today_realized_pnl()
                daily_crypto_pnl[today_str] = today_pnl
                if today_pnl < -args.daily_loss_limit:
                    crypto_daily_stopped = True
                    logger.info(
                        f"  📛 Daily crypto loss limit reached: "
                        f"${today_pnl:.2f} < -${args.daily_loss_limit:.0f}"
                    )

            for symbol, market, fetcher in pairs:
                if market == "FOREX" and not forex_open:
                    continue

                now_utc = datetime.now(UTC)
                sym = active.setdefault(symbol, {})

                # Expire
                expired = [k for k, t in sym.items()
                           if t.completed or now_utc >= t.expires_utc]
                for k in expired:
                    del sym[k]

                # Fetch
                for iv in refresh:
                    fetcher.fetch_ohlcv(symbol, iv, {
                        "4h": 200, "1h": 500, "15m": 1000, "5m": 3000,
                    }.get(iv, 800), force_refresh=True)

                df_4h  = fetcher.fetch_ohlcv(symbol, "4h", 200)
                df_1h  = fetcher.fetch_ohlcv(symbol, "1h", 500)
                df_15m = fetcher.fetch_ohlcv(symbol, "15m", 1000)
                df_5m  = fetcher.fetch_ohlcv(symbol, "5m", 3000)

                df_4h  = _drop_incomplete(df_4h, 240, now_utc)
                df_1h  = _drop_incomplete(df_1h, 60, now_utc)
                df_15m = _drop_incomplete(df_15m, 15, now_utc)
                df_5m  = _drop_incomplete(df_5m, 5, now_utc)

                if any(x is None or x.empty for x in [df_4h, df_1h, df_15m, df_5m]):
                    continue

                # ═══ PHASE 1 — 4H Bias ═══════════════════════════
                if check_4h:
                    biases = strategy.determine_bias(df_4h, args.bias_lookback_hours)

                    phase_log.log_bias(
                        symbol=symbol,
                        biases_found=len(biases),
                        bias_details=[{
                            "direction": b.direction.value,
                            "reason": b.reason,
                            "sweep_time": str(b.sweep_timestamp),
                        } for b in biases],
                        lookback_hours=args.bias_lookback_hours,
                    )

                    for b in biases:
                        k = _bias_key(b)
                        if k in sym:
                            continue
                        evt = b.sweep_timestamp
                        if hasattr(evt, 'to_pydatetime'):
                            evt = evt.to_pydatetime()
                        if evt.tzinfo is None:
                            evt = UTC.localize(evt)
                        sym[k] = BiasTrack(
                            key=k, bias=b, market=market,
                            created_utc=now_utc,
                            expires_utc=now_utc + timedelta(hours=args.bias_ttl_hours),
                        )

                if not sym:
                    continue

                # ═══ PHASE 2 — 1H MSS ════════════════════════════
                if check_1h:
                    for t in sym.values():
                        if t.mss is not None:
                            continue
                        mss_conf = strategy.confirm_mss(df_1h, t.bias)
                        t.mss = mss_conf

                        phase_log.log_mss(
                            symbol=symbol,
                            bias_direction=t.bias.direction.value,
                            sweep_time=t.bias.sweep_timestamp,
                            confirmed=mss_conf is not None,
                            mss_break_price=mss_conf.mss.break_price if mss_conf else None,
                            mss_time=mss_conf.timestamp if mss_conf else None,
                            details=mss_conf.details if mss_conf else "",
                        )

                # ═══ PHASE 3+4 — 15M OB + 5M Tap ═════════════════
                for t in sym.values():
                    if t.alerted or t.completed:
                        continue
                    if t.mss is None:
                        continue

                    # If we previously rejected a stale tap, trim 5M
                    # data so the strategy only sees candles AFTER it.
                    # This lets us skip old taps and find fresh ones.
                    df_5m_filtered = df_5m
                    if t.last_rejected_tap is not None:
                        cutoff = t.last_rejected_tap
                        if cutoff.tzinfo is None:
                            cutoff = UTC.localize(cutoff)
                        df_5m_filtered = df_5m[df_5m.index > cutoff]
                        if df_5m_filtered.empty:
                            continue  # no new 5M data yet

                    ob_entry = strategy.find_ob_entry(
                        df_15m, df_5m_filtered, t.bias, t.mss,
                    )
                    if ob_entry is None:
                        continue

                    # ── Quality check ─────────────────────────────
                    quality = strategy.calculate_quality_score(t.bias, t.mss, ob_entry)
                    if quality < args.min_quality:
                        # Low quality — reject this tap, keep looking
                        tap_ts = ob_entry.timestamp
                        if hasattr(tap_ts, 'to_pydatetime'):
                            tap_ts = tap_ts.to_pydatetime()
                        if tap_ts.tzinfo is None:
                            tap_ts = UTC.localize(tap_ts)
                        t.last_rejected_tap = tap_ts
                        continue

                    # ── Freshness ─────────────────────────────────
                    if not _is_fresh(ob_entry.timestamp, now_utc, args.fresh_window):
                        # Stale tap — DON'T kill the track.
                        # Record its time so next cycle we skip past it
                        # and can catch a newer, fresh tap.
                        tap_ts = ob_entry.timestamp
                        if hasattr(tap_ts, 'to_pydatetime'):
                            tap_ts = tap_ts.to_pydatetime()
                        if tap_ts.tzinfo is None:
                            tap_ts = UTC.localize(tap_ts)
                        t.last_rejected_tap = tap_ts
                        continue

                    # ── Fresh tap found — lock it in ──────────────
                    t.ob_entry = ob_entry

                    # Log Phase 3 (OB) + Phase 4 (Tap)
                    phase_log.log_ob(
                        symbol=symbol,
                        bias_direction=t.bias.direction.value,
                        mss_time=t.mss.timestamp,
                        ob_found=True,
                        ob_time=ob_entry.order_block.datetime,
                        ob_top=ob_entry.order_block.top,
                        ob_bottom=ob_entry.order_block.bottom,
                        fib_level=ob_entry.fib_level,
                        in_ote=ob_entry.in_ote_zone,
                    )
                    phase_log.log_tap(
                        symbol=symbol,
                        bias_direction=t.bias.direction.value,
                        ob_time=ob_entry.order_block.datetime,
                        tapped=True,
                        tap_time=ob_entry.timestamp,
                        entry_price=ob_entry.entry_price,
                    )

                    # ── Build signal JSON for logging / email ─────
                    sig_dir = (
                        Direction.BULLISH if t.bias.direction == BiasType.BULLISH
                        else Direction.BEARISH
                    )
                    signal = MSSOB_Signal(
                        index=0, datetime=ob_entry.timestamp, direction=sig_dir,
                        entry_price=ob_entry.entry_price,
                        stop_loss=ob_entry.stop_loss,
                        take_profit=ob_entry.tp_price,
                        risk_reward=ob_entry.risk_reward,
                        signal_type="MSS_OB_ENTRY", symbol=symbol,
                        daily_bias=t.bias, mss_confirmation=t.mss,
                        ob_entry=ob_entry, quality_score=quality,
                    )
                    sig_json = format_signal_for_jsonl(signal)
                    sig_json["market"] = market

                    # ── Dedup ─────────────────────────────────────
                    sid = _signal_id(sig_json)
                    if sid in sent:
                        t.alerted = True
                        t.completed = True
                        continue

                    # ══════════════════════════════════════════════
                    # FRESH SIGNAL → FILTERS → COMPUTE SL → EXECUTE
                    # ══════════════════════════════════════════════

                    # ── STRICT 4H EMA TREND FILTER ─────────────
                    if not args.no_ema_filter:
                        ema_ok, ema_detail = _check_strict_ema(
                            df_4h, t.bias.direction.value,
                        )
                        if not ema_ok:
                            # Reject this tap but keep bias alive
                            # so a fresh retap can be taken later
                            # if 4H alignment becomes valid.
                            tap_ts = ob_entry.timestamp
                            if hasattr(tap_ts, 'to_pydatetime'):
                                tap_ts = tap_ts.to_pydatetime()
                            if tap_ts.tzinfo is None:
                                tap_ts = UTC.localize(tap_ts)
                            t.last_rejected_tap = tap_ts
                            t.alerted = False
                            t.completed = False
                            logger.info(
                                f"  [{symbol}] EMA REJECT: {ema_detail}"
                            )
                            print(
                                f"  ⊘ [{symbol}] "
                                f"{t.bias.direction.value} rejected: "
                                f"{ema_detail}"
                            )
                            _append_jsonl(trades_log, {
                                "timestamp_ist": now_ist.strftime(
                                    "%Y-%m-%d %H:%M:%S IST"
                                ),
                                "market": market,
                                "symbol": symbol,
                                "direction": t.bias.direction.value,
                                "entry_price": ob_entry.entry_price,
                                "quality": quality,
                                "trade_result": {
                                    "success": False,
                                    "error": f"EMA: {ema_detail}",
                                },
                            })
                            continue
                        else:
                            logger.info(
                                f"  [{symbol}] EMA OK: {ema_detail}"
                            )

                    t.alerted = True
                    t.completed = True

                    direction_str = t.bias.direction.value

                    # ── SYMBOL BLACKLIST FILTER (crypto) ──────────
                    if market == "CRYPTO" and not args.no_blacklist:
                        if symbol in CRYPTO_SYMBOL_BLACKLIST:
                            logger.info(f"  [{symbol}] BLACKLIST REJECT")
                            print(f"  ⊘ [{symbol}] {direction_str} rejected: blacklisted symbol")
                            _append_jsonl(trades_log, {
                                "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                                "market": market, "symbol": symbol,
                                "direction": direction_str,
                                "entry_price": ob_entry.entry_price,
                                "quality": quality,
                                "trade_result": {"success": False, "error": "Blacklisted symbol"},
                            })
                            continue

                    # ── DAILY LOSS LIMIT FILTER (crypto) ──────────
                    if market == "CRYPTO" and crypto_daily_stopped:
                        pnl_so_far = daily_crypto_pnl.get(today_str, 0.0)
                        logger.info(
                            f"  [{symbol}] DAILY LIMIT REJECT "
                            f"(today P/L: ${pnl_so_far:.2f}, limit: -${args.daily_loss_limit:.0f})"
                        )
                        print(
                            f"  ⊘ [{symbol}] {direction_str} rejected: "
                            f"daily loss limit hit (${pnl_so_far:.2f})"
                        )
                        _append_jsonl(trades_log, {
                            "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                            "market": market, "symbol": symbol,
                            "direction": direction_str,
                            "entry_price": ob_entry.entry_price,
                            "quality": quality,
                            "trade_result": {
                                "success": False,
                                "error": f"Daily loss limit (${pnl_so_far:.2f} < -${args.daily_loss_limit:.0f})",
                            },
                        })
                        continue

                    # ── KILLZONE FILTER (Forex + Crypto) ─────────
                    if not args.no_killzone:
                        kz_result = check_killzone(now_ist, market=market, allow_asian=args.allow_asian)
                        if not kz_result.passed:
                            logger.info(f"  [{symbol}] KILLZONE REJECT: {kz_result.reason}")
                            print(f"  ⊘ [{symbol}] {direction_str} rejected: {kz_result.reason}")
                            trade_record = {
                                "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                                "market": market,
                                "symbol": symbol,
                                "direction": direction_str,
                                "entry_price": ob_entry.entry_price,
                                "quality": quality,
                                "signal_time_ist": sig_json.get("signal_datetime_ist", ""),
                                "trade_result": {
                                    "success": False,
                                    "error": f"Killzone: {kz_result.reason}",
                                },
                            }
                            _append_jsonl(trades_log, trade_record)
                            continue
                        else:
                            logger.info(f"  [{symbol}] Killzone OK: {kz_result.reason}")

                    # ── LONDON OPEN BLOCK (07:00-08:00 UTC) ────────
                    if not args.no_london_block:
                        if _is_london_open_block(now_utc):
                            _lo_reason = (
                                f"London Open block (07:00-08:00 UTC) — "
                                f"manipulation phase, banks sweeping liquidity"
                            )
                            logger.info(f"  [{symbol}] LONDON BLOCK: {_lo_reason}")
                            print(f"  ⊘ [{symbol}] {direction_str} rejected: {_lo_reason}")
                            _append_jsonl(trades_log, {
                                "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                                "market": market, "symbol": symbol,
                                "direction": direction_str,
                                "entry_price": ob_entry.entry_price,
                                "quality": quality,
                                "trade_result": {
                                    "success": False,
                                    "error": f"London Open block: {_lo_reason}",
                                },
                            })
                            continue

                    # ── CORRELATION GUARD (opt-in) ────────────────
                    if args.correlation_guard:
                        _exposed = _get_exposed_currencies(mt5_exec, binance)
                        _corr_ok, _corr_reason = _check_correlation_guard(
                            symbol, market, _exposed,
                        )
                        if not _corr_ok:
                            logger.info(f"  [{symbol}] CORRELATION BLOCK: {_corr_reason}")
                            print(f"  ⊘ [{symbol}] {direction_str} rejected: {_corr_reason}")
                            _append_jsonl(trades_log, {
                                "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                                "market": market, "symbol": symbol,
                                "direction": direction_str,
                                "entry_price": ob_entry.entry_price,
                                "quality": quality,
                                "trade_result": {
                                    "success": False,
                                    "error": f"Correlation: {_corr_reason}",
                                },
                            })
                            continue
                        else:
                            logger.info(f"  [{symbol}] Correlation OK: {_corr_reason}")

                    # ── SELL EXTRA CONFIRMATION ─────────────────
                    # DISABLED: Real-data analysis proved this was curve-fitted
                    # to a specific market regime (Apr 2026 USD bull).
                    # The OB strategy enters on pullbacks, so by design the
                    # 1H close is on the "wrong" side of EMA21 at entry.
                    # Keeping the function for future reference but not using it.
                    # See: generic_filter_analysis.md for full reasoning.

                    # Smart SL
                    sl_price, sl_reason = compute_smart_sl(
                        ob=ob_entry.order_block,
                        bias_direction=t.bias.direction,
                        df_5m=df_5m,
                        tap_time=ob_entry.timestamp,
                        entry_price=ob_entry.entry_price,
                        market=market,
                        symbol=symbol,
                    )

                    # TP at 1.5R
                    tp_price = compute_tp(
                        ob_entry.entry_price, sl_price, rr=1.5,
                    )

                    # Verify SL is on correct side
                    if t.bias.direction == BiasType.BEARISH:
                        if sl_price <= ob_entry.entry_price:
                            logger.warning(f"  [{symbol}] SL {sl_price} <= entry {ob_entry.entry_price}, skip")
                            continue
                    else:
                        if sl_price >= ob_entry.entry_price:
                            logger.warning(f"  [{symbol}] SL {sl_price} >= entry {ob_entry.entry_price}, skip")
                            continue

                    # Log the signal
                    sig_time_ist = sig_json.get("signal_datetime_ist", "")
                    phase_log.log_signal(
                        symbol=symbol,
                        direction=direction_str,
                        signal_time_ist=sig_time_ist,
                        entry_price=ob_entry.entry_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        quality_score=quality,
                        risk_reward=1.5,
                        extra={"tp": tp_price,
                               "sl_reason": sl_reason, "market": market,
                               "killzone": "ON" if not args.no_killzone else "OFF"},
                    )

                    # Console
                    pfmt = ".5f" if market == "FOREX" else ".6f"
                    lev_info = ""
                    if market == "CRYPTO" and binance:
                        lev_info = f"  (max lev: {binance._get_max_leverage(symbol, args.crypto_margin)}x)"
                    elif market == "FOREX" and mt5_exec:
                        lev_info = f"  (account lev: 1:{mt5_exec.account_leverage})"
                    print(f"\n{'═'*62}")
                    print(f"  🚨 TRADE: [{market}] {symbol} {direction_str.upper()}{lev_info}")
                    print(f"     Entry:  {ob_entry.entry_price:{pfmt}}")
                    print(f"     SL:     {sl_price:{pfmt}}  ({sl_reason})")
                    print(f"     TP:     {tp_price:{pfmt}}  (1.5R)")
                    print(f"     Q:      {quality}/100")
                    print(f"{'═'*62}")

                    # ── Execute ───────────────────────────────────
                    trade_result: Dict[str, Any] = {"success": False}

                    if args.dry_run:
                        trade_result = {
                            "success": True, "order_id": "DRY_RUN",
                            "exec_price": ob_entry.entry_price,
                            "sl": sl_price, "tp": tp_price,
                        }
                        print("  (dry-run — trade NOT placed)")
                    elif market == "CRYPTO":
                        if binance is None:
                            binance = BinanceExecutor(testnet=True)
                            if not binance.connect():
                                trade_result["error"] = "Binance not connected"
                                print(f"  ⊘ Binance not connected — trade skipped")
                                binance = None
                        if binance is not None:
                            if not binance.ensure_connected():
                                trade_result["error"] = "Binance reconnect failed"
                                print(f"  ⊘ Binance reconnect failed")
                                binance = None
                            elif binance.get_open_position_count() >= MAX_CRYPTO_POSITIONS:
                                trade_result["error"] = f"Max positions ({MAX_CRYPTO_POSITIONS})"
                                print(f"  ⊘ Max crypto positions ({MAX_CRYPTO_POSITIONS}) reached")
                            else:
                                trade_result = binance.execute(
                                    symbol, direction_str,
                                    sl_price, tp_price,
                                    margin_usdt=args.crypto_margin,
                                )
                    elif market == "FOREX":
                        if mt5_exec is None:
                            mt5_exec = MT5Executor()
                            if not mt5_exec.connect():
                                trade_result["error"] = "MT5 not connected (is MT5 terminal running?)"
                                print(f"  ⊘ MT5 not connected — is the MT5 terminal app open?")
                                mt5_exec = None
                        if mt5_exec is not None:
                            if not mt5_exec.ensure_connected():
                                trade_result["error"] = "MT5 reconnect failed"
                                print(f"  ⊘ MT5 reconnect failed (will retry)")
                            elif mt5_exec.get_open_position_count() >= MAX_FOREX_POSITIONS:
                                trade_result["error"] = f"Max positions ({MAX_FOREX_POSITIONS})"
                                print(f"  ⊘ Max forex positions ({MAX_FOREX_POSITIONS}) reached")
                            else:
                                # ── Hard loss cap: tighten SL if needed ───
                                final_sl = sl_price
                                if HARD_LOSS_CAP_ENABLED:
                                    lot = mt5_exec._compute_lot_from_margin(
                                        symbol, direction_str, args.forex_margin
                                    )
                                    est_loss = mt5_exec.estimate_sl_loss(
                                        symbol, direction_str, lot,
                                        ob_entry.entry_price, sl_price
                                    )
                                    # Use a tighter effective cap that accounts
                                    # for commission so the TOTAL loss stays
                                    # within HARD_LOSS_CAP_AMOUNT.
                                    effective_cap = max(
                                        100,
                                        HARD_LOSS_CAP_AMOUNT - HARD_LOSS_CAP_COMMISSION_BUFFER,
                                    )
                                    if est_loss is not None and est_loss > effective_cap:
                                        final_sl = mt5_exec.calculate_capped_sl(
                                            symbol, direction_str, lot,
                                            ob_entry.entry_price, sl_price,
                                            effective_cap
                                        )
                                        print(
                                            f"     ⚠ SL capped: {sl_price:.5f} → {final_sl:.5f} "
                                            f"(₹{HARD_LOSS_CAP_AMOUNT} cap, ₹{HARD_LOSS_CAP_COMMISSION_BUFFER} commission buffer)"
                                        )

                                trade_result = mt5_exec.execute_with_retries(
                                    symbol, direction_str,
                                    final_sl, tp_price,
                                    margin_inr=args.forex_margin,
                                    retries=MT5_RECONNECT_RETRIES,
                                    retry_delay_sec=MT5_RETRY_DELAY_SEC,
                                )
                    else:
                        trade_result["error"] = f"Unknown market: {market}"
                        print(f"  ⊘ Unknown market: {market}")

                    # ── Log trade ─────────────────────────────────
                    trade_record = {
                        "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                        "market": market,
                        "symbol": symbol,
                        "direction": direction_str,
                        "entry_price": ob_entry.entry_price,
                        "smart_sl": sl_price,
                        "sl_reason": sl_reason,
                        "tp": tp_price,
                        "quality": quality,
                        "signal_time_ist": sig_json.get("signal_datetime_ist", ""),
                        "ob_top": ob_entry.order_block.top,
                        "ob_bottom": ob_entry.order_block.bottom,
                        "trade_result": trade_result,
                    }
                    _append_jsonl(trades_log, trade_record)

                    if trade_result.get("success"):
                        cycle_trades += 1
                        # Track for trailing SL (use actual SL from trade, may be capped)
                        actual_sl = trade_result.get("sl", sl_price)
                        trailing_sl_tracker[symbol] = {
                            "entry_price": trade_result.get("exec_price", ob_entry.entry_price),
                            "current_sl": actual_sl,
                            "tp": tp_price,
                            "direction": direction_str,
                            "last_trail_level": 0,
                            "trail_step_inr": (
                                args.trailing_sl_step
                                if args.trailing_sl_step > 0
                                else (args.crypto_margin * INR_USD_RATE if market == "CRYPTO" else args.forex_margin)
                            ),
                            "market": market,
                        }

                    # Email
                    if notifier.enabled and trade_result.get("success"):
                        sig_json["sl_price"] = sl_price
                        sig_json["tp_price"] = tp_price
                        sig_json["sl_reason"] = sl_reason
                        sig_json["trade_executed"] = True
                        sig_json["trade_result"] = {
                            k: v for k, v in trade_result.items()
                            if k in ("order_id", "exec_price", "sl", "tp")
                        }
                        notifier.send_signal(sig_json, market=market)

                    # Persist dedup
                    sent.add(sid)
                    try:
                        state_path.parent.mkdir(parents=True, exist_ok=True)
                        state_path.write_text(
                            json.dumps(sorted(sent), indent=2), encoding="utf-8",
                        )
                    except Exception:
                        pass

            # ── Cycle summary ─────────────────────────────────────
            n_scanned = sum(
                1 for _, m, _ in pairs
                if m == "CRYPTO" or (m == "FOREX" and forex_open)
            )
            _append_jsonl(cycle_log, {
                "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                "cycle": cycle_str,
                "phases": label,
                "pairs_scanned": n_scanned,
                "trades_executed": cycle_trades,
            })
            phase_log.log_cycle(
                cycle_time_ist=now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                symbols_scanned=n_scanned,
                signals_generated=cycle_trades,
            )

            if cycle_trades == 0:
                print(f"  ── No trades this cycle ──")
            else:
                print(f"  ✅ {cycle_trades} trade(s) executed")

            elapsed = (datetime.now(IST) - loop_start).total_seconds()
            first_run = False
            time.sleep(max(1, args.poll_seconds - int(elapsed)))

        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
            break
        except Exception as e:
            ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{ts}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            time.sleep(30)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
