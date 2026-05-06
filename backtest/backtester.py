"""
Strategy 09 Backtester Engine
==============================

Mirrors ``auto_trader.py`` replay semantics on historical data:

  • Clock steps at **5m bar close** (no lookahead on the forming candle).
  • ``_drop_incomplete`` on every timeframe — same as MT5 fetch (no synthetic
    partial 4H/1H/15m OHLC).
  • Stateful bias tracks + ``last_rejected_tap`` — same tap-skipping as live.

**How close to live can you get?**

  1. **Same code path** — filters, cycle gates (:31 1H/4H), EMA on closed 4H, MSS on closed 1H,
     stateful tracks: implemented here.
  2. **Same bars** — required for trade-for-trade match. Export CSVs from the **same** MT5
     symbol settings / session as ``auto_trader``. If 4H candle opens differ (e.g. sweep index
     at 05:00 UTC live vs 04:00 UTC in CSV), phases diverge even with identical code.
  3. **Optional** ``intrabar_htf_bias_mss=True`` (default) rebuilds a *forming* 4H candle from
     closed 5m for **Phase 1 sweep timing** only (often closer to what you see on a chart);
     ``auto_trader.py`` itself uses **closed** 4H only for ``determine_bias`` — use
     ``intrabar_htf_bias_mss=False`` for strict file-for-file parity with that script.
  4. **Residual gap** — tick path inside 5m bars (SL vs TP ordering), spread, fill time, and
     trailing SL are not fully reproducible from OHLC alone; expect close alignment of
     *decision windows*, not always identical tickets.

  Phase 1: 4H Bias (liquidity sweep)
  Phase 2: 1H MSS confirmation
  Phase 3: 15M Order Block in OTE zone
  Phase 4: 5M tap → entry signal

Filters (same as live):
  • Strict 4H EMA: Close > EMA21 > EMA50 (bull) / Close < EMA21 < EMA50 (bear)
  • Killzone: London (13–17 IST) + NY (18:30–22:30 IST)
  • London Open block: 07:00–08:00 UTC
  • Quality >= min_quality (default 50)
  • Symbol blacklist (crypto)

SL/TP:
  • Smart SL: OB extreme → 5M swing scan → spread buffer
  • TP at 1.5R

Trade Simulation:
  • Walk forward through 5M bars after entry
  • Check if SL or TP is hit first
  • Trailing SL ladder (optional)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
import pytz
import sys

# ---------------------------------------------------------------------------
# Path setup — import strategy modules from the repo
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "strategy_09_mss_ob_entry"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from strategy_09_mss_ob_entry.strategy import (
    MSSOrderBlockStrategy, MSSOB_Signal, DailyBias,
    MSSConfirmation, OBEntrySetup, BiasType,
    format_signal_for_jsonl,
)
from strategy_09_mss_ob_entry.entry_filters import (
    check_killzone, FilterResult, calculate_ema,
)
from scripts.utils.indicators import Direction, OrderBlock

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC

# ============================================================================
# CONSTANTS (mirrored from auto_trader.py)
# ============================================================================

CRYPTO_SL_BUFFER_PCT = 0.35       # 0.35% buffer for crypto
FOREX_SL_BUFFER_PIPS = 15         # 15 pips buffer for forex (10 for JPY)

LONDON_OPEN_BLOCK_START_UTC = 7 * 60   # 07:00 UTC in minutes
LONDON_OPEN_BLOCK_END_UTC   = 8 * 60   # 08:00 UTC in minutes

MIN_4H_BARS_FOR_EMA = 50

# Same as auto_trader.py — tap must be fresh; bias setups expire
FRESH_WINDOW_MINUTES_DEFAULT = 15
BIAS_TTL_HOURS_DEFAULT = 16.0

# Position-sizing defaults for INR P&L calculation
DEFAULT_CAPITAL_INR = 1000.0
DEFAULT_LEVERAGE    = 2000
HARD_LOSS_CAP_INR   = 3000.0

CRYPTO_SYMBOL_BLACKLIST = {
    "AVAXUSDT", "LTCUSDT", "BTCUSDT", "ETCUSDT",
    "XLMUSDT", "NEARUSDT", "XRPUSDT", "ARBUSDT", "LINKUSDT",
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BacktestTrade:
    """A single simulated trade."""
    symbol: str
    market: str
    direction: str          # "bullish" / "bearish"
    entry_time: datetime
    entry_price: float
    sl_price: float
    tp_price: float
    sl_reason: str
    quality: int
    ob_top: float
    ob_bottom: float
    # Result
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None   # "TP", "SL", "TSL", "TIMEOUT"
    pnl_pct: float = 0.0
    pnl_inr: float = 0.0            # INR P&L (capital × leverage × pnl_pct/100)
    # Signal metadata
    bias_direction: str = ""
    bias_reason: str = ""
    mss_time: Optional[datetime] = None
    ob_time: Optional[datetime] = None
    # Phase timeline (for detailed trade view)
    sweep_time: Optional[datetime] = None        # 4H sweep candle timestamp
    swept_level_time: Optional[datetime] = None  # Candle that was swept (source)
    tap_time: Optional[datetime] = None          # 5M tap bar timestamp
    # Filter info
    ema_detail: str = ""
    killzone: str = ""
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class BacktestResult:
    """Aggregate backtesting results."""
    symbol: str
    start: datetime
    end: datetime
    trades: List[BacktestTrade]
    signals_found: int = 0
    signals_rejected: int = 0

    @property
    def executed_trades(self) -> List[BacktestTrade]:
        return [t for t in self.trades if not t.rejected]

    @property
    def winning_trades(self) -> List[BacktestTrade]:
        # Count any profitable close as a win (including trailing-SL positive exits).
        return [t for t in self.executed_trades if t.pnl_inr > 0]

    @property
    def losing_trades(self) -> List[BacktestTrade]:
        return [t for t in self.executed_trades if t.pnl_inr < 0]

    @property
    def win_rate(self) -> float:
        ex = self.executed_trades
        if not ex:
            return 0.0
        return len(self.winning_trades) / len(ex) * 100

    @property
    def total_pnl_pct(self) -> float:
        return sum(t.pnl_pct for t in self.executed_trades)

    def summary(self) -> Dict[str, Any]:
        ex = self.executed_trades
        return {
            "symbol": self.symbol,
            "period": f"{self.start.strftime('%Y-%m-%d')} → {self.end.strftime('%Y-%m-%d')}",
            "signals_found": self.signals_found,
            "signals_rejected": self.signals_rejected,
            "trades_executed": len(ex),
            "wins": len(self.winning_trades),
            "losses": len(self.losing_trades),
            "win_rate": f"{self.win_rate:.1f}%",
            "total_pnl_pct": f"{self.total_pnl_pct:+.2f}%",
            "avg_pnl_pct": f"{self.total_pnl_pct / len(ex):+.2f}%" if ex else "N/A",
        }


# ============================================================================
# FILTERS (same logic as auto_trader.py)
# ============================================================================

def _is_jpy_pair(symbol: str) -> bool:
    return "JPY" in symbol.upper()


def _check_strict_ema(
    df_4h: pd.DataFrame, direction: str,
) -> Tuple[bool, str]:
    """Strict 4H EMA alignment (same as auto_trader)."""
    if df_4h is None or df_4h.empty:
        return False, "No 4H data"
    if len(df_4h) < MIN_4H_BARS_FOR_EMA:
        return False, f"Insufficient 4H history ({len(df_4h)} < {MIN_4H_BARS_FOR_EMA})"

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
            detail = f"ALIGNED: Close({last_close:.5f}) > EMA21({last_ema21:.5f}) > EMA50({last_ema50:.5f})"
        elif last_ema21 <= last_ema50:
            detail = f"COUNTER-TREND: Bullish but EMA21({last_ema21:.5f}) <= EMA50({last_ema50:.5f}) — trend is bearish"
        else:
            detail = f"WEAK ALIGNMENT: EMA21({last_ema21:.5f}) > EMA50({last_ema50:.5f}) but Close({last_close:.5f}) < EMA21 — price not confirming trend"
    else:
        passed = last_close < last_ema21 < last_ema50
        if passed:
            detail = f"ALIGNED: Close({last_close:.5f}) < EMA21({last_ema21:.5f}) < EMA50({last_ema50:.5f})"
        elif last_ema21 >= last_ema50:
            detail = f"COUNTER-TREND: Bearish but EMA21({last_ema21:.5f}) >= EMA50({last_ema50:.5f}) — trend is bullish"
        else:
            detail = f"WEAK ALIGNMENT: EMA21({last_ema21:.5f}) < EMA50({last_ema50:.5f}) but Close({last_close:.5f}) > EMA21 — price not confirming trend"

    return passed, detail


def _is_london_open_block(now_utc: datetime) -> bool:
    t = now_utc.hour * 60 + now_utc.minute
    return LONDON_OPEN_BLOCK_START_UTC <= t <= LONDON_OPEN_BLOCK_END_UTC


def _is_due_4h(cycle_ist: datetime) -> bool:
    """Same scheduler gate as auto_trader (_is_due_4h)."""
    return cycle_ist.minute == 31 and (cycle_ist.hour % 4) == 1


def _is_due_1h(cycle_ist: datetime) -> bool:
    """Same scheduler gate as auto_trader (_is_due_1h)."""
    return cycle_ist.minute == 31


def _drop_incomplete(
    df: pd.DataFrame,
    interval_min: int,
    now_utc: datetime,
    safety_s: int = 60,
) -> pd.DataFrame:
    """Drop the still-forming last candle — identical logic to auto_trader."""
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


def _reconstruct_forming_htf_bar(
    slice_htf: pd.DataFrame,
    slice_5m: pd.DataFrame,
    now_ts: pd.Timestamp,
    bar_delta: pd.Timedelta,
) -> pd.DataFrame:
    """OHLC for the still-forming HTF candle from aggregated closed 5m bars.

    Used only for **structure** detection (4H liquidity sweep, 1H MSS). The strict
    closed-bar slice stays separate for the 4H EMA filter — same split live terminals
    implicitly make when they chart forming HTF candles while indicators often use
    closed bars only.

    Applies to any symbol / timeframe length via ``bar_delta`` (4h, 1h, …).
    """
    out = slice_htf.copy()
    if out.empty or slice_5m is None or slice_5m.empty:
        return out
    last_idx = out.index[-1]
    if last_idx + bar_delta > now_ts:
        recent_5m = slice_5m[slice_5m.index >= last_idx]
        if not recent_5m.empty:
            out.loc[last_idx, "high"] = recent_5m["high"].max()
            out.loc[last_idx, "low"] = recent_5m["low"].min()
            out.loc[last_idx, "close"] = recent_5m["close"].iloc[-1]
            if "volume" in out.columns:
                out.loc[last_idx, "volume"] = recent_5m["volume"].sum()
    return out


def _bias_key(bias: DailyBias) -> str:
    ts = bias.sweep_timestamp
    iso = ts.isoformat() if ts is not None else "none"
    return f"{bias.direction.value}_{iso}"


def _is_fresh(tap_time, now_utc: datetime, window_min: int) -> bool:
    if tap_time is None:
        return False
    tt = tap_time
    if hasattr(tt, "to_pydatetime"):
        tt = tt.to_pydatetime()
    if tt.tzinfo is None:
        tt = UTC.localize(tt)
    tt = tt.astimezone(UTC)
    age = (now_utc.astimezone(UTC) - tt).total_seconds() / 60.0
    return age <= window_min


@dataclass
class _BiasTrack:
    """Stateful bias pipeline — mirrors auto_trader.BiasTrack."""

    key: str
    bias: DailyBias
    market: str
    created_utc: datetime
    expires_utc: datetime
    mss: Optional[MSSConfirmation] = None
    ob_entry: Optional[OBEntrySetup] = None
    alerted: bool = False
    completed: bool = False
    last_rejected_tap: Optional[datetime] = None


# ============================================================================
# SMART SL / TP (same logic as auto_trader.py)
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
    """Compute smart stop-loss — exact replica of auto_trader.compute_smart_sl."""
    # Step 1: Base SL at OB extreme
    if bias_direction == BiasType.BEARISH:
        base_sl = ob.top
        reason = f"OB high ({base_sl:.6f})"
    else:
        base_sl = ob.bottom
        reason = f"OB low ({base_sl:.6f})"

    # Step 2: Scan for worse extreme between OB and tap
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

    # Step 3: Add spread buffer
    if market == "CRYPTO":
        buffer = base_sl * (CRYPTO_SL_BUFFER_PCT / 100.0)
    else:
        if _is_jpy_pair(symbol):
            buffer = FOREX_SL_BUFFER_PIPS * 0.01
        else:
            buffer = FOREX_SL_BUFFER_PIPS * 0.0001

    if bias_direction == BiasType.BEARISH:
        sl = base_sl + buffer
    else:
        sl = base_sl - buffer

    reason += f" + {FOREX_SL_BUFFER_PIPS}pip buffer" if market == "FOREX" else f" + {CRYPTO_SL_BUFFER_PCT}% buffer"
    return sl, reason


FillModel = Literal["tap_bar_close", "strategy"]


def resolve_fill_price_and_time(
    slice_5m: pd.DataFrame,
    tap_bar_open: pd.Timestamp,
    bar_minutes: int = 5,
) -> Tuple[float, datetime]:
    """Broker-neutral fill (any symbol): price = tap candle **close**.

    Bar indices are candle **opens**. The position is assumed active from the
    **next** bar's open onward (same instant as the tap bar's close in this
    discrete model), which keeps ``simulate_trade`` aligned with OHLC paths.

    Note: real MT5 ticket times often fall inside the tap bar; without ticks,
    close-of-tap is the general OHLC-consistent execution price.
    """
    idx = slice_5m.index
    ts = pd.Timestamp(tap_bar_open)
    if idx.tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC).tz_convert(idx.tz)
        else:
            ts = ts.tz_convert(idx.tz)
    try:
        row = slice_5m.loc[ts]
    except KeyError:
        pos = int(idx.searchsorted(ts, side="right")) - 1
        if pos < 0 or pos >= len(idx):
            raise KeyError(f"tap bar {tap_bar_open} not in slice_5m") from None
        ts = idx[pos]
        row = slice_5m.iloc[pos]

    fill_px = float(row["close"])
    next_open = ts + pd.Timedelta(minutes=bar_minutes)
    fill_dt = next_open.to_pydatetime()
    if fill_dt.tzinfo is None:
        fill_dt = UTC.localize(fill_dt)
    else:
        fill_dt = fill_dt.astimezone(UTC)
    return fill_px, fill_dt


def compute_tp(entry_price: float, sl_price: float, rr: float = 1.5) -> float:
    """TP at rr × risk from entry."""
    risk = abs(entry_price - sl_price)
    if entry_price > sl_price:
        return entry_price + rr * risk
    else:
        return entry_price - rr * risk


# ============================================================================
# TRAILING STOP-LOSS (mirrored from auto_trader._get_trailing_target)
# ============================================================================

def _get_trailing_target(
    profit_inr: float,
    step_inr: float,
) -> Optional[Tuple[int, float, float]]:
    """Return trailing ladder info for the current unrealized profit.

    The ladder is dynamic from the initial margin (step_inr) used on the trade:
      - First trigger at 1.2x step profit -> SL locks 1.0x step
      - Then every +0.5x step profit     -> SL locks +0.5x step more

    Example for step=1000:
      Profit 1200 -> lock 1000
      Profit 1700 -> lock 1500
      Profit 2200 -> lock 2000
    """
    if step_inr <= 0 or profit_inr <= 0:
        return None

    first_trigger_inr = step_inr * 1.2
    if profit_inr < first_trigger_inr:
        return None

    ladder_increment_inr = step_inr * 0.5
    if ladder_increment_inr <= 0:
        return None

    extra_levels = int(math.floor(
        ((profit_inr - first_trigger_inr) / ladder_increment_inr) + 1e-9
    ))
    trail_level = 1 + max(0, extra_levels)
    trigger_inr = first_trigger_inr + extra_levels * ladder_increment_inr
    locked_inr = step_inr + extra_levels * ladder_increment_inr
    return trail_level, trigger_inr, locked_inr


# ============================================================================
# TRADE SIMULATION
# ============================================================================

def simulate_trade(
    df_5m: pd.DataFrame,
    entry_time: datetime,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    direction: str,
    max_hold_bars: int = 24 * 12,  # 24 hours of 5M bars
    capital_inr: float = DEFAULT_CAPITAL_INR,
    leverage: int = DEFAULT_LEVERAGE,
) -> Tuple[Optional[datetime], Optional[float], str, float, float]:
    """Walk forward through 5M bars to determine trade outcome.

    Trailing SL mirrors auto_trader._get_trailing_target:
      - step_inr = capital_inr (the margin used)
      - On each bar's close, compute unrealized INR profit
      - If the ladder triggers, tighten the SL (move toward entry+profit)

    Returns: (exit_time, exit_price, outcome, pnl_pct, pnl_inr)
    """
    if entry_time.tzinfo is None:
        entry_time = UTC.localize(entry_time)

    et = pd.Timestamp(entry_time)
    idx = df_5m.index
    if idx.tz is not None:
        et = (
            et.tz_localize(UTC).tz_convert(idx.tz)
            if et.tzinfo is None
            else et.tz_convert(idx.tz)
        )
    start_idx = idx.searchsorted(et, side="left")
    is_long = direction.lower() in ("bullish", "long")

    position_value = capital_inr * leverage
    step_inr = capital_inr
    # Hard loss cap (auto_trader parity): tighten SL so max per-trade loss is capped.
    # In the backtest this is enforced at entry with a hard INR cap.
    max_loss_inr = HARD_LOSS_CAP_INR
    if position_value > 0 and max_loss_inr > 0:
        max_loss_frac = max_loss_inr / position_value
        if is_long:
            capped_sl = entry_price * (1.0 - max_loss_frac)
            current_sl = max(sl_price, capped_sl)
        else:
            capped_sl = entry_price * (1.0 + max_loss_frac)
            current_sl = min(sl_price, capped_sl)
    else:
        current_sl = sl_price
    last_trail_level = 0

    for i in range(start_idx, min(start_idx + max_hold_bars, len(df_5m))):
        candle = df_5m.iloc[i]
        candle_time = df_5m.index[i]

        if is_long:
            if candle['low'] <= current_sl:
                pnl_pct = (current_sl - entry_price) / entry_price * 100
                pnl_inr = position_value * (current_sl - entry_price) / entry_price
                outcome = "TSL" if pnl_inr > 0 else "SL"
                return candle_time, current_sl, outcome, pnl_pct, pnl_inr
            if candle['high'] >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price * 100
                pnl_inr = position_value * (tp_price - entry_price) / entry_price
                return candle_time, tp_price, "TP", pnl_pct, pnl_inr

            # Trailing SL check on bar close
            close_profit_pct = (candle['close'] - entry_price) / entry_price
            close_profit_inr = position_value * close_profit_pct
            trail = _get_trailing_target(close_profit_inr, step_inr)
            if trail is not None:
                trail_level, _trigger, locked_inr = trail
                if trail_level > last_trail_level:
                    price_shift = locked_inr / position_value * entry_price
                    new_sl = entry_price + price_shift
                    if new_sl > current_sl:
                        current_sl = new_sl
                        last_trail_level = trail_level
        else:
            if candle['high'] >= current_sl:
                pnl_pct = (entry_price - current_sl) / entry_price * 100
                pnl_inr = position_value * (entry_price - current_sl) / entry_price
                outcome = "TSL" if pnl_inr > 0 else "SL"
                return candle_time, current_sl, outcome, pnl_pct, pnl_inr
            if candle['low'] <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price * 100
                pnl_inr = position_value * (entry_price - tp_price) / entry_price
                return candle_time, tp_price, "TP", pnl_pct, pnl_inr

            # Trailing SL check on bar close
            close_profit_pct = (entry_price - candle['close']) / entry_price
            close_profit_inr = position_value * close_profit_pct
            trail = _get_trailing_target(close_profit_inr, step_inr)
            if trail is not None:
                trail_level, _trigger, locked_inr = trail
                if trail_level > last_trail_level:
                    price_shift = locked_inr / position_value * entry_price
                    new_sl = entry_price - price_shift
                    if new_sl < current_sl:
                        current_sl = new_sl
                        last_trail_level = trail_level

    # Timeout — close at last available price
    if start_idx < len(df_5m):
        last = df_5m.iloc[min(start_idx + max_hold_bars - 1, len(df_5m) - 1)]
        last_time = df_5m.index[min(start_idx + max_hold_bars - 1, len(df_5m) - 1)]
        exit_price = last['close']
        if is_long:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_inr = position_value * (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            pnl_inr = position_value * (entry_price - exit_price) / entry_price
        return last_time, exit_price, "TIMEOUT", pnl_pct, pnl_inr

    return None, None, "NO_DATA", 0.0, 0.0


# ============================================================================
# MAIN BACKTESTER
# ============================================================================

def run_backtest(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    start: datetime,
    end: datetime,
    market: str = "FOREX",
    min_quality: int = 50,
    bias_lookback_hours: int = 72,
    fresh_window_minutes: int = FRESH_WINDOW_MINUTES_DEFAULT,
    bias_ttl_hours: float = BIAS_TTL_HOURS_DEFAULT,
    enable_ema_filter: bool = True,
    enable_killzone: bool = True,
    enable_london_block: bool = True,
    enable_blacklist: bool = True,
    verbose: bool = False,
    fill_model: FillModel = "tap_bar_close",
    intrabar_htf_bias_mss: bool = True,
) -> BacktestResult:
    """Run Strategy 09 backtest aligned with ``auto_trader`` replay rules.

    - Clock advances at **5m bar close** (no lookahead on the forming candle).
    - HTF / 5m data uses ``_drop_incomplete`` like live MT5 fetch (no synthetic
      partial 4H/1H/15m merges).
    - Stateful ``BiasTrack`` + ``last_rejected_tap`` match live tap skipping.
    - ``fill_model``: ``tap_bar_close`` (default) uses tap candle **close** and its
      **close time** as execution — OHLC-consistent on any symbol. ``strategy``
      uses OB-edge heuristic entry + poll clock (legacy / diagnostic).
    - ``intrabar_htf_bias_mss``: when True (default), Phase 1 sees **forming** 4H
      candles rebuilt from closed 5m (liquidity sweep timing aligns with live chart).
      Phase 2 MSS remains strictly on closed 1H candles, same as auto_trader.
      Strict **closed** 4H is still used **only** for the EMA filter.
    """
    strategy = MSSOrderBlockStrategy()
    result = BacktestResult(symbol=symbol, start=start, end=end, trades=[])

    if verbose:
        logger.info(
            f"  [{symbol}] replay: intrabar_htf_bias_mss={intrabar_htf_bias_mss} "
            f"fill_model={fill_model}"
        )

    if start.tzinfo is None:
        start = UTC.localize(start)
    if end.tzinfo is None:
        end = UTC.localize(end)

    for df in [df_4h, df_1h, df_15m, df_5m]:
        if df is not None and df.index.tz is None:
            df.index = df.index.tz_localize(UTC)

    target_5m = df_5m[(df_5m.index >= start) & (df_5m.index <= end)]
    if target_5m.empty:
        logger.warning(f"No 5M data in range {start}–{end}")
        return result

    tracks: Dict[str, _BiasTrack] = {}
    sent_ids: Set[str] = set()

    def _tap_ts_dt(ts) -> datetime:
        tt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if tt.tzinfo is None:
            tt = UTC.localize(tt)
        return tt.astimezone(UTC)

    first_run = True
    for step_idx in range(len(target_5m)):
        bar_open = target_5m.index[step_idx]
        # Mirror auto_trader's cycle trigger at :01/:06/:11... (IST): for each
        # 5m bar open, cycle time is bar_open + 1 minute.
        now_pd = bar_open + pd.Timedelta(minutes=1)
        now_utc_dt = now_pd.to_pydatetime()
        if now_utc_dt.tzinfo is None:
            now_utc_dt = UTC.localize(now_utc_dt)
        else:
            now_utc_dt = now_utc_dt.astimezone(UTC)

        now_ist = now_utc_dt.astimezone(IST)
        check_4h = _is_due_4h(now_ist) or first_run
        check_1h = _is_due_1h(now_ist) or first_run
        ts_cut = pd.Timestamp(now_utc_dt)
        if df_5m.index.tz is not None:
            ts_cut = ts_cut.tz_convert(df_5m.index.tz) if ts_cut.tz else ts_cut.tz_localize(UTC).tz_convert(df_5m.index.tz)

        slice_4h = _drop_incomplete(df_4h[df_4h.index <= ts_cut].copy(), 240, now_utc_dt)
        slice_1h = _drop_incomplete(df_1h[df_1h.index <= ts_cut].copy(), 60, now_utc_dt)
        slice_15m = _drop_incomplete(df_15m[df_15m.index <= ts_cut].copy(), 15, now_utc_dt)
        slice_5m = _drop_incomplete(df_5m[df_5m.index <= ts_cut].copy(), 5, now_utc_dt)

        if any(s is None or s.empty for s in [slice_4h, slice_1h, slice_15m, slice_5m]):
            continue

        # Phase 1: optional forming 4H for bias timing.
        # Phase 2 (1H MSS) remains closed-bar only like auto_trader.
        if intrabar_htf_bias_mss:
            slice_4h_bias = _reconstruct_forming_htf_bar(
                df_4h[df_4h.index <= ts_cut].copy(),
                slice_5m,
                ts_cut,
                pd.Timedelta(hours=4),
            )
        else:
            slice_4h_bias = slice_4h
        slice_1h_mss = slice_1h

        # Expire completed / TTL tracks
        dead = [k for k, tr in tracks.items() if tr.completed or now_utc_dt >= tr.expires_utc]
        for k in dead:
            del tracks[k]

        # ── Phase 1: 4H bias ───────────────────────────────────────
        if check_4h:
            biases = strategy.determine_bias(slice_4h_bias, bias_lookback_hours)
            for b in biases:
                k = _bias_key(b)
                if k not in tracks:
                    tracks[k] = _BiasTrack(
                        key=k,
                        bias=b,
                        market=market,
                        created_utc=now_utc_dt,
                        expires_utc=now_utc_dt + timedelta(hours=bias_ttl_hours),
                    )

        if not tracks:
            first_run = False
            continue

        # ── Phase 2: 1H MSS ───────────────────────────────────────
        if check_1h:
            for tr in tracks.values():
                if tr.mss is None:
                    tr.mss = strategy.confirm_mss(slice_1h_mss, tr.bias)

        # ── Phase 3+4: OB + tap (per track) ───────────────────────
        for tr in tracks.values():
            if tr.completed or tr.alerted:
                continue
            if tr.mss is None:
                continue

            df_5m_use = slice_5m
            if tr.last_rejected_tap is not None:
                cutoff = tr.last_rejected_tap
                if cutoff.tzinfo is None:
                    cutoff = UTC.localize(cutoff)
                co_ts = pd.Timestamp(cutoff)
                if slice_5m.index.tz is not None:
                    co_ts = co_ts.tz_convert(slice_5m.index.tz)
                df_5m_use = slice_5m[slice_5m.index > co_ts]
                if df_5m_use.empty:
                    continue

            ob_entry = strategy.find_ob_entry(slice_15m, df_5m_use, tr.bias, tr.mss)
            if ob_entry is None:
                continue

            direction_str = tr.bias.direction.value
            result.signals_found += 1

            quality = strategy.calculate_quality_score(tr.bias, tr.mss, ob_entry)
            if quality < min_quality:
                result.signals_rejected += 1
                tr.last_rejected_tap = _tap_ts_dt(ob_entry.timestamp)
                if verbose:
                    logger.info(
                        f"  [{now_ist:%H:%M}] {symbol} {direction_str} REJECTED: "
                        f"quality {quality} < {min_quality}"
                    )
                continue

            if not _is_fresh(ob_entry.timestamp, now_utc_dt, fresh_window_minutes):
                result.signals_rejected += 1
                tr.last_rejected_tap = _tap_ts_dt(ob_entry.timestamp)
                if verbose:
                    logger.info(
                        f"  [{now_ist:%H:%M}] {symbol} {direction_str} STALE TAP "
                        f"(>{fresh_window_minutes}m)"
                    )
                continue

            try:
                if fill_model == "tap_bar_close":
                    fill_px, fill_time_utc = resolve_fill_price_and_time(
                        slice_5m, pd.Timestamp(ob_entry.timestamp)
                    )
                else:
                    fill_px = float(ob_entry.entry_price)
                    fill_time_utc = now_utc_dt
            except KeyError as e:
                logger.warning(f"{symbol}: could not resolve fill for tap {ob_entry.timestamp}: {e}")
                continue

            ema_detail = ""
            if enable_ema_filter:
                ema_ok, ema_detail = _check_strict_ema(slice_4h, direction_str)
                if not ema_ok:
                    result.signals_rejected += 1
                    tr.last_rejected_tap = _tap_ts_dt(ob_entry.timestamp)
                    result.trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            market=market,
                            direction=direction_str,
                            entry_time=fill_time_utc,
                            entry_price=fill_px,
                            sl_price=0,
                            tp_price=0,
                            sl_reason="",
                            quality=quality,
                            ob_top=ob_entry.order_block.top,
                            ob_bottom=ob_entry.order_block.bottom,
                            rejected=True,
                            reject_reason=f"EMA: {ema_detail}",
                            ema_detail=ema_detail,
                        )
                    )
                    if verbose:
                        logger.info(
                            f"  [{now_ist:%H:%M}] {symbol} {direction_str} EMA REJECT: {ema_detail}"
                        )
                    continue

            if enable_blacklist and market == "CRYPTO" and symbol in CRYPTO_SYMBOL_BLACKLIST:
                result.signals_rejected += 1
                result.trades.append(
                    BacktestTrade(
                        symbol=symbol,
                        market=market,
                        direction=direction_str,
                        entry_time=fill_time_utc,
                        entry_price=fill_px,
                        sl_price=0,
                        tp_price=0,
                        sl_reason="",
                        quality=quality,
                        ob_top=ob_entry.order_block.top,
                        ob_bottom=ob_entry.order_block.bottom,
                        rejected=True,
                        reject_reason="Blacklisted symbol",
                    )
                )
                continue

            kz_detail = ""
            if enable_killzone:
                kz_result = check_killzone(now_ist, market=market, allow_asian=(market == "CRYPTO"))
                kz_detail = kz_result.reason
                if not kz_result.passed:
                    result.signals_rejected += 1
                    result.trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            market=market,
                            direction=direction_str,
                            entry_time=fill_time_utc,
                            entry_price=fill_px,
                            sl_price=0,
                            tp_price=0,
                            sl_reason="",
                            quality=quality,
                            ob_top=ob_entry.order_block.top,
                            ob_bottom=ob_entry.order_block.bottom,
                            rejected=True,
                            reject_reason=f"Killzone: {kz_detail}",
                            killzone=kz_detail,
                        )
                    )
                    if verbose:
                        logger.info(
                            f"  [{now_ist:%H:%M}] {symbol} {direction_str} KZ REJECT: {kz_detail}"
                        )
                    continue

            if enable_london_block and _is_london_open_block(now_utc_dt):
                result.signals_rejected += 1
                result.trades.append(
                    BacktestTrade(
                        symbol=symbol,
                        market=market,
                        direction=direction_str,
                        entry_time=fill_time_utc,
                        entry_price=fill_px,
                        sl_price=0,
                        tp_price=0,
                        sl_reason="",
                        quality=quality,
                        ob_top=ob_entry.order_block.top,
                        ob_bottom=ob_entry.order_block.bottom,
                        rejected=True,
                        reject_reason="London Open block (07:00-08:00 UTC)",
                    )
                )
                continue

            sig = MSSOB_Signal(
                index=0,
                datetime=ob_entry.timestamp,
                direction=(
                    Direction.BULLISH if tr.bias.direction == BiasType.BULLISH else Direction.BEARISH
                ),
                entry_price=fill_px,
                stop_loss=ob_entry.stop_loss,
                take_profit=ob_entry.tp_price,
                risk_reward=ob_entry.risk_reward,
                signal_type="MSS_OB_ENTRY",
                symbol=symbol,
                daily_bias=tr.bias,
                mss_confirmation=tr.mss,
                ob_entry=ob_entry,
                quality_score=quality,
            )
            sig_json = format_signal_for_jsonl(sig)
            sig_json["market"] = market
            sid = "|".join(
                [
                    str(sig_json.get("symbol", "")),
                    str(sig_json.get("direction", "")),
                    str(sig_json.get("signal_datetime_ist", "")),
                ]
            )
            if sid in sent_ids:
                tr.alerted = True
                tr.completed = True
                continue

            sl_price, sl_reason = compute_smart_sl(
                ob=ob_entry.order_block,
                bias_direction=tr.bias.direction,
                df_5m=slice_5m,
                tap_time=ob_entry.timestamp,
                entry_price=fill_px,
                market=market,
                symbol=symbol,
            )
            tp_price = compute_tp(fill_px, sl_price, rr=1.5)

            if tr.bias.direction == BiasType.BEARISH:
                if sl_price <= fill_px:
                    continue
            else:
                if sl_price >= fill_px:
                    continue

            exit_time, exit_price, outcome, pnl_pct, pnl_inr = simulate_trade(
                df_5m,
                fill_time_utc,
                fill_px,
                sl_price,
                tp_price,
                direction_str,
            )

            result.trades.append(
                BacktestTrade(
                    symbol=symbol,
                    market=market,
                    direction=direction_str,
                    entry_time=fill_time_utc,
                    entry_price=fill_px,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_reason=sl_reason,
                    quality=quality,
                    ob_top=ob_entry.order_block.top,
                    ob_bottom=ob_entry.order_block.bottom,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    outcome=outcome,
                    pnl_pct=pnl_pct,
                    pnl_inr=pnl_inr,
                    bias_direction=direction_str,
                    bias_reason=tr.bias.reason,
                    mss_time=tr.mss.timestamp,
                    ob_time=ob_entry.order_block.datetime,
                    sweep_time=tr.bias.sweep_timestamp,
                    swept_level_time=tr.bias.source_timestamp,
                    tap_time=ob_entry.timestamp,
                    ema_detail=ema_detail,
                    killzone=kz_detail,
                )
            )
            sent_ids.add(sid)
            tr.alerted = True
            tr.completed = True

            if verbose:
                logger.info(
                    f"  [{now_ist:%H:%M}] TRADE: {symbol} {direction_str.upper()} "
                    f"entry={fill_px:.5f} @ {fill_time_utc.astimezone(IST):%H:%M} IST "
                    f"SL={sl_price:.5f} TP={tp_price:.5f} "
                    f"Q={quality} → {outcome} ({pnl_pct:+.3f}%)"
                )

        first_run = False

    return result
