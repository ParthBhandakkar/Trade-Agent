"""
Entry Filters for Strategy 09 — MSS + Order Block
====================================================

Confluence filters to REJECT low-probability signals before execution.
Based on analysis of 25 live trades where direction was correct only 48%.

These filters use data already available in the auto_trader loop
(df_4h, df_1h, df_15m, df_5m) — no additional API calls needed.

Filters implemented:
  1. EMA Trend Alignment  (4H)  — CRITICAL, rejects counter-trend entries
  2. ADX Momentum          (1H)  — rejects choppy/ranging markets
  3. Displacement Check    (1H)  — verifies MSS move was impulsive
  4. Market Structure      (1H)  — confirms HH/HL or LH/LL
  5. Killzone Session      (IST) — only trade London/NY high-volume hours
  6. FVG Confluence        (15M) — checks for supporting FVG near entry
  7. RSI Divergence Guard  (1H)  — blocks entries with hidden divergence
"""

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FilterResult:
    """Result of a single filter check."""
    name: str
    passed: bool
    reason: str
    score_bonus: int = 0  # Extra quality points if passed strongly


@dataclass
class FilterVerdict:
    """Aggregate result of all filters."""
    passed: bool
    filters_passed: int
    filters_failed: int
    total_filters: int
    rejection_reason: str  # First failing filter's reason
    results: list  # List[FilterResult]
    confluence_score: int  # 0-100, how many supporting factors


# ============================================================================
# INDIVIDUAL FILTERS
# ============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_adx(
    df: pd.DataFrame, period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, +DI, -DI."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx, plus_di, minus_di


# --------------------------------------------------------------------------
# FILTER 1: EMA Trend Alignment (4H) — MOST CRITICAL
# --------------------------------------------------------------------------

def check_ema_trend_alignment(
    signal_direction: str,
    df_4h: pd.DataFrame,
    ema_fast: int = 21,
    ema_slow: int = 50,
) -> FilterResult:
    """
    Check if signal direction aligns with 4H EMA trend.

    Bullish signal requires:
      - EMA21 > EMA50 (uptrend)
      - Price above EMA21 (momentum)

    Bearish signal requires:
      - EMA21 < EMA50 (downtrend)
      - Price below EMA21 (momentum)

    From Strategy 08 analysis: 100% of winning trades were trend-aligned.
    This is the single most powerful filter.
    """
    name = "EMA Trend (4H)"

    if df_4h is None or len(df_4h) < ema_slow + 5:
        return FilterResult(name, True, "Insufficient 4H data — allowing")

    ema_f = calculate_ema(df_4h['close'], ema_fast)
    ema_s = calculate_ema(df_4h['close'], ema_slow)

    price = df_4h['close'].iloc[-1]
    fast_val = ema_f.iloc[-1]
    slow_val = ema_s.iloc[-1]
    sig = signal_direction.upper()

    # Check EMA slope (last 3 candles = 12 hours)
    ema_slope = (ema_f.iloc[-1] - ema_f.iloc[-4]) / ema_f.iloc[-4] * 100

    if sig in ("BULLISH", "LONG", "BUY"):
        if fast_val > slow_val and price > fast_val:
            bonus = 10 if ema_slope > 0.1 else 5
            return FilterResult(
                name, True,
                f"ALIGNED: Price({price:.5f}) > EMA21({fast_val:.5f}) > EMA50({slow_val:.5f}), slope={ema_slope:+.3f}%",
                score_bonus=bonus,
            )
        elif fast_val > slow_val:
            # Uptrend but price below EMA — weak alignment
            return FilterResult(
                name, True,
                f"WEAK: EMA21 > EMA50 but price below EMA21 (pullback zone)",
                score_bonus=2,
            )
        else:
            return FilterResult(
                name, False,
                f"COUNTER-TREND: BULLISH signal but EMA21({fast_val:.5f}) < EMA50({slow_val:.5f})",
            )

    elif sig in ("BEARISH", "SHORT", "SELL"):
        if fast_val < slow_val and price < fast_val:
            bonus = 10 if ema_slope < -0.1 else 5
            return FilterResult(
                name, True,
                f"ALIGNED: Price({price:.5f}) < EMA21({fast_val:.5f}) < EMA50({slow_val:.5f}), slope={ema_slope:+.3f}%",
                score_bonus=bonus,
            )
        elif fast_val < slow_val:
            return FilterResult(
                name, True,
                f"WEAK: EMA21 < EMA50 but price above EMA21 (pullback zone)",
                score_bonus=2,
            )
        else:
            return FilterResult(
                name, False,
                f"COUNTER-TREND: BEARISH signal but EMA21({fast_val:.5f}) > EMA50({slow_val:.5f})",
            )

    return FilterResult(name, True, "Unknown direction — allowing")


# --------------------------------------------------------------------------
# FILTER 2: ADX Momentum (1H)
# --------------------------------------------------------------------------

def check_adx_momentum(
    signal_direction: str,
    df_1h: pd.DataFrame,
    min_adx: float = 20.0,
    period: int = 14,
) -> FilterResult:
    """
    Check ADX for trend strength and DI direction alignment.

    - ADX < 20: Choppy/ranging → REJECT (false MSS signals in chop)
    - ADX 20-25: Developing trend → careful
    - ADX > 25: Confirmed trend → good
    - ADX > 35: Very strong trend → excellent

    Also verifies +DI/-DI aligns with signal direction.
    """
    name = "ADX Momentum (1H)"

    if df_1h is None or len(df_1h) < period + 10:
        return FilterResult(name, True, "Insufficient 1H data — allowing")

    adx, plus_di, minus_di = calculate_adx(df_1h, period)

    current_adx = adx.iloc[-1]
    pdi = plus_di.iloc[-1]
    mdi = minus_di.iloc[-1]

    sig = signal_direction.upper()

    # Check minimum ADX
    if current_adx < min_adx:
        return FilterResult(
            name, False,
            f"CHOPPY: ADX={current_adx:.1f} < {min_adx} (ranging market, MSS is noise)",
        )

    # Check DI alignment
    if sig in ("BULLISH", "LONG", "BUY"):
        if pdi < mdi:
            return FilterResult(
                name, False,
                f"DI MISMATCH: BULL signal but -DI({mdi:.1f}) > +DI({pdi:.1f}), ADX={current_adx:.1f}",
            )
        bonus = 8 if current_adx > 30 else 3
        return FilterResult(
            name, True,
            f"MOMENTUM OK: ADX={current_adx:.1f}, +DI={pdi:.1f} > -DI={mdi:.1f}",
            score_bonus=bonus,
        )

    elif sig in ("BEARISH", "SHORT", "SELL"):
        if mdi < pdi:
            return FilterResult(
                name, False,
                f"DI MISMATCH: BEAR signal but +DI({pdi:.1f}) > -DI({mdi:.1f}), ADX={current_adx:.1f}",
            )
        bonus = 8 if current_adx > 30 else 3
        return FilterResult(
            name, True,
            f"MOMENTUM OK: ADX={current_adx:.1f}, -DI={mdi:.1f} > +DI={pdi:.1f}",
            score_bonus=bonus,
        )

    return FilterResult(name, True, "Unknown direction — allowing")


# --------------------------------------------------------------------------
# FILTER 3: MSS Displacement Check (1H)
# --------------------------------------------------------------------------

def check_displacement(
    signal_direction: str,
    df_1h: pd.DataFrame,
    mss_timestamp,
    min_body_pct: float = 0.15,
) -> FilterResult:
    """
    Verify the 1H MSS candle was an impulsive (displacement) move,
    not a weak/grinding break that often leads to fakeouts.

    Displacement = large-body candle (body ≥ min_body_pct% of price).

    A real structure break by institutional order flow produces
    a large, decisive candle. Weak breaks are retail noise.
    """
    name = "Displacement (1H)"

    if df_1h is None or len(df_1h) < 5:
        return FilterResult(name, True, "Insufficient data — allowing")

    # Find the MSS candle
    ts = mss_timestamp
    if hasattr(ts, 'to_pydatetime'):
        ts = ts.to_pydatetime()

    mss_idx = df_1h.index.searchsorted(ts)
    if mss_idx >= len(df_1h):
        mss_idx = len(df_1h) - 1

    candle = df_1h.iloc[mss_idx]
    body = abs(candle['close'] - candle['open'])
    body_pct = body / candle['close'] * 100

    # Also check range vs ATR
    atr = calculate_atr(df_1h, 14)
    if mss_idx < len(atr):
        current_atr = atr.iloc[mss_idx]
        range_size = candle['high'] - candle['low']
        atr_ratio = range_size / current_atr if current_atr > 0 else 0
    else:
        atr_ratio = 1.0

    if body_pct < min_body_pct:
        return FilterResult(
            name, False,
            f"WEAK MSS: Body={body_pct:.3f}% < {min_body_pct}% (no displacement, likely fakeout)",
        )

    # Strong displacement
    if body_pct > min_body_pct * 2 and atr_ratio > 1.2:
        return FilterResult(
            name, True,
            f"STRONG displacement: Body={body_pct:.3f}%, ATR ratio={atr_ratio:.2f}x",
            score_bonus=8,
        )

    return FilterResult(
        name, True,
        f"OK displacement: Body={body_pct:.3f}%, ATR ratio={atr_ratio:.2f}x",
        score_bonus=3,
    )


# --------------------------------------------------------------------------
# FILTER 4: 1H Market Structure Confirmation
# --------------------------------------------------------------------------

def check_market_structure(
    signal_direction: str,
    df_1h: pd.DataFrame,
    lookback: int = 30,
) -> FilterResult:
    """
    Confirm market structure aligns with signal:
      - Bullish → recent Higher Highs + Higher Lows
      - Bearish → recent Lower Highs + Lower Lows

    Uses swing point detection on last `lookback` candles.
    """
    name = "Market Structure (1H)"

    if df_1h is None or len(df_1h) < lookback + 5:
        return FilterResult(name, True, "Insufficient data — allowing")

    recent = df_1h.tail(lookback)
    highs, lows = [], []

    for i in range(2, len(recent) - 2):
        h = recent.iloc[i]['high']
        l = recent.iloc[i]['low']
        if (h > recent.iloc[i - 1]['high'] and h > recent.iloc[i - 2]['high'] and
                h > recent.iloc[i + 1]['high'] and h > recent.iloc[i + 2]['high']):
            highs.append(h)
        if (l < recent.iloc[i - 1]['low'] and l < recent.iloc[i - 2]['low'] and
                l < recent.iloc[i + 1]['low'] and l < recent.iloc[i + 2]['low']):
            lows.append(l)

    if len(highs) < 2 or len(lows) < 2:
        return FilterResult(name, True, "Not enough swing points — neutral")

    sig = signal_direction.upper()

    if sig in ("BULLISH", "LONG", "BUY"):
        hh = highs[-1] > highs[-2]
        hl = lows[-1] > lows[-2]
        if hh and hl:
            return FilterResult(name, True, "BULLISH structure: HH + HL confirmed", score_bonus=5)
        elif hl:
            return FilterResult(name, True, "Bullish structure: HL confirmed", score_bonus=2)
        else:
            return FilterResult(
                name, False,
                f"NO BULLISH STRUCTURE: Need HH/HL but got H={highs[-2:]}, L={lows[-2:]}",
            )
    else:
        lh = highs[-1] < highs[-2]
        ll = lows[-1] < lows[-2]
        if lh and ll:
            return FilterResult(name, True, "BEARISH structure: LH + LL confirmed", score_bonus=5)
        elif lh:
            return FilterResult(name, True, "Bearish structure: LH confirmed", score_bonus=2)
        else:
            return FilterResult(
                name, False,
                f"NO BEARISH STRUCTURE: Need LH/LL but got H={highs[-2:]}, L={lows[-2:]}",
            )


# --------------------------------------------------------------------------
# FILTER 5: Killzone / Session Filter
# --------------------------------------------------------------------------

# ICT Killzones in IST (UTC+5:30)
KILLZONES_IST = {
    "london_open":    (13, 0,  17, 0),   # 07:30-11:30 UTC → 13:00-17:00 IST
    "ny_open":        (18, 30, 22, 30),   # 13:00-17:00 UTC → 18:30-22:30 IST
    "london_ny_overlap": (18, 30, 21, 0), # Best: 13:00-15:30 UTC → 18:30-21:00 IST
    "asian_session":  (5, 30,  9, 30),    # 00:00-04:00 UTC → 05:30-09:30 IST
}


def check_killzone(
    now_ist: datetime,
    market: str = "FOREX",
    allow_asian: bool = False,
) -> FilterResult:
    """
    Check if current time is within an ICT killzone.

    For FOREX: Only trade during London or NY session.
    Avoid Asian session (low volume → fake moves).
    Best entries: London/NY overlap (18:30-21:00 IST).

    For CRYPTO: Same killzone windows apply — 58% of crypto
    trades fall in the Dead Zone and are net negative.
    Asian session allowed for crypto (profitable in backtest).
    """
    name = "Killzone Session"

    # Crypto treats Asian as always allowed (profitable in backtest)
    effective_allow_asian = allow_asian or (market == "CRYPTO")

    h, m = now_ist.hour, now_ist.minute
    t = h * 60 + m

    # Check London/NY overlap (best)
    overlap_start = 18 * 60 + 30
    overlap_end = 21 * 60
    if overlap_start <= t <= overlap_end:
        return FilterResult(
            name, True,
            f"LONDON/NY OVERLAP ({h:02d}:{m:02d} IST) — highest probability window",
            score_bonus=10,
        )

    # Check London session
    london_start = 13 * 60
    london_end = 17 * 60
    if london_start <= t <= london_end:
        return FilterResult(
            name, True,
            f"LONDON SESSION ({h:02d}:{m:02d} IST) — good liquidity",
            score_bonus=5,
        )

    # Check NY session
    ny_start = 18 * 60 + 30
    ny_end = 22 * 60 + 30
    if ny_start <= t <= ny_end:
        return FilterResult(
            name, True,
            f"NY SESSION ({h:02d}:{m:02d} IST) — good liquidity",
            score_bonus=5,
        )

    # Asian session
    asian_start = 5 * 60 + 30
    asian_end = 9 * 60 + 30
    if asian_start <= t <= asian_end:
        if effective_allow_asian:
            return FilterResult(
                name, True,
                f"ASIAN SESSION ({h:02d}:{m:02d} IST) — allowed but lower probability",
                score_bonus=0,
            )
        return FilterResult(
            name, False,
            f"ASIAN SESSION ({h:02d}:{m:02d} IST) — low volume, fake moves, SKIP",
        )

    # Dead zone (outside all killzones)
    return FilterResult(
        name, False,
        f"DEAD ZONE ({h:02d}:{m:02d} IST) — no institutional activity, SKIP",
    )


# --------------------------------------------------------------------------
# FILTER 6: RSI Divergence Guard (1H)
# --------------------------------------------------------------------------

def check_rsi_divergence(
    signal_direction: str,
    df_1h: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = 20,
) -> FilterResult:
    """
    Guard against RSI divergence — a reliable reversal warning.

    Bullish signal rejected if: Price making higher highs but RSI lower
    (bearish divergence = momentum exhaustion, likely reversal DOWN).

    Bearish signal rejected if: Price making lower lows but RSI higher
    (bullish divergence = momentum exhaustion, likely reversal UP).
    """
    name = "RSI Divergence (1H)"

    if df_1h is None or len(df_1h) < rsi_period + lookback:
        return FilterResult(name, True, "Insufficient data — allowing")

    rsi = calculate_rsi(df_1h['close'], rsi_period)
    recent = df_1h.tail(lookback)
    recent_rsi = rsi.tail(lookback)

    # Find last 2 swing highs and lows
    highs_price, highs_rsi = [], []
    lows_price, lows_rsi = [], []

    for i in range(2, len(recent) - 2):
        h = recent.iloc[i]['high']
        l = recent.iloc[i]['low']
        r = recent_rsi.iloc[i]

        if (h > recent.iloc[i - 1]['high'] and h > recent.iloc[i - 2]['high'] and
                h > recent.iloc[i + 1]['high'] and h > recent.iloc[i + 2]['high']):
            highs_price.append(h)
            highs_rsi.append(r)

        if (l < recent.iloc[i - 1]['low'] and l < recent.iloc[i - 2]['low'] and
                l < recent.iloc[i + 1]['low'] and l < recent.iloc[i + 2]['low']):
            lows_price.append(l)
            lows_rsi.append(r)

    sig = signal_direction.upper()

    if sig in ("BULLISH", "LONG", "BUY") and len(highs_price) >= 2:
        # Check for bearish divergence (price HH but RSI LH → reversal coming)
        if highs_price[-1] > highs_price[-2] and highs_rsi[-1] < highs_rsi[-2]:
            return FilterResult(
                name, False,
                f"BEARISH DIVERGENCE: Price HH but RSI LH ({highs_rsi[-2]:.0f}→{highs_rsi[-1]:.0f}) — reversal likely",
            )

    elif sig in ("BEARISH", "SHORT", "SELL") and len(lows_price) >= 2:
        # Check for bullish divergence (price LL but RSI HL → reversal coming)
        if lows_price[-1] < lows_price[-2] and lows_rsi[-1] > lows_rsi[-2]:
            return FilterResult(
                name, False,
                f"BULLISH DIVERGENCE: Price LL but RSI HL ({lows_rsi[-2]:.0f}→{lows_rsi[-1]:.0f}) — reversal likely",
            )

    # Also check for extreme RSI (overbought/oversold AGAINST signal)
    current_rsi = rsi.iloc[-1]
    if sig in ("BULLISH", "LONG", "BUY") and current_rsi > 75:
        return FilterResult(
            name, False,
            f"OVERBOUGHT: RSI={current_rsi:.0f} — buying into exhaustion",
        )
    if sig in ("BEARISH", "SHORT", "SELL") and current_rsi < 25:
        return FilterResult(
            name, False,
            f"OVERSOLD: RSI={current_rsi:.0f} — selling into exhaustion",
        )

    return FilterResult(name, True, f"No divergence, RSI={current_rsi:.0f}", score_bonus=2)


# --------------------------------------------------------------------------
# FILTER 7: FVG Confluence (15M)
# --------------------------------------------------------------------------

def check_fvg_confluence(
    signal_direction: str,
    df_15m: pd.DataFrame,
    entry_price: float,
    max_distance_pct: float = 0.3,
) -> FilterResult:
    """
    Check if there's an unmitigated Fair Value Gap (FVG) supporting the entry.

    An FVG near the Order Block entry price adds confluence —
    two institutional footprints pointing the same direction.

    Bullish: Need bullish FVG (gap up) near/below entry price
    Bearish: Need bearish FVG (gap down) near/above entry price
    """
    name = "FVG Confluence (15M)"

    if df_15m is None or len(df_15m) < 20:
        return FilterResult(name, True, "Insufficient data — allowing")

    sig = signal_direction.upper()

    # Simple FVG detection on last 60 candles (15 hours)
    lookback_start = max(0, len(df_15m) - 60)
    segment = df_15m.iloc[lookback_start:]

    supporting_fvgs = 0

    for i in range(2, len(segment)):
        c1 = segment.iloc[i - 2]
        c3 = segment.iloc[i]

        if sig in ("BULLISH", "LONG", "BUY"):
            # Bullish FVG: candle3.low > candle1.high (gap up)
            if c3['low'] > c1['high']:
                fvg_mid = (c3['low'] + c1['high']) / 2
                dist_pct = abs(fvg_mid - entry_price) / entry_price * 100
                if dist_pct <= max_distance_pct:
                    supporting_fvgs += 1
        else:
            # Bearish FVG: candle1.low > candle3.high (gap down)
            if c1['low'] > c3['high']:
                fvg_mid = (c1['low'] + c3['high']) / 2
                dist_pct = abs(fvg_mid - entry_price) / entry_price * 100
                if dist_pct <= max_distance_pct:
                    supporting_fvgs += 1

    if supporting_fvgs > 0:
        return FilterResult(
            name, True,
            f"{supporting_fvgs} supporting FVG(s) near entry — strong confluence",
            score_bonus=7,
        )

    # No FVG confluence — this is a soft filter (doesn't reject)
    return FilterResult(
        name, True,
        "No FVG confluence near entry (soft — not rejecting)",
        score_bonus=0,
    )


# ============================================================================
# MASTER FILTER: Apply All
# ============================================================================

def apply_entry_filters(
    signal_direction: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    entry_price: float,
    mss_timestamp=None,
    now_ist: datetime = None,
    market: str = "FOREX",
    # ── Filter toggles ───────────────────────────────
    require_ema_trend: bool = True,
    require_adx: bool = True,
    require_displacement: bool = True,
    require_structure: bool = False,   # Soft by default
    require_killzone: bool = True,
    check_rsi: bool = True,
    check_fvg: bool = False,          # Bonus only, not required
    # ── Thresholds ────────────────────────────────────
    min_adx: float = 20.0,
    min_displacement_pct: float = 0.15,
    allow_asian: bool = False,
    debug: bool = False,
) -> FilterVerdict:
    """
    Apply all entry filters and return a verdict.

    Hard filters (reject if failed): EMA Trend, ADX, Displacement, Killzone
    Soft filters (bonus only): Market Structure, FVG Confluence
    Guard filters (reject on warning): RSI Divergence

    Returns FilterVerdict with:
      - passed: True if all hard filters pass
      - confluence_score: 0-100 total bonus from passing filters
      - rejection_reason: why it failed (if applicable)
    """
    results: List[FilterResult] = []
    failed = False
    rejection_reason = ""

    # ── FILTER 1: EMA Trend (CRITICAL) ────────────────
    if require_ema_trend:
        f1 = check_ema_trend_alignment(signal_direction, df_4h)
        results.append(f1)
        if not f1.passed:
            failed = True
            rejection_reason = f1.reason
            if debug:
                logger.info(f"    ❌ {f1.name}: {f1.reason}")
            # Early exit — counter-trend is always bad
            return FilterVerdict(
                passed=False,
                filters_passed=0,
                filters_failed=1,
                total_filters=1,
                rejection_reason=rejection_reason,
                results=results,
                confluence_score=0,
            )
        elif debug:
            logger.info(f"    ✅ {f1.name}: {f1.reason}")

    # ── FILTER 2: ADX Momentum ────────────────────────
    if require_adx:
        f2 = check_adx_momentum(signal_direction, df_1h, min_adx=min_adx)
        results.append(f2)
        if not f2.passed:
            failed = True
            rejection_reason = f2.reason
            if debug:
                logger.info(f"    ❌ {f2.name}: {f2.reason}")
            return FilterVerdict(
                passed=False,
                filters_passed=sum(1 for r in results if r.passed),
                filters_failed=sum(1 for r in results if not r.passed),
                total_filters=len(results),
                rejection_reason=rejection_reason,
                results=results,
                confluence_score=0,
            )
        elif debug:
            logger.info(f"    ✅ {f2.name}: {f2.reason}")

    # ── FILTER 3: Displacement ────────────────────────
    if require_displacement and mss_timestamp is not None:
        f3 = check_displacement(signal_direction, df_1h, mss_timestamp, min_displacement_pct)
        results.append(f3)
        if not f3.passed:
            failed = True
            rejection_reason = f3.reason
            if debug:
                logger.info(f"    ❌ {f3.name}: {f3.reason}")
            return FilterVerdict(
                passed=False,
                filters_passed=sum(1 for r in results if r.passed),
                filters_failed=sum(1 for r in results if not r.passed),
                total_filters=len(results),
                rejection_reason=rejection_reason,
                results=results,
                confluence_score=0,
            )
        elif debug:
            logger.info(f"    ✅ {f3.name}: {f3.reason}")

    # ── FILTER 4: Market Structure (soft) ─────────────
    if require_structure:
        f4 = check_market_structure(signal_direction, df_1h)
        results.append(f4)
        if not f4.passed:
            failed = True
            rejection_reason = f4.reason
            if debug:
                logger.info(f"    ❌ {f4.name}: {f4.reason}")
            return FilterVerdict(
                passed=False,
                filters_passed=sum(1 for r in results if r.passed),
                filters_failed=sum(1 for r in results if not r.passed),
                total_filters=len(results),
                rejection_reason=rejection_reason,
                results=results,
                confluence_score=0,
            )
        elif debug:
            logger.info(f"    ✅ {f4.name}: {f4.reason}")

    # ── FILTER 5: Killzone Session ────────────────────
    if require_killzone and now_ist is not None:
        f5 = check_killzone(now_ist, market, allow_asian)
        results.append(f5)
        if not f5.passed:
            failed = True
            rejection_reason = f5.reason
            if debug:
                logger.info(f"    ❌ {f5.name}: {f5.reason}")
            return FilterVerdict(
                passed=False,
                filters_passed=sum(1 for r in results if r.passed),
                filters_failed=sum(1 for r in results if not r.passed),
                total_filters=len(results),
                rejection_reason=rejection_reason,
                results=results,
                confluence_score=0,
            )
        elif debug:
            logger.info(f"    ✅ {f5.name}: {f5.reason}")

    # ── FILTER 6: RSI Divergence Guard ────────────────
    if check_rsi:
        f6 = check_rsi_divergence(signal_direction, df_1h)
        results.append(f6)
        if not f6.passed:
            failed = True
            rejection_reason = f6.reason
            if debug:
                logger.info(f"    ⚠️ {f6.name}: {f6.reason}")
            return FilterVerdict(
                passed=False,
                filters_passed=sum(1 for r in results if r.passed),
                filters_failed=sum(1 for r in results if not r.passed),
                total_filters=len(results),
                rejection_reason=rejection_reason,
                results=results,
                confluence_score=0,
            )
        elif debug:
            logger.info(f"    ✅ {f6.name}: {f6.reason}")

    # ── FILTER 7: FVG Confluence (bonus only) ─────────
    if check_fvg:
        f7 = check_fvg_confluence(signal_direction, df_15m, entry_price)
        results.append(f7)
        if debug:
            logger.info(f"    {'✅' if f7.passed else '⚠️'} {f7.name}: {f7.reason}")

    # ── Compute confluence score ──────────────────────
    total_bonus = sum(r.score_bonus for r in results if r.passed)
    confluence_score = min(100, total_bonus)
    n_passed = sum(1 for r in results if r.passed)
    n_failed = sum(1 for r in results if not r.passed)

    if debug:
        logger.info(
            f"    📊 FILTERS: {n_passed}/{len(results)} passed, "
            f"confluence={confluence_score}/100"
        )

    return FilterVerdict(
        passed=not failed,
        filters_passed=n_passed,
        filters_failed=n_failed,
        total_filters=len(results),
        rejection_reason=rejection_reason,
        results=results,
        confluence_score=confluence_score,
    )
