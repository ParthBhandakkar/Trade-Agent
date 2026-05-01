"""
Strategy 09: ICT MSS + Order Block Entry (Path B)
===================================================

A separate, standalone ICT strategy focusing on:
  Phase 1: 4H Bias from Liquidity Sweep (reused from Strategy 08)
  Phase 2: 1H Market Structure Shift (MSS) confirming bias direction
  Phase 3: 15M Order Block near OTE zone (62-79% Fib retracement) for entry
  Phase 4: Price taps the Order Block → enter with SL behind OB, TP at 2R

Key differences from Strategy 08 (Inversion FVG):
- Uses MSS on 1H instead of PDA order-flow confirmation
- Uses Order Block on 15M instead of FVG → 5M inversion trigger
- Simpler entry model, fewer conditions, potentially more trades
- Close 100% at 2R (no trailing)
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
import pytz

# Add parent directories to path for shared indicators
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.utils.indicators import (
    Direction,
    FVG, LiquidityLevel, MSS, OrderBlock, Signal,
    detect_fvg,
    detect_liquidity_levels,
    detect_liquidity_sweep,
    detect_mss,
    detect_order_block,
    calculate_fib_levels,
    is_bullish_candle,
    is_bearish_candle,
)

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

class BiasType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class DailyBias:
    """4H bias from liquidity sweep — reused concept from Strategy 08."""
    direction: BiasType
    sweep_level: LiquidityLevel       # The swept liquidity level
    sweep_index: int                   # Candle index of the sweep
    sweep_timestamp: pd.Timestamp
    confidence: int                    # 0-100
    reason: str
    source_timestamp: Optional[pd.Timestamp] = None
    event_timestamp: Optional[pd.Timestamp] = None


@dataclass
class MSSConfirmation:
    """1H Market Structure Shift confirming the bias."""
    mss: MSS
    direction: BiasType
    timestamp: pd.Timestamp
    details: str


@dataclass
class OBEntrySetup:
    """15M Order Block entry setup near OTE zone."""
    order_block: OrderBlock
    fib_level: float                  # Which fib level the OB is near (0.618-0.786)
    in_ote_zone: bool                 # Whether OB is within OTE zone
    entry_price: float                # Entry when price taps OB
    stop_loss: float                  # SL behind OB
    tp_price: float                   # TP at 2R
    risk_reward: float
    timestamp: pd.Timestamp


@dataclass
class MSSOB_Signal(Signal):
    """Full MSS + OB signal."""
    symbol: str
    daily_bias: DailyBias
    mss_confirmation: MSSConfirmation
    ob_entry: OBEntrySetup
    quality_score: int                # 0-100


# ============================================================================
# STRATEGY
# ============================================================================

class MSSOrderBlockStrategy:
    """
    ICT MSS + Order Block Strategy

    Phase 1: 4H Liquidity Sweep → Determines directional bias
    Phase 2: 1H MSS → Confirms structure shift in bias direction
    Phase 3: 15M Order Block → Find OB near OTE (0.618-0.786 fib) of the MSS move
    Phase 4: Entry when price taps OB body zone → SL behind OB, TP at 2R
    """

    def __init__(
        self,
        lookback_4h: int = 50,
        lookback_1h: int = 100,
        lookback_15m: int = 100,
        ote_fib_low: float = 0.618,
        ote_fib_high: float = 0.786,
        risk_reward_target: float = 2.0,
        max_ob_age_candles: int = 50,
    ):
        self.lookback_4h = lookback_4h
        self.lookback_1h = lookback_1h
        self.lookback_15m = lookback_15m
        self.ote_fib_low = ote_fib_low
        self.ote_fib_high = ote_fib_high
        self.risk_reward_target = risk_reward_target
        self.max_ob_age_candles = max_ob_age_candles

    # ------------------------------------------------------------------
    # Phase 1: 4H Bias from Liquidity Sweep
    # ------------------------------------------------------------------

    def determine_bias(
        self,
        df_4h: pd.DataFrame,
        lookback_window_hours: int = 72,
    ) -> List[DailyBias]:
        """
        Detect 4H liquidity sweeps to establish directional bias.

        A sweep of a swing high → bearish bias (sell into the sweep).
        A sweep of a swing low → bullish bias (buy into the sweep).
        """
        if df_4h is None or len(df_4h) < 20:
            return []

        biases: List[DailyBias] = []

        cutoff = df_4h.index[-1] - timedelta(hours=lookback_window_hours)
        levels = detect_liquidity_levels(df_4h, lookback=3, lookforward=1)

        for level in levels:
            if level.datetime < cutoff:
                continue

            sweep = detect_liquidity_sweep(df_4h, level, start_index=level.index + 1)
            if sweep is None:
                continue

            sweep_idx, opposite_close = sweep
            if sweep_idx >= len(df_4h):
                continue

            sweep_ts = df_4h.index[sweep_idx]

            if level.level_type == "high":
                # Swept a swing high → bearish bias
                direction = BiasType.BEARISH
                reason = f"4H swing high swept at {level.price:.6f}"
            else:
                # Swept a swing low → bullish bias
                direction = BiasType.BULLISH
                reason = f"4H swing low swept at {level.price:.6f}"

            confidence = 70 if opposite_close else 50

            biases.append(DailyBias(
                direction=direction,
                sweep_level=level,
                sweep_index=sweep_idx,
                sweep_timestamp=sweep_ts,
                confidence=confidence,
                reason=reason,
                source_timestamp=level.datetime,
                event_timestamp=sweep_ts,
            ))

        return biases

    # ------------------------------------------------------------------
    # Phase 2: 1H Market Structure Shift
    # ------------------------------------------------------------------

    def confirm_mss(
        self,
        df_1h: pd.DataFrame,
        bias: DailyBias,
        precomputed_mss_list: Optional[List['MSS']] = None
    ) -> Optional[MSSConfirmation]:
        """
        Look for a Market Structure Shift on 1H that aligns with the bias.

        Bullish bias → need bullish MSS (break above swing high).
        Bearish bias → need bearish MSS (break below swing low).

        Only consider MSS that occurred AFTER the 4H sweep.
        """
        if df_1h is None or len(df_1h) < 20:
            return None

        if precomputed_mss_list is not None:
            mss_list = precomputed_mss_list
        else:
            mss_list = detect_mss(df_1h, lookback=5, require_body_close=True)

        target_dir = Direction.BULLISH if bias.direction == BiasType.BULLISH else Direction.BEARISH

        # Find the most recent MSS in the correct direction after the 4H sweep
        for mss in reversed(mss_list):
            if mss.direction != target_dir:
                continue
            if mss.datetime <= bias.sweep_timestamp:
                continue

            return MSSConfirmation(
                mss=mss,
                direction=bias.direction,
                timestamp=mss.datetime,
                details=f"1H {mss.direction.value} MSS at {mss.break_price:.6f}, confirmed close {mss.confirmation_close:.6f}",
            )

        return None

    # ------------------------------------------------------------------
    # Phase 3: 15M Order Block in OTE Zone
    # ------------------------------------------------------------------

    def find_ob_entry(
        self,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        bias: DailyBias,
        mss: MSSConfirmation,
    ) -> Optional[OBEntrySetup]:
        """
        After MSS is confirmed on 1H, look for an Order Block on 15M
        in the OTE zone (0.618–0.786 Fib retracement of the MSS move).

        Then check if price has tapped (or is tapping) the OB on 5M.
        """
        if df_15m is None or len(df_15m) < 10:
            return None
        if df_5m is None or len(df_5m) < 5:
            return None

        import logging
        _log = logging.getLogger(__name__)

        # Determine the impulse leg for Fib calculation
        # The MSS move = from the swing point to the MSS break
        mss_idx_15m = df_15m.index.searchsorted(mss.timestamp)
        if mss_idx_15m >= len(df_15m):
            mss_idx_15m = len(df_15m) - 1

        # Find swing low/high before the MSS on 15M for fib calculation
        if bias.direction == BiasType.BULLISH:
            # Bullish: impulse is from swing low to swing high (MSS break)
            lookback_start = max(0, mss_idx_15m - 60)
            segment = df_15m.iloc[lookback_start:mss_idx_15m + 1]
            if len(segment) < 3:
                _log.debug(f"    P3 BULL: 15m segment too short ({len(segment)})")
                return None

            swing_low = segment['low'].min()
            swing_high = mss.mss.break_price
            if swing_high <= swing_low:
                _log.debug(f"    P3 BULL: swing_high({swing_high}) <= swing_low({swing_low})")
                return None

            # Calculate OTE zone
            fib_range = swing_high - swing_low
            ote_top = swing_high - self.ote_fib_low * fib_range      # 0.618 level
            ote_bottom = swing_high - self.ote_fib_high * fib_range   # 0.786 level
            _log.debug(f"    P3 BULL: OTE zone {ote_bottom:.2f} - {ote_top:.2f}, range={fib_range:.2f}")

            # Find bullish Order Block (last bearish candle before up move) near OTE
            ob = self._find_ob_in_zone(df_15m, Direction.BULLISH, mss_idx_15m, ote_top, ote_bottom)
            if ob is None:
                _log.debug(f"    P3 BULL: No OB found in OTE zone")
                return None
            _log.debug(f"    P3 BULL: OB found at {ob.datetime}, body {ob.body_bottom:.2f}-{ob.body_top:.2f}")

            # Check if price has tapped the OB on 5M AFTER MSS confirms
            # (can't act on OB until MSS confirms the bias)
            tap = self._check_ob_tap(df_5m, ob, Direction.BULLISH, mss.timestamp)
            if tap is None:
                _log.debug(f"    P4 BULL: No 5M tap of OB after MSS time {mss.timestamp}")
                return None

            tap_idx, entry_price = tap
            # Entry at OB body top (conservative) for bullish
            stop_loss = ob.bottom * 0.999  # SL just below OB low with small buffer
            risk = entry_price - stop_loss
            if risk <= 0:
                return None
            tp_price = entry_price + self.risk_reward_target * risk
            rr = self.risk_reward_target

        else:
            # Bearish: impulse is from swing high down to swing low (MSS break)
            lookback_start = max(0, mss_idx_15m - 60)
            segment = df_15m.iloc[lookback_start:mss_idx_15m + 1]
            if len(segment) < 3:
                _log.debug(f"    P3 BEAR: 15m segment too short ({len(segment)})")
                return None

            swing_high = segment['high'].max()
            swing_low = mss.mss.break_price
            if swing_high <= swing_low:
                _log.debug(f"    P3 BEAR: swing_high({swing_high}) <= swing_low({swing_low})")
                return None

            fib_range = swing_high - swing_low
            ote_bottom = swing_low + self.ote_fib_low * fib_range     # 0.618 level
            ote_top = swing_low + self.ote_fib_high * fib_range       # 0.786 level
            _log.debug(f"    P3 BEAR: OTE zone {ote_bottom:.2f} - {ote_top:.2f}, range={fib_range:.2f}")

            # Find bearish Order Block (last bullish candle before down move) near OTE
            ob = self._find_ob_in_zone(df_15m, Direction.BEARISH, mss_idx_15m, ote_top, ote_bottom)
            if ob is None:
                _log.debug(f"    P3 BEAR: No OB found in OTE zone")
                return None
            _log.debug(f"    P3 BEAR: OB found at {ob.datetime}, body {ob.body_bottom:.2f}-{ob.body_top:.2f}")

            # Check if price has tapped the OB on 5M AFTER MSS confirms
            tap = self._check_ob_tap(df_5m, ob, Direction.BEARISH, mss.timestamp)
            if tap is None:
                _log.debug(f"    P4 BEAR: No 5M tap of OB after MSS time {mss.timestamp}")
                return None

            tap_idx, entry_price = tap
            stop_loss = ob.top * 1.001  # SL just above OB high
            risk = stop_loss - entry_price
            if risk <= 0:
                return None
            tp_price = entry_price - self.risk_reward_target * risk
            rr = self.risk_reward_target

        # Calculate fib level where OB sits
        ob_mid = (ob.top + ob.bottom) / 2
        if bias.direction == BiasType.BULLISH:
            fib_level = (swing_high - ob_mid) / fib_range if fib_range > 0 else 0
        else:
            fib_level = (ob_mid - swing_low) / fib_range if fib_range > 0 else 0

        in_ote = self.ote_fib_low <= fib_level <= self.ote_fib_high

        return OBEntrySetup(
            order_block=ob,
            fib_level=round(fib_level, 3),
            in_ote_zone=in_ote,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp_price=tp_price,
            risk_reward=round(rr, 2),
            timestamp=df_5m.index[tap_idx],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_ob_in_zone(
        self,
        df_15m: pd.DataFrame,
        direction: Direction,
        mss_idx: int,
        zone_top: float,
        zone_bottom: float,
    ) -> Optional[OrderBlock]:
        """
        Find the most recent Order Block whose body overlaps the zone.
        An OB overlaps if its body intersects with [zone_bottom, zone_top].
        """
        search_start = max(0, mss_idx - self.max_ob_age_candles)

        for i in range(mss_idx - 1, search_start - 1, -1):
            candle = df_15m.iloc[i]

            if direction == Direction.BULLISH:
                if is_bearish_candle(candle):
                    body_top = candle['open']
                    body_bottom = candle['close']
                    # Check overlap: body intersects zone
                    if body_top >= zone_bottom and body_bottom <= zone_top:
                        return OrderBlock(
                            index=i,
                            datetime=df_15m.index[i],
                            direction=Direction.BULLISH,
                            top=candle['high'],
                            bottom=candle['low'],
                            body_top=body_top,
                            body_bottom=body_bottom,
                        )
            else:
                if is_bullish_candle(candle):
                    body_top = candle['close']
                    body_bottom = candle['open']
                    # Check overlap: body intersects zone
                    if body_top >= zone_bottom and body_bottom <= zone_top:
                        return OrderBlock(
                            index=i,
                            datetime=df_15m.index[i],
                            direction=Direction.BEARISH,
                            top=candle['high'],
                            bottom=candle['low'],
                            body_top=body_top,
                            body_bottom=body_bottom,
                        )

        return None

    def _check_ob_tap(
        self,
        df_5m: pd.DataFrame,
        ob: OrderBlock,
        direction: Direction,
        after_time: pd.Timestamp,
    ) -> Optional[Tuple[int, float]]:
        """
        Check if price tapped the Order Block on 5M after the MSS time.
        Returns (5m_index, entry_price) if tapped.
        """
        start_idx = df_5m.index.searchsorted(after_time)

        for i in range(start_idx, len(df_5m)):
            candle = df_5m.iloc[i]

            if direction == Direction.BULLISH:
                # Price dips into the OB body zone
                if candle['low'] <= ob.body_top:
                    # Enter at OB body top (or candle close if above)
                    entry = max(ob.body_top, candle['close'])
                    return (i, entry)
            else:
                # Price rallies into the OB body zone
                if candle['high'] >= ob.body_bottom:
                    entry = min(ob.body_bottom, candle['close'])
                    return (i, entry)

        return None

    # ------------------------------------------------------------------
    # Quality Score
    # ------------------------------------------------------------------

    def calculate_quality_score(
        self,
        bias: DailyBias,
        mss: MSSConfirmation,
        ob_entry: OBEntrySetup,
    ) -> int:
        """
        Quality score (0-100):
        - Bias confidence (0-30)
        - MSS quality: body close beyond level (0-25)
        - OB in OTE zone (0-25)
        - OB body size relative to range (0-20)
        """
        score = 0

        # Bias confidence (0-30)
        score += int(bias.confidence * 0.3)

        # MSS quality (0-25)
        score += 15  # Base for having MSS
        # Bonus if candle closed decisively beyond level
        break_dist = abs(mss.mss.confirmation_close - mss.mss.break_price) / mss.mss.break_price * 100
        if break_dist > 0.2:
            score += 10  # Decisive break

        # OB in OTE zone (0-25)
        if ob_entry.in_ote_zone:
            score += 25
        else:
            # Partial credit for being close
            score += 10

        # OB body size (0-20) — bigger body = stronger institutional print
        ob = ob_entry.order_block
        body_size = abs(ob.body_top - ob.body_bottom)
        total_range = ob.top - ob.bottom
        body_ratio = body_size / total_range if total_range > 0 else 0
        score += int(body_ratio * 20)

        return min(100, score)

    # ------------------------------------------------------------------
    # Full Signal Generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame,
        lookback_window_hours: int = 72,
    ) -> List[MSSOB_Signal]:
        """Generate MSS + OB signals for a symbol."""
        import logging
        logger = logging.getLogger(__name__)
        signals: List[MSSOB_Signal] = []

        # Phase 1: 4H bias
        biases = self.determine_bias(df_4h, lookback_window_hours)
        if not biases:
            logger.debug(f"  [{symbol}] Phase 1: No 4H biases found")
            return []
        logger.debug(f"  [{symbol}] Phase 1: {len(biases)} biases found")

        phase2_pass = 0
        phase3_pass = 0
        for bias in biases:
            if bias.direction == BiasType.NEUTRAL:
                continue

            # Phase 2: 1H MSS confirmation
            mss_conf = self.confirm_mss(df_1h, bias)
            if mss_conf is None:
                continue
            phase2_pass += 1

            # Phase 3+4: 15M OB entry + 5M tap
            ob_entry = self.find_ob_entry(df_15m, df_5m, bias, mss_conf)
            if ob_entry is None:
                continue
            phase3_pass += 1

            quality = self.calculate_quality_score(bias, mss_conf, ob_entry)

            sig_direction = Direction.BULLISH if bias.direction == BiasType.BULLISH else Direction.BEARISH

            signal = MSSOB_Signal(
                index=0,
                datetime=ob_entry.timestamp,
                direction=sig_direction,
                entry_price=ob_entry.entry_price,
                stop_loss=ob_entry.stop_loss,
                take_profit=ob_entry.tp_price,
                risk_reward=ob_entry.risk_reward,
                signal_type="MSS_OB_ENTRY",
                symbol=symbol,
                daily_bias=bias,
                mss_confirmation=mss_conf,
                ob_entry=ob_entry,
                quality_score=quality,
            )
            signals.append(signal)

        logger.debug(f"  [{symbol}] Phase 2 pass: {phase2_pass}, Phase 3+4 pass: {phase3_pass}, Final signals: {len(signals)}")
        return signals


def format_signal_for_jsonl(signal: MSSOB_Signal) -> Dict[str, Any]:
    """Format MSS+OB signal for JSONL output with IST timestamps."""

    def to_ist_str(dt) -> Optional[str]:
        if dt is None:
            return None
        if hasattr(dt, 'tzinfo') and dt.tzinfo is None:
            dt = UTC.localize(dt)
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")

    return {
        "signal_datetime_ist": to_ist_str(signal.datetime),
        "symbol": signal.symbol,
        "direction": signal.direction.value,
        "quality_score": signal.quality_score,
        "entry_price": signal.entry_price,
        "sl_price": signal.stop_loss,
        "tp_price": signal.take_profit,
        "risk_reward": round(signal.risk_reward, 2),
        "daily_bias": {
            "direction": signal.daily_bias.direction.value,
            "confidence": signal.daily_bias.confidence,
            "reason": signal.daily_bias.reason,
            "sweep_time_ist": to_ist_str(signal.daily_bias.sweep_timestamp),
            "source_time_ist": to_ist_str(signal.daily_bias.source_timestamp),
        },
        "mss_confirmation": {
            "direction": signal.mss_confirmation.direction.value,
            "break_price": signal.mss_confirmation.mss.break_price,
            "confirmation_close": signal.mss_confirmation.mss.confirmation_close,
            "time_ist": to_ist_str(signal.mss_confirmation.timestamp),
            "details": signal.mss_confirmation.details,
        },
        "ob_entry": {
            "ob_top": signal.ob_entry.order_block.top,
            "ob_bottom": signal.ob_entry.order_block.bottom,
            "ob_body_top": signal.ob_entry.order_block.body_top,
            "ob_body_bottom": signal.ob_entry.order_block.body_bottom,
            "ob_time_ist": to_ist_str(signal.ob_entry.order_block.datetime),
            "fib_level": signal.ob_entry.fib_level,
            "in_ote_zone": signal.ob_entry.in_ote_zone,
        },
        "detected_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
    }
