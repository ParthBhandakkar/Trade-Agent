"""
ICT/SMC Trading Indicators and Pattern Detection

Contains functions for detecting:
- Fair Value Gaps (FVG)
- Liquidity Levels (Swing Highs/Lows)
- Liquidity Sweeps
- Market Structure Shifts (MSS)
- Breaker Blocks
- Order Blocks
- Inversion FVGs
- Fibonacci Standard Deviation levels
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class FVG:
    """Fair Value Gap structure"""
    index: int                    # Index in DataFrame where FVG was formed (candle 2)
    datetime: pd.Timestamp        # Timestamp of the FVG
    direction: Direction          # Bullish or Bearish
    top: float                    # Top of the gap
    bottom: float                 # Bottom of the gap
    mitigated: bool = False       # Whether the FVG has been filled
    mitigation_index: Optional[int] = None
    
    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2
    
    @property
    def size(self) -> float:
        return self.top - self.bottom


@dataclass
class LiquidityLevel:
    """Swing High/Low liquidity level"""
    index: int
    datetime: pd.Timestamp
    price: float
    level_type: Literal["high", "low"]
    swept: bool = False
    sweep_index: Optional[int] = None


@dataclass
class MSS:
    """Market Structure Shift"""
    index: int                    # Index where MSS occurred
    datetime: pd.Timestamp
    direction: Direction          # New direction after shift
    break_price: float            # Price level that was broken
    confirmation_close: float     # Close price that confirmed the MSS


@dataclass
class BreakerBlock:
    """Breaker Block - failed Order Block that becomes entry zone"""
    index: int
    datetime: pd.Timestamp
    direction: Direction          # Direction to trade (bullish = buy at breaker)
    top: float
    bottom: float
    original_ob_index: int        # Index of the original Order Block


@dataclass
class OrderBlock:
    """Order Block structure"""
    index: int
    datetime: pd.Timestamp
    direction: Direction
    top: float                    # High of the OB candle
    bottom: float                 # Low of the OB candle
    body_top: float              # Top of candle body
    body_bottom: float           # Bottom of candle body
    mitigated: bool = False


@dataclass
class Signal:
    """Trading signal"""
    index: int
    datetime: pd.Timestamp
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    signal_type: str              # Name of the pattern/setup


def is_bullish_candle(row: pd.Series) -> bool:
    """Check if candle is bullish (close > open)"""
    return row['close'] > row['open']


def is_bearish_candle(row: pd.Series) -> bool:
    """Check if candle is bearish (close < open)"""
    return row['close'] < row['open']


def detect_fvg(
    df: pd.DataFrame,
    min_gap_percent: float = 0.0,
    check_mitigation: bool = True
) -> List[FVG]:
    """
    Detect Fair Value Gaps in price data
    
    A Bullish FVG forms when:
    - Candle 1's high < Candle 3's low (gap up)
    
    A Bearish FVG forms when:
    - Candle 1's low > Candle 3's high (gap down)
    
    Args:
        df: OHLCV DataFrame
        min_gap_percent: Minimum gap size as percentage of price (0.0 = any gap)
        check_mitigation: Whether to check if FVGs have been mitigated
        
    Returns:
        List of FVG objects
    """
    fvgs = []
    
    for i in range(2, len(df)):
        candle_1 = df.iloc[i - 2]
        candle_2 = df.iloc[i - 1]  # The FVG candle
        candle_3 = df.iloc[i]
        
        # Bullish FVG: Gap between candle 1 high and candle 3 low
        if candle_3['low'] > candle_1['high']:
            gap_size = candle_3['low'] - candle_1['high']
            gap_percent = gap_size / candle_2['close'] * 100
            
            if gap_percent >= min_gap_percent:
                fvg = FVG(
                    index=i - 1,
                    datetime=df.index[i - 1],
                    direction=Direction.BULLISH,
                    top=candle_3['low'],
                    bottom=candle_1['high'],
                )
                fvgs.append(fvg)
        
        # Bearish FVG: Gap between candle 1 low and candle 3 high
        elif candle_1['low'] > candle_3['high']:
            gap_size = candle_1['low'] - candle_3['high']
            gap_percent = gap_size / candle_2['close'] * 100
            
            if gap_percent >= min_gap_percent:
                fvg = FVG(
                    index=i - 1,
                    datetime=df.index[i - 1],
                    direction=Direction.BEARISH,
                    top=candle_1['low'],
                    bottom=candle_3['high'],
                )
                fvgs.append(fvg)
    
    # Check mitigation if requested
    if check_mitigation:
        fvgs = _check_fvg_mitigation(df, fvgs)
    
    return fvgs


def _check_fvg_mitigation(df: pd.DataFrame, fvgs: List[FVG]) -> List[FVG]:
    """Check if FVGs have been mitigated (price returned to fill the gap)

    FVG is only invalidated when a candle CLOSES inside the FVG range:
    - Bullish FVG: Invalidated when candle closes BELOW the FVG bottom
    - Bearish FVG: Invalidated when candle closes ABOVE the FVG top
    """
    for fvg in fvgs:
        for i in range(fvg.index + 2, len(df)):
            candle = df.iloc[i]

            if fvg.direction == Direction.BULLISH:
                # Bullish FVG is only mitigated when candle closes BELOW the FVG bottom
                if candle['close'] < fvg.bottom:
                    fvg.mitigated = True
                    fvg.mitigation_index = i
                    break
            else:
                # Bearish FVG is only mitigated when candle closes ABOVE the FVG top
                if candle['close'] > fvg.top:
                    fvg.mitigated = True
                    fvg.mitigation_index = i
                    break

    return fvgs


def get_unmitigated_fvgs(fvgs: List[FVG], current_index: int) -> List[FVG]:
    """Get FVGs that haven't been mitigated up to current index"""
    return [
        fvg for fvg in fvgs 
        if not fvg.mitigated or (fvg.mitigation_index and fvg.mitigation_index > current_index)
    ]


def detect_liquidity_levels(
    df: pd.DataFrame,
    lookback: int = 5,
    lookforward: int = 5
) -> List[LiquidityLevel]:
    """
    Detect swing highs and lows (liquidity levels)
    Optimized via vectorized operations.
    """
    levels = []
    
    highs = df['high']
    lows = df['low']
    
    # Vectorized calculation for 'before' window
    before_high_max = highs.rolling(window=lookback, min_periods=lookback).max().shift(1)
    before_low_min = lows.rolling(window=lookback, min_periods=lookback).min().shift(1)
    
    # Vectorized calculation for 'after' window (using reversed series)
    after_high_max = highs.iloc[::-1].rolling(window=lookforward, min_periods=lookforward).max().shift(1).iloc[::-1]
    after_low_min = lows.iloc[::-1].rolling(window=lookforward, min_periods=lookforward).min().shift(1).iloc[::-1]
    
    # Identify swing highs and lows
    is_swing_high = (highs >= before_high_max) & (highs >= after_high_max)
    is_swing_low = (lows <= before_low_min) & (lows <= after_low_min)
    
    # Get indices where swings occur
    # We must also ensure we skip the first `lookback` and last `lookforward` elements
    valid_range = pd.Series(False, index=df.index)
    valid_range.iloc[lookback:len(df)-lookforward] = True
    
    swing_high_dates = df[is_swing_high & valid_range]
    swing_low_dates = df[is_swing_low & valid_range]
    
    # In order to maintain chronological order without much overhead, 
    # we can construct both lists, merge them or keep them as they are and sort by index.
    
    # To mimic original behavior which appends in chronological order:
    for idx_pos, (dt, row) in enumerate(swing_high_dates.iterrows()):
        # We need the integer index (i). 
        # Since iterrows doesn't give integer location efficiently, we can use np.where
        pass
        
    # An efficient way to maintain order is to combine all swings and then sort
    sh_df = pd.DataFrame({
        'index': np.where(is_swing_high & valid_range)[0],
        'datetime': swing_high_dates.index,
        'price': swing_high_dates['high'],
        'level_type': 'high'
    })
    
    sl_df = pd.DataFrame({
        'index': np.where(is_swing_low & valid_range)[0],
        'datetime': swing_low_dates.index,
        'price': swing_low_dates['low'],
        'level_type': 'low'
    })
    
    combined = pd.concat([sh_df, sl_df]).sort_values('index')
    
    # Vectorized unpacking instead of iterrows
    idx_vals = combined['index'].values
    dt_vals = combined['datetime'].tolist() # Preserve timezone info
    pr_vals = combined['price'].values
    lt_vals = combined['level_type'].values
    
    for i in range(len(combined)):
        levels.append(LiquidityLevel(
            index=int(idx_vals[i]),
            datetime=dt_vals[i],
            price=float(pr_vals[i]),
            level_type=str(lt_vals[i])
        ))
        
    return levels


def detect_liquidity_sweep(
    df: pd.DataFrame,
    level: LiquidityLevel,
    start_index: Optional[int] = None
) -> Optional[Tuple[int, bool]]:
    """
    Detect if a liquidity level has been swept
    """
    import numpy as np
    
    start = start_index if start_index is not None else level.index + 1
    
    if start >= len(df):
        return None
        
    prices = df.iloc[start:]
    
    if level.level_type == "high":
        sweeps = (prices['high'] > level.price).values
        if sweeps.any():
            idx_rel = sweeps.argmax()
            i = start + idx_rel
            candle = df.iloc[i]
            return (i, is_bearish_candle(candle))
    else:  # level_type == "low"
        sweeps = (prices['low'] < level.price).values
        if sweeps.any():
            idx_rel = sweeps.argmax()
            i = start + idx_rel
            candle = df.iloc[i]
            return (i, is_bullish_candle(candle))
            
def detect_liquidity_sweep_first_candle(
    df: pd.DataFrame,
    level: LiquidityLevel,
    start_index: Optional[int] = None
) -> Optional[int]:
    """
    Detect liquidity sweep where the FIRST sweeping candle closes opposite
    
    This is the key concept for TBE (Time-Based Volume) strategy.
    The very first candle that sweeps must close in the opposite direction.
    
    Args:
        df: OHLCV DataFrame
        level: LiquidityLevel to check
        start_index: Index to start checking from
        
    Returns:
        Index of the sweep candle if valid, None otherwise
    """
    result = detect_liquidity_sweep(df, level, start_index)
    
    if result is not None:
        sweep_index, opposite_close = result
        if opposite_close:
            return sweep_index
    
    return None


def detect_mss(
    df: pd.DataFrame,
    lookback: int = 10,
    require_body_close: bool = True
) -> List[MSS]:
    """
    Detect Market Structure Shifts
    
    A bullish MSS occurs when price breaks above a recent swing high.
    A bearish MSS occurs when price breaks below a recent swing low.
    
    Args:
        df: OHLCV DataFrame
        lookback: Number of candles to look back for swing points
        require_body_close: Require candle body to close beyond the level
        
    Returns:
        List of MSS objects
    """
    mss_list = []
    
    # First detect swing highs and lows
    levels = detect_liquidity_levels(df, lookback=3, lookforward=1)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for level in levels:
        start_idx = level.index + 2
        if start_idx >= len(df):
            continue
            
        if level.level_type == "high":
            # Bullish MSS - break above swing high
            search_vals = closes[start_idx:] if require_body_close else highs[start_idx:]
            breaks = search_vals > level.price
            if breaks.any():
                idx_rel = breaks.argmax()
                i = start_idx + idx_rel
                mss_list.append(MSS(
                    index=i,
                    datetime=pd.Timestamp(df.index[i]),
                    direction=Direction.BULLISH,
                    break_price=level.price,
                    confirmation_close=closes[i]
                ))
        else:  # swing low
            # Bearish MSS - break below swing low
            search_vals = closes[start_idx:] if require_body_close else lows[start_idx:]
            breaks = search_vals < level.price
            if breaks.any():
                idx_rel = breaks.argmax()
                i = start_idx + idx_rel
                mss_list.append(MSS(
                    index=i,
                    datetime=pd.Timestamp(df.index[i]),
                    direction=Direction.BEARISH,
                    break_price=level.price,
                    confirmation_close=closes[i]
                ))
    
    return mss_list


def detect_order_block(
    df: pd.DataFrame,
    direction: Direction,
    start_index: int,
    lookback: int = 10
) -> Optional[OrderBlock]:
    """
    Detect the most recent Order Block before a move
    
    Bullish OB: Last bearish candle before a bullish move
    Bearish OB: Last bullish candle before a bearish move
    
    Args:
        df: OHLCV DataFrame
        direction: Direction of the expected move (bullish = look for bearish OB)
        start_index: Index to start looking back from
        lookback: Maximum candles to look back
        
    Returns:
        OrderBlock object or None
    """
    search_start = max(0, start_index - lookback)
    
    for i in range(start_index - 1, search_start - 1, -1):
        candle = df.iloc[i]
        
        if direction == Direction.BULLISH:
            # Look for bearish candle (last down candle before up move)
            if is_bearish_candle(candle):
                return OrderBlock(
                    index=i,
                    datetime=df.index[i],
                    direction=Direction.BULLISH,
                    top=candle['high'],
                    bottom=candle['low'],
                    body_top=candle['open'],
                    body_bottom=candle['close']
                )
        else:
            # Look for bullish candle (last up candle before down move)
            if is_bullish_candle(candle):
                return OrderBlock(
                    index=i,
                    datetime=df.index[i],
                    direction=Direction.BEARISH,
                    top=candle['high'],
                    bottom=candle['low'],
                    body_top=candle['close'],
                    body_bottom=candle['open']
                )
    
    return None


def detect_breaker_block(
    df: pd.DataFrame,
    direction: Direction,
    sweep_index: int,
    lookback: int = 20
) -> Optional[BreakerBlock]:
    """
    Detect Breaker Block after a liquidity sweep
    
    A Breaker Block is formed when:
    1. An Order Block is created
    2. Price sweeps through the OB (invalidating it)
    3. Price reverses - the violated OB becomes a Breaker Block
    
    For longs: After sweep of low, find the bearish OB that was violated
    For shorts: After sweep of high, find the bullish OB that was violated
    
    Args:
        df: OHLCV DataFrame
        direction: Direction to trade (bullish = looking for support breaker)
        sweep_index: Index where the sweep occurred
        lookback: Candles to look back for the original OB
        
    Returns:
        BreakerBlock object or None
    """
    # Look for Order Block before the sweep
    search_start = max(0, sweep_index - lookback)
    
    for i in range(sweep_index - 1, search_start - 1, -1):
        candle = df.iloc[i]
        
        if direction == Direction.BULLISH:
            # For bullish breaker, find bearish candle that was swept through
            if is_bearish_candle(candle):
                # Check if this candle's low was swept
                sweep_candle = df.iloc[sweep_index]
                if sweep_candle['low'] < candle['low']:
                    return BreakerBlock(
                        index=i,
                        datetime=df.index[i],
                        direction=Direction.BULLISH,
                        top=candle['high'],
                        bottom=candle['low'],
                        original_ob_index=i
                    )
        else:
            # For bearish breaker, find bullish candle that was swept through
            if is_bullish_candle(candle):
                # Check if this candle's high was swept
                sweep_candle = df.iloc[sweep_index]
                if sweep_candle['high'] > candle['high']:
                    return BreakerBlock(
                        index=i,
                        datetime=df.index[i],
                        direction=Direction.BEARISH,
                        top=candle['high'],
                        bottom=candle['low'],
                        original_ob_index=i
                    )
    
    return None


def detect_inversion_fvg(
    df: pd.DataFrame,
    fvg: FVG,
    current_index: int
) -> bool:
    """
    Check if an FVG has been inverted (violated and now acts as opposite zone)
    
    An FVG inverts when:
    - Bullish FVG: Price closes below the FVG bottom (now acts as resistance)
    - Bearish FVG: Price closes above the FVG top (now acts as support)
    
    Args:
        df: OHLCV DataFrame
        fvg: FVG to check
        current_index: Current candle index
        
    Returns:
        True if FVG has been inverted
    """
    for i in range(fvg.index + 2, min(current_index + 1, len(df))):
        candle = df.iloc[i]
        
        if fvg.direction == Direction.BULLISH:
            # Bullish FVG inverts when price closes below it
            if candle['close'] < fvg.bottom:
                return True
        else:
            # Bearish FVG inverts when price closes above it
            if candle['close'] > fvg.top:
                return True
    
    return False


def calculate_fib_levels(
    swing_high: float,
    swing_low: float,
    direction: Direction,
    levels: List[float] = None
) -> Dict[float, float]:
    """
    Calculate Fibonacci retracement/extension levels
    
    Args:
        swing_high: High price of the range
        swing_low: Low price of the range
        direction: Direction of the move (bullish = measuring upward move)
        levels: List of Fib levels to calculate (default includes standard + SD levels)
        
    Returns:
        Dictionary mapping level to price
    """
    if levels is None:
        # Standard retracement + ICT Standard Deviation levels
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 
                  -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]  # Negative = extensions
    
    range_size = swing_high - swing_low
    fib_prices = {}
    
    for level in levels:
        if direction == Direction.BULLISH:
            # For bullish move, 0 = low, 1 = high, negative = above high
            fib_prices[level] = swing_low + (range_size * (1 - level))
        else:
            # For bearish move, 0 = high, 1 = low, negative = below low
            fib_prices[level] = swing_high - (range_size * (1 - level))
    
    return fib_prices


def calculate_standard_deviation_levels(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    direction: Direction
) -> Dict[float, float]:
    """
    Calculate ICT Standard Deviation projection levels
    
    Used for targeting where manipulation/distribution may end.
    Common levels: -2.0, -2.5 (where price often reverses)
    
    Args:
        df: OHLCV DataFrame
        start_index: Start of the move
        end_index: End of the move  
        direction: Direction of the original move
        
    Returns:
        Dictionary with SD levels and their prices
    """
    start_candle = df.iloc[start_index]
    end_candle = df.iloc[end_index]
    
    if direction == Direction.BULLISH:
        swing_low = start_candle['low']
        swing_high = end_candle['high']
    else:
        swing_high = start_candle['high']
        swing_low = end_candle['low']
    
    return calculate_fib_levels(swing_high, swing_low, direction)


def find_nearest_fvg(
    fvgs: List[FVG],
    price: float,
    direction: Direction,
    max_distance_percent: float = 2.0
) -> Optional[FVG]:
    """
    Find the nearest unmitigated FVG to a price level
    
    Args:
        fvgs: List of FVG objects
        price: Current price
        direction: Direction to look (bullish = below price, bearish = above)
        max_distance_percent: Maximum distance as percentage of price
        
    Returns:
        Nearest FVG or None
    """
    candidates = []
    
    for fvg in fvgs:
        if fvg.mitigated:
            continue
        
        if direction == Direction.BULLISH:
            # For bullish, look for FVG below current price
            if fvg.top < price:
                distance = (price - fvg.midpoint) / price * 100
                if distance <= max_distance_percent:
                    candidates.append((fvg, distance))
        else:
            # For bearish, look for FVG above current price
            if fvg.bottom > price:
                distance = (fvg.midpoint - price) / price * 100
                if distance <= max_distance_percent:
                    candidates.append((fvg, distance))
    
    if not candidates:
        return None
    
    # Return the nearest FVG
    return min(candidates, key=lambda x: x[1])[0]


def price_in_fvg(price: float, fvg: FVG) -> bool:
    """Check if price is within an FVG zone"""
    return fvg.bottom <= price <= fvg.top


def price_in_zone(price: float, top: float, bottom: float) -> bool:
    """Check if price is within a zone"""
    return bottom <= price <= top

