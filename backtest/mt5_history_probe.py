"""MT5 History probe utilities"""

# Timeframe mapping for MT5
TIMEFRAME_MAP = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
    "1w": "TIMEFRAME_W1",
    "1mo": "TIMEFRAME_MN1",
}

def _resolve_timeframes(timeframes_arg):
    """Resolve timeframe argument to list of timeframes"""
    if timeframes_arg.lower() == "all":
        return list(TIMEFRAME_MAP.keys())
    
    return [tf.strip() for tf in timeframes_arg.split(",") if tf.strip()]

def _find_max_pos(mt5, symbol, tf_value, max_doublings=30):
    """Binary search to find maximum position (oldest bar) for a symbol/timeframe"""
    try:
        # Try to get bars from position 0
        rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 1)
        if rates is None or len(rates) == 0:
            return None, "No data available"
        
        # Binary search for max position
        low, high = 0, 2 ** max_doublings
        max_pos = 0
        
        for _ in range(max_doublings):
            mid = (low + high) // 2
            rates = mt5.copy_rates_from_pos(symbol, tf_value, mid, 1)
            
            if rates is not None and len(rates) > 0:
                max_pos = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return max_pos, None
    except Exception as e:
        return None, str(e)
