import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def fetch_binance_futures_klines(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    """Fetch historical klines from Binance USD-M Futures."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1500
    all_klines = []
    
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data or isinstance(data, dict) and 'code' in data:
                logger.warning(f"Binance API error or no data: {data}")
                break
                
            all_klines.extend(data)
            
            # The last candle's open time + 1ms to get the next batch
            current_start = data[-1][0] + 1
            
            # Rate limit protection
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            break

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.index = df.index.tz_localize('UTC')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_historical_data(symbol: str, start_str: str, end_str: str, cache_dir: Path) -> dict:
    """Gets 4H, 1H, 15m, 5m data for the specified timeframe using local CSV caching."""
    import pytz
    
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    
    # Pad start time to ensure enough lookback for indicators
    # We need at least 200 4H bars so pad by ~35 days.
    pad_start = start_dt - pd.Timedelta(days=40)
    
    start_ts = int(pad_start.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    data = {}
    intervals = {
        '4h': '4h',
        '1h': '1h',
        '15m': '15m',
        '5m': '5m'
    }
    
    logger.info(f"Fetching data for {symbol} from {pad_start.date()} to {end_dt.date()} (including lookback margin)")
    
    for key, binance_interval in intervals.items():
        cache_file = cache_dir / f"{symbol}_{key}_{start_str}_{end_str}.csv"
        
        if cache_file.exists():
            logger.info(f"Loading cached {key} data for {symbol}")
            df = pd.read_csv(cache_file, index_col='datetime', parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
        else:
            logger.info(f"Downloading {key} data for {symbol} via Binance Futures API...")
            df = fetch_binance_futures_klines(symbol, binance_interval, start_ts, end_ts)
            if not df.empty:
                df.to_csv(cache_file)
                
        data[key] = df
        
    return data
