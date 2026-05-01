"""
Backtest Runner - Strategy 09 MSS + OB
======================================

Stand-alone backtesting module allowing exact specification
of target time frames (start/end dates).
Uses historical Binance Futures klines to accurately reconstruct
the 4H, 1H, 15M, and 5M data needed for Strategy 09.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any

# Ensure we can import the original core strategy codebase directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from strategy_09_mss_ob_entry.strategy import MSSOrderBlockStrategy, MSSOB_Signal
from strategy_09_mss_ob_entry.crypto_backtester import MSSObBacktester, BacktestResult, BacktestSummary
from backtest.historical_fetcher import get_historical_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class HistoricalBacktester(MSSObBacktester):
    """
    Extends the existing backtester to feed historical 
    hard-bounded timeframe data rather than N-latest bars.
    """
    def __init__(self, strategy: MSSOrderBlockStrategy, start_date: str, end_date: str, cache_dir: Path):
        super().__init__(strategy=strategy, fetcher=None) # We override the data fetching phase
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir

    def backtest_symbol(self, symbol: str, scan_lookback_hours: int = 200) -> List[BacktestResult]:
        logger.info(f"Backtesting {symbol} from {self.start_date} to {self.end_date}...")
        
        # 1. Fetch exact historical date ranges
        data = get_historical_data(symbol, self.start_date, self.end_date, self.cache_dir)
        
        for tf in ['4h', '1h', '15m', '5m']:
            if data.get(tf) is None or data[tf].empty:
                logger.warning(f"Missing {tf} data for {symbol}, skipping symbol.")
                return []

        df_4h = data['4h']
        df_1h = data['1h']
        df_15m = data['15m']
        df_5m = data['5m']

        results: List[BacktestResult] = []
        used_bias_keys = set()

        logger.info(f"  Scanning for signals on historical data ({len(df_1h)} 1H bars)...")
        # 2. Run the core path B generator on the entire historical dataset
        signals = self.strategy.generate_signal(
            symbol=symbol,
            df_4h=df_4h,
            df_1h=df_1h,
            df_15m=df_15m,
            df_5m=df_5m,
            lookback_window_hours=scan_lookback_hours,
        )
        logger.info(f"  Raw signals generated: {len(signals)}")

        for sig in signals:
            import pytz
            utc = pytz.UTC
            
            sig_time = sig.datetime
            if sig_time.tzinfo is None:
                sig_time = utc.localize(sig_time)
                
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d").replace(tzinfo=utc)
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d").replace(tzinfo=utc)
            
            if sig_time < start_dt or sig_time > end_dt:
                continue

            # Deduplication logic copied verbatim from generic `crypto_backtester`
            bias_key = f"{sig.daily_bias.source_timestamp}|{sig.daily_bias.direction.value}"
            if bias_key in used_bias_keys:
                continue

            is_dup = False
            for prev in results:
                time_diff = abs((sig.datetime - prev.entry_time).total_seconds())
                if time_diff < 4 * 3600:
                    is_dup = True
                    break
            if is_dup:
                continue

            used_bias_keys.add(bias_key)
            result = self._simulate_trade(sig, df_5m)
            results.append(result)
            logger.info(
                f"  Signal {sig.datetime}: {sig.direction.value} "
                f"Entry={sig.entry_price:.4f}, Exit={result.exit_reason}, "
                f"P&L={result.pnl_rr:.2f}R"
            )

        logger.info(f"  Completed {symbol}: {len(results)} trades")
        return results

def main():
    parser = argparse.ArgumentParser(description="Timeframe-bound Backtest for Strategy 09")
    parser.add_argument("--pairs", nargs="+", required=True, help="List of pairs e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--start", type=str, required=True, help="Start Date in YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End Date in YYYY-MM-DD")
    parser.add_argument("--lookback", type=int, default=200, help="Lookback hours for signal scan")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSONL file")
    args = parser.parse_args()

    cache_dir = PROJECT_ROOT / "backtest" / "data_cache"
    output_file = args.output or str(PROJECT_ROOT / "backtest" / f"results_{args.start}_{args.end}.jsonl")

    strategy = MSSOrderBlockStrategy()
    backtester = HistoricalBacktester(
        strategy=strategy, 
        start_date=args.start, 
        end_date=args.end, 
        cache_dir=cache_dir
    )
    
    result = backtester.run_backtest(
        symbols=args.pairs,
        output_file=output_file,
        scan_lookback_hours=args.lookback,
    )
    logger.info(f"\\nTimeframe Backtest complete! Results saved to: {result['output_file']}")

if __name__ == "__main__":
    main()
