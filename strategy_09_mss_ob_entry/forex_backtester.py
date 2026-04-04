"""
Strategy 09: MSS + Order Block Entry — Forex Backtester
========================================================

Backtests the MSS + OB strategy on forex pairs using
OANDA data via TradingView.

Usage:
    python forex_backtester.py
    python forex_backtester.py --pairs EURUSD GBPUSD --lookback 200
    python forex_backtester.py --output results/forex_backtest.jsonl
"""

import os
import sys
import json
import logging
import argparse
import time
import tempfile
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pytz
import requests
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.utils.env_loader import load_root_env

load_root_env(Path(__file__).parent.parent)

from strategy import (
    MSSOrderBlockStrategy,
    MSSOB_Signal,
    format_signal_for_jsonl,
    BiasType,
)
from scripts.utils.indicators import Direction

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    print("ERROR: tvDatafeed not installed.")
    print("pip install git+https://github.com/rongardF/tvdatafeed.git")
    sys.exit(1)

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC
TV_SOURCE_TZ_NAME = os.environ.get("TV_SOURCE_TZ", "Asia/Kolkata").strip() or "Asia/Kolkata"
try:
    TV_SOURCE_TZ = pytz.timezone(TV_SOURCE_TZ_NAME)
except Exception:
    TV_SOURCE_TZ_NAME = "Asia/Kolkata"
    TV_SOURCE_TZ = IST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# FOREX PAIRS
# ============================================================================

FOREX_PAIRS = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    # Crosses
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDJPY", "NZDCAD", "NZDCHF",
    "CADJPY", "CADCHF",
    "CHFJPY",
    # Gold
    "XAUUSD",
]

INTERVALS = {
    "1m": Interval.in_1_minute,
    "3m": Interval.in_3_minute,
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "1h": Interval.in_1_hour,
    "4h": Interval.in_4_hour,
}


# ============================================================================
# DATA FETCHING (multiprocessing for timeout safety)
# ============================================================================

def _resolve_auth_token(session_id: str) -> Optional[str]:
    """Convert a TradingView browser sessionid cookie into the websocket auth_token."""
    import re
    try:
        session = requests.Session()
        session.cookies.set('sessionid', session_id, domain='.tradingview.com')
        resp = session.get(
            'https://www.tradingview.com/chart/',
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                     'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0'},
            timeout=30,
        )
        m = re.search(r'"auth_token":"([^"]+)"', resp.text)
        if m:
            return m.group(1)
        logger.warning("Could not extract auth_token from TradingView chart page")
    except Exception as e:
        logger.warning(f"Failed to resolve auth_token from sessionid: {e}")
    return None


def _subprocess_fetch(symbol, exchange, interval_str, n_bars, result_file,
                      tv_username=None, tv_password=None, tv_session_token=None):
    """Top-level function for multiprocessing: fetch data and write CSV."""
    try:
        if tv_session_token:
            tv = TvDatafeed()
            tv.token = tv_session_token
        elif tv_username and tv_password:
            tv = TvDatafeed(username=tv_username, password=tv_password)
        else:
            tv = TvDatafeed()
        interval_map = {
            "in_1_minute": Interval.in_1_minute,
            "in_3_minute": Interval.in_3_minute,
            "in_5_minute": Interval.in_5_minute,
            "in_15_minute": Interval.in_15_minute,
            "in_1_hour": Interval.in_1_hour,
            "in_4_hour": Interval.in_4_hour,
        }
        tv_interval = interval_map.get(interval_str)
        df = tv.get_hist(symbol=symbol, exchange=exchange,
                         interval=tv_interval, n_bars=n_bars)
        if df is not None and not df.empty:
            df.to_csv(result_file)
    except Exception:
        pass


class ForexDataFetcher:
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._tv_auth_token: Optional[str] = None
        self._tv_username: Optional[str] = None
        self._tv_password: Optional[str] = None

        logger.info("TvDatafeed source timezone: %s", TV_SOURCE_TZ_NAME)

        # Load TradingView credentials from env
        session_id = os.environ.get("TV_SESSION_TOKEN", "").strip()
        if session_id:
            logger.info("Resolving TradingView auth_token from sessionid cookie...")
            self._tv_auth_token = _resolve_auth_token(session_id)
            if self._tv_auth_token:
                logger.info(f"✅ TvDatafeed: AUTH TOKEN resolved (len={len(self._tv_auth_token)})")
            else:
                logger.warning("❌ Could not resolve auth_token — falling back to nologin")
        else:
            self._tv_username = os.environ.get("TV_USERNAME", "").strip() or None
            self._tv_password = os.environ.get("TV_PASSWORD", "").strip() or None
            if self._tv_username:
                logger.info(f"TvDatafeed: using LOGIN mode (user={self._tv_username})")
            else:
                logger.info("TvDatafeed: using NOLOGIN mode (limited 5M history)")

    def _fetch_with_timeout(self, symbol, exchange, tv_interval, n_bars,
                            timeout=20) -> Optional[pd.DataFrame]:
        """Spawn a separate process for the TvDatafeed call with hard timeout."""
        interval_name = tv_interval.name if hasattr(tv_interval, 'name') else str(tv_interval)

        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp_path = tmp.name
        tmp.close()

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        proc = multiprocessing.Process(
            target=_subprocess_fetch,
            args=(symbol, exchange, interval_name, n_bars, tmp_path,
                  self._tv_username, self._tv_password,
                  self._tv_auth_token),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            logger.warning(f"Timeout ({timeout}s) fetching {symbol} "
                           f"{interval_name} ({n_bars} bars) — killing")
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)

        if os.path.exists(tmp_path):
            try:
                df = pd.read_csv(tmp_path, index_col=0, parse_dates=True)
                return df
            except Exception:
                return None
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
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

        tv_interval = INTERVALS.get(interval.lower())
        if tv_interval is None:
            logger.error(f"Unknown interval: {interval}")
            return None

        n_bars_candidates = sorted(
            set([int(n_bars), min(int(n_bars), 1500), 800, 400]),
            reverse=True,
        )

        for attempt in range(1, 5):
            try:
                bars = n_bars_candidates[
                    min(attempt - 1, len(n_bars_candidates) - 1)
                ]
                logger.info(f"    {interval} ({bars} bars, attempt {attempt})...")
                df = self._fetch_with_timeout(
                    symbol, "OANDA", tv_interval, bars, timeout=20,
                )
                if df is None or df.empty:
                    time.sleep(1.5)
                    continue

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize(TV_SOURCE_TZ)
                df.index = df.index.tz_convert(UTC)
                ohlcv = ['open', 'high', 'low', 'close', 'volume']
                df = df[[c for c in ohlcv if c in df.columns]]
                self._cache[cache_key] = df
                return df
            except Exception as e:
                logger.error(f"Fetch error {symbol} {interval} "
                             f"attempt {attempt}: {e}")
                time.sleep(2.0)
        return None

    def fetch_multi_timeframe(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data = {}
        logger.info(f"  Fetching data for {symbol}...")
        data['4h'] = self.fetch_ohlcv(symbol, '4h', 200)
        data['1h'] = self.fetch_ohlcv(symbol, '1h', 500)
        data['15m'] = self.fetch_ohlcv(symbol, '15m', 1000)
        n_5m = 3000 if self._tv_auth_token else 800
        data['5m'] = self.fetch_ohlcv(symbol, '5m', n_5m)
        return data

    def clear_cache(self):
        self._cache.clear()


# ============================================================================
# BACKTEST RESULT
# ============================================================================

@dataclass
class BacktestResult:
    signal: Dict[str, Any]
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    exit_reason: str
    pnl_percent: float
    pnl_rr: float


@dataclass
class BacktestSummary:
    symbol: str
    total_signals: int
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_percent: float
    total_pnl_rr: float
    avg_win_percent: float
    avg_loss_percent: float
    profit_factor: float
    max_drawdown_percent: float


# ============================================================================
# BACKTESTER
# ============================================================================

class MSSObForexBacktester:
    def __init__(
        self,
        strategy: Optional[MSSOrderBlockStrategy] = None,
        max_trade_duration: timedelta = timedelta(hours=48),
    ):
        self.strategy = strategy or MSSOrderBlockStrategy()
        self.fetcher = ForexDataFetcher()
        self.max_trade_duration = max_trade_duration

    def _simulate_trade(
        self,
        signal: MSSOB_Signal,
        df_5m: pd.DataFrame,
    ) -> BacktestResult:
        """Simulate trade candle-by-candle on 5M data. Close 100% at 2R."""
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        tp = signal.take_profit
        entry_time = signal.datetime
        direction = signal.direction

        start_idx = df_5m.index.searchsorted(entry_time)
        if start_idx >= len(df_5m):
            return BacktestResult(
                signal=format_signal_for_jsonl(signal),
                entry_time=entry_time, entry_price=entry_price,
                exit_time=None, exit_price=None,
                exit_reason="pending", pnl_percent=0.0, pnl_rr=0.0,
            )

        if direction == Direction.BULLISH:
            initial_risk = entry_price - stop_loss
        else:
            initial_risk = stop_loss - entry_price

        if initial_risk <= 0:
            return BacktestResult(
                signal=format_signal_for_jsonl(signal),
                entry_time=entry_time, entry_price=entry_price,
                exit_time=entry_time, exit_price=entry_price,
                exit_reason="invalid_risk", pnl_percent=0.0, pnl_rr=0.0,
            )

        for i in range(start_idx, len(df_5m)):
            candle = df_5m.iloc[i]
            candle_time = df_5m.index[i]

            # Timeout
            if candle_time - entry_time > self.max_trade_duration:
                exit_price = candle['close']
                if direction == Direction.BULLISH:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    pnl_rr = (exit_price - entry_price) / initial_risk
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                    pnl_rr = (entry_price - exit_price) / initial_risk
                return BacktestResult(
                    signal=format_signal_for_jsonl(signal),
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=candle_time, exit_price=exit_price,
                    exit_reason="timeout",
                    pnl_percent=round(pnl_pct, 4), pnl_rr=round(pnl_rr, 2),
                )

            if direction == Direction.BULLISH:
                if candle['low'] <= stop_loss:
                    pnl_pct = (stop_loss - entry_price) / entry_price * 100
                    pnl_rr = (stop_loss - entry_price) / initial_risk
                    return BacktestResult(
                        signal=format_signal_for_jsonl(signal),
                        entry_time=entry_time, entry_price=entry_price,
                        exit_time=candle_time, exit_price=stop_loss,
                        exit_reason="sl",
                        pnl_percent=round(pnl_pct, 4), pnl_rr=round(pnl_rr, 2),
                    )
                if candle['high'] >= tp:
                    pnl_pct = (tp - entry_price) / entry_price * 100
                    pnl_rr = (tp - entry_price) / initial_risk
                    return BacktestResult(
                        signal=format_signal_for_jsonl(signal),
                        entry_time=entry_time, entry_price=entry_price,
                        exit_time=candle_time, exit_price=tp,
                        exit_reason="tp",
                        pnl_percent=round(pnl_pct, 4), pnl_rr=round(pnl_rr, 2),
                    )
            else:
                if candle['high'] >= stop_loss:
                    pnl_pct = (entry_price - stop_loss) / entry_price * 100
                    pnl_rr = (entry_price - stop_loss) / initial_risk
                    return BacktestResult(
                        signal=format_signal_for_jsonl(signal),
                        entry_time=entry_time, entry_price=entry_price,
                        exit_time=candle_time, exit_price=stop_loss,
                        exit_reason="sl",
                        pnl_percent=round(pnl_pct, 4), pnl_rr=round(pnl_rr, 2),
                    )
                if candle['low'] <= tp:
                    pnl_pct = (entry_price - tp) / entry_price * 100
                    pnl_rr = (entry_price - tp) / initial_risk
                    return BacktestResult(
                        signal=format_signal_for_jsonl(signal),
                        entry_time=entry_time, entry_price=entry_price,
                        exit_time=candle_time, exit_price=tp,
                        exit_reason="tp",
                        pnl_percent=round(pnl_pct, 4), pnl_rr=round(pnl_rr, 2),
                    )

        # Still open
        exit_price = df_5m.iloc[-1]['close']
        if direction == Direction.BULLISH:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_rr = (exit_price - entry_price) / initial_risk
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            pnl_rr = (entry_price - exit_price) / initial_risk
        return BacktestResult(
            signal=format_signal_for_jsonl(signal),
            entry_time=entry_time, entry_price=entry_price,
            exit_time=df_5m.index[-1], exit_price=exit_price,
            exit_reason="open",
            pnl_percent=round(pnl_pct, 4), pnl_rr=round(pnl_rr, 2),
        )

    def backtest_symbol(
        self,
        symbol: str,
        scan_lookback_hours: int = 200,
    ) -> List[BacktestResult]:
        logger.info(f"Backtesting {symbol}...")
        data = self.fetcher.fetch_multi_timeframe(symbol)

        for tf in ['4h', '1h', '15m', '5m']:
            if data.get(tf) is None or data[tf].empty:
                logger.warning(f"Missing {tf} data for {symbol}, skipping")
                return []

        df_4h = data['4h']
        df_1h = data['1h']
        df_15m = data['15m']
        df_5m = data['5m']

        results: List[BacktestResult] = []
        used_bias_keys = set()

        logger.info(f"  Scanning for signals on full data ({len(df_1h)} 1H bars)...")
        signals = self.strategy.generate_signal(
            symbol=symbol,
            df_4h=df_4h,
            df_1h=df_1h,
            df_15m=df_15m,
            df_5m=df_5m,
            lookback_window_hours=scan_lookback_hours,
        )
        logger.info(f"  Raw signals found: {len(signals)}")

        for sig in signals:
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
                f"Entry={sig.entry_price:.5f}, Exit={result.exit_reason}, "
                f"P&L={result.pnl_rr:.2f}R"
            )

        logger.info(f"  Completed {symbol}: {len(results)} trades")
        return results

    def calculate_summary(self, symbol: str, results: List[BacktestResult]) -> BacktestSummary:
        if not results:
            return BacktestSummary(symbol=symbol, total_signals=0, total_trades=0,
                                   wins=0, losses=0, win_rate=0, total_pnl_percent=0,
                                   total_pnl_rr=0, avg_win_percent=0, avg_loss_percent=0,
                                   profit_factor=0, max_drawdown_percent=0)

        completed = [r for r in results if r.exit_reason not in ('pending', 'open')]
        if not completed:
            return BacktestSummary(symbol=symbol, total_signals=len(results), total_trades=0,
                                   wins=0, losses=0, win_rate=0, total_pnl_percent=0,
                                   total_pnl_rr=0, avg_win_percent=0, avg_loss_percent=0,
                                   profit_factor=0, max_drawdown_percent=0)

        wins = [r for r in completed if r.pnl_percent > 0]
        losses = [r for r in completed if r.pnl_percent <= 0]
        total_pnl = sum(r.pnl_percent for r in completed)
        total_rr = sum(r.pnl_rr for r in completed)
        avg_win = sum(r.pnl_percent for r in wins) / len(wins) if wins else 0
        avg_loss = sum(abs(r.pnl_percent) for r in losses) / len(losses) if losses else 0
        gross_profit = sum(r.pnl_percent for r in wins)
        gross_loss = sum(abs(r.pnl_percent) for r in losses)
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        cum = peak = max_dd = 0
        for r in completed:
            cum += r.pnl_percent
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        return BacktestSummary(
            symbol=symbol, total_signals=len(results), total_trades=len(completed),
            wins=len(wins), losses=len(losses),
            win_rate=round(len(wins) / len(completed) * 100, 1) if completed else 0,
            total_pnl_percent=round(total_pnl, 4), total_pnl_rr=round(total_rr, 2),
            avg_win_percent=round(avg_win, 4), avg_loss_percent=round(avg_loss, 4),
            profit_factor=round(pf, 2), max_drawdown_percent=round(max_dd, 4),
        )

    def run_backtest(
        self,
        symbols: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        scan_lookback_hours: int = 200,
    ) -> Dict[str, Any]:
        symbols = symbols or FOREX_PAIRS
        if output_file is None:
            output_file = Path(__file__).parent / "results" / "mss_ob_forex_backtest_signals.jsonl"
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results: List[BacktestResult] = []
        all_summaries: List[BacktestSummary] = []

        logger.info(f"Starting MSS+OB FOREX backtest on {len(symbols)} pairs...")

        for symbol in tqdm(symbols, desc="Backtesting MSS+OB Forex"):
            try:
                results = self.backtest_symbol(symbol, scan_lookback_hours)
                all_results.extend(results)
                all_summaries.append(self.calculate_summary(symbol, results))
                self.fetcher.clear_cache()
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
                continue

        with open(output_path, 'w', encoding='utf-8') as f:
            for r in all_results:
                out = {
                    **r.signal,
                    "backtest_exit_time_ist": r.exit_time.astimezone(IST).strftime(
                        "%Y-%m-%d %H:%M:%S IST") if r.exit_time else None,
                    "backtest_exit_price": r.exit_price,
                    "backtest_exit_reason": r.exit_reason,
                    "backtest_pnl_percent": r.pnl_percent,
                    "backtest_pnl_rr": r.pnl_rr,
                }
                f.write(json.dumps(out, default=str) + '\n')

        logger.info(f"Saved {len(all_results)} signals to {output_path}")
        self._print_summary(all_summaries)
        return {"results": all_results, "summaries": all_summaries,
                "output_file": str(output_path)}

    def _print_summary(self, summaries: List[BacktestSummary]):
        print("\n" + "=" * 100)
        print("FOREX BACKTEST SUMMARY — MSS + ORDER BLOCK STRATEGY (Path B)")
        print("=" * 100)
        print(f"\n{'Symbol':<12} {'Signals':>8} {'Trades':>7} {'Wins':>5} "
              f"{'Losses':>7} {'WR%':>7} {'PnL%':>9} {'PnL RR':>8} {'PF':>6}")
        print("-" * 100)

        ts = tt = tw = tl = 0
        tp = tr = 0.0
        for s in summaries:
            print(
                f"{s.symbol:<12} {s.total_signals:>8} {s.total_trades:>7} "
                f"{s.wins:>5} {s.losses:>7} {s.win_rate:>6.1f}% "
                f"{s.total_pnl_percent:>8.4f}% {s.total_pnl_rr:>7.2f}R "
                f"{s.profit_factor:>6.2f}"
            )
            ts += s.total_signals; tt += s.total_trades
            tw += s.wins; tl += s.losses
            tp += s.total_pnl_percent; tr += s.total_pnl_rr

        print("-" * 100)
        wr = tw / tt * 100 if tt > 0 else 0
        print(f"{'TOTAL':<12} {ts:>8} {tt:>7} {tw:>5} {tl:>7} "
              f"{wr:>6.1f}% {tp:>8.4f}% {tr:>7.2f}R")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Backtest MSS+OB Strategy on Forex")
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--lookback", type=int, default=200)
    args = parser.parse_args()

    strategy = MSSOrderBlockStrategy()
    backtester = MSSObForexBacktester(strategy=strategy)
    result = backtester.run_backtest(
        symbols=args.pairs,
        output_file=args.output,
        scan_lookback_hours=args.lookback,
    )
    print(f"\nBacktest complete! Results saved to: {result['output_file']}")


if __name__ == "__main__":
    main()
