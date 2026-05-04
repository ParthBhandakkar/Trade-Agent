"""
Strategy 09: MSS + Order Block Entry — Forex Backtester
========================================================

Backtests the MSS + OB strategy on forex pairs using
Exness data via MetaTrader 5.

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pytz
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

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC
MT5_SOURCE_TZ_NAME = os.environ.get("MT5_SOURCE_TZ", "UTC").strip() or "UTC"
try:
    MT5_SOURCE_TZ = pytz.timezone(MT5_SOURCE_TZ_NAME)
except Exception:
    MT5_SOURCE_TZ_NAME = "UTC"
    MT5_SOURCE_TZ = UTC

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

# ============================================================================
# DATA FETCHING (MetaTrader 5 / Exness)
# ============================================================================

MT5_INTERVAL_NAMES = {
    "1m": "TIMEFRAME_M1",
    "3m": "TIMEFRAME_M3",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
}


class ForexDataFetcher:
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._mt5 = None
        self._connected = False

        self._login = int(os.getenv("MT5_LOGIN", os.getenv("XM_MT5_LOGIN", "0")) or 0)
        self._password = os.getenv("MT5_PASSWORD", os.getenv("XM_MT5_PASSWORD", ""))
        self._server = os.getenv("MT5_SERVER", os.getenv("XM_MT5_SERVER", ""))
        self._symbol_suffix = os.getenv(
            "MT5_SYMBOL_SUFFIX",
            os.getenv("EXNESS_MT5_SYMBOL_SUFFIX", ""),
        ).strip()

        self._source_tz_name = os.getenv("MT5_SOURCE_TZ", "UTC").strip() or "UTC"
        self._source_tz = MT5_SOURCE_TZ

        logger.info("MT5 source timezone: %s", self._source_tz_name)
        if self._symbol_suffix:
            logger.info("MT5 symbol suffix enabled: %s", self._symbol_suffix)

        self.connect()

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.error("MetaTrader5 library not installed. Install MetaTrader5 first.")
            return False

        self._mt5 = mt5
        if not mt5.initialize():
            logger.error("MT5 init failed: %s", mt5.last_error())
            return False

        if self._login and self._password:
            if not mt5.login(login=self._login, password=self._password, server=self._server or None):
                logger.error("MT5 login failed: %s", mt5.last_error())
                mt5.shutdown()
                return False

        info = mt5.account_info()
        if info is None:
            logger.warning("MT5 initialized but account_info() returned None")
        else:
            logger.info(
                "✅ MT5 connected: %s | Balance: %s %s | Leverage: 1:%s",
                info.name,
                info.balance,
                info.currency,
                getattr(info, "leverage", "?")
            )

        self._connected = True
        return True

    def ensure_connected(self) -> bool:
        if self._connected and self._mt5 is not None:
            try:
                if self._mt5.account_info() is not None:
                    return True
            except Exception:
                pass

        self._connected = False
        try:
            if self._mt5 is not None:
                self._mt5.shutdown()
        except Exception:
            pass
        return self.connect()

    def _timeframe(self, interval: str):
        if self._mt5 is None:
            return None
        attr = MT5_INTERVAL_NAMES.get(interval.lower())
        if attr is None:
            return None
        return getattr(self._mt5, attr, None)

    def _resolve_symbol(self, symbol: str) -> Optional[str]:
        if self._mt5 is None:
            return None

        base = (symbol or "").upper().strip()
        if not base:
            return None

        candidates = [base]
        if self._symbol_suffix:
            candidates.append(f"{base}{self._symbol_suffix}")

        try:
            symbols = self._mt5.symbols_get() or []
            available = {getattr(s, "name", "") for s in symbols}
        except Exception:
            available = set()

        for candidate in candidates:
            if candidate in available:
                return candidate
            try:
                if self._mt5.symbol_select(candidate, True):
                    return candidate
            except Exception:
                pass

        return candidates[0] if candidates[0] in available else None

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

        if not self.ensure_connected():
            return None

        timeframe = self._timeframe(interval)
        if timeframe is None:
            logger.error("Unknown interval: %s", interval)
            return None

        mt5_symbol = self._resolve_symbol(symbol)
        if not mt5_symbol:
            logger.error("Symbol not available in MT5: %s", symbol)
            return None

        bars_candidates = sorted(
            set([int(n_bars), min(int(n_bars), 1500), 800, 400]),
            reverse=True,
        )

        for attempt in range(1, 5):
            try:
                bars = bars_candidates[min(attempt - 1, len(bars_candidates) - 1)]
                logger.info("    %s (%s bars, attempt %s)...", interval, bars, attempt)

                rates = self._mt5.copy_rates_from_pos(mt5_symbol, timeframe, 0, bars)
                if rates is None or len(rates) == 0:
                    time.sleep(1.0)
                    continue

                df = pd.DataFrame(rates)
                if df.empty or "time" not in df.columns:
                    time.sleep(1.0)
                    continue

                df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("datetime", inplace=True)

                if df.index.tz is None:
                    df.index = df.index.tz_localize(self._source_tz)
                df.index = df.index.tz_convert(UTC)

                if "volume" not in df.columns:
                    if "tick_volume" in df.columns:
                        df["volume"] = df["tick_volume"]
                    elif "real_volume" in df.columns:
                        df["volume"] = df["real_volume"]

                ohlcv = ["open", "high", "low", "close", "volume"]
                df = df[[c for c in ohlcv if c in df.columns]]
                self._cache[cache_key] = df
                return df
            except Exception as e:
                logger.error("Fetch error %s %s attempt %s: %s", symbol, interval, attempt, e)
                time.sleep(1.5)

        return None

    def fetch_multi_timeframe(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data = {}
        logger.info("  Fetching MT5 data for %s...", symbol)
        data["4h"] = self.fetch_ohlcv(symbol, "4h", 200)
        data["1h"] = self.fetch_ohlcv(symbol, "1h", 500)
        data["15m"] = self.fetch_ohlcv(symbol, "15m", 1000)
        data["5m"] = self.fetch_ohlcv(symbol, "5m", 3000)
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
