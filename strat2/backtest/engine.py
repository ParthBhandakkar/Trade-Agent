from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

BACKTEST_DIR = Path(__file__).resolve().parent
STRAT2_DIR = BACKTEST_DIR.parent
REPO_ROOT = STRAT2_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strat2.auto_trader import (  # noqa: E402
    Config,
    DEFAULT_FOREX_PAIRS,
    TradeSignal,
    build_signal,
    pip_size,
    to_ist_str,
    UTC,
    IST,
)
from strat2.backtest.data_cache import (  # noqa: E402
    MT5HistoryCache,
    TIMEFRAME_MINUTES,
    clean_symbol,
)


TIMEFRAMES = ("D1", "H4", "H1", "M15", "M5")
TAIL_BARS = {"D1": 320, "H4": 620, "H1": 900, "M15": 1600, "M5": 2600}
FETCH_WARMUP_DAYS = {"D1": 520, "H4": 150, "H1": 70, "M15": 20, "M5": 14}


@dataclass
class BacktestConfig:
    symbols: List[str]
    start_utc: datetime
    end_utc: datetime
    risk_inr: float = 500.0
    min_quality: int = 78
    force_refresh: bool = False


class HistoricalBridge:
    def __init__(self, cache: MT5HistoryCache):
        self.cache = cache
        self.current_symbol: Optional[str] = None
        self.current_spread_points: float = 0.0

    def set_context(self, symbol: str, m5_row: pd.Series) -> None:
        self.current_symbol = clean_symbol(symbol)
        self.current_spread_points = float(m5_row.get("spread", 0.0) or 0.0)

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        return self.cache.resolve_symbol(symbol) or clean_symbol(symbol)

    def symbol_info(self, symbol: str) -> Any:
        return self.cache.symbol_info(symbol)

    def spread_pips(self, symbol: str) -> Optional[float]:
        info = self.symbol_info(symbol)
        pip = pip_size(symbol, info)
        point = float(getattr(info, "point", 0.00001) or 0.00001) if info else (0.01 if "JPY" in symbol.upper() else 0.00001)
        return abs(self.current_spread_points * point / pip)


def default_strategy_config(risk_inr: float = 500.0, min_quality: int = 78) -> Config:
    return Config(
        mt5_login=0,
        mt5_password="",
        mt5_server="",
        pairs=[],
        poll_seconds=60,
        dry_run=True,
        magic=909102,
        risk_inr=float(risk_inr),
        risk_pct_equity=0.0,
        max_risk_inr=max(float(risk_inr), 1000.0),
        max_open_positions=2,
        max_daily_loss_inr=1500.0,
        min_quality=int(min_quality),
        max_spread_pips=2.2,
        min_stop_pips=6.0,
        sweep_lookback_15m=96,
        max_sweep_age_min=240,
        max_fvg_age_min=180,
        fvg_touch_tolerance_pips=0.5,
        tp1_r=1.0,
        tp2_min_r=1.8,
        tp1_lot_fraction=0.70,
        allow_asian=False,
        allow_london=True,
        allow_ny=True,
    )


def parse_date_ist(date_text: str, end_of_day: bool = False) -> datetime:
    raw = date_text.strip()
    if "T" in raw:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        return dt.astimezone(UTC)
    day = datetime.strptime(raw, "%Y-%m-%d").date()
    local_time = time(23, 59, 59) if end_of_day else time(0, 0, 0)
    return datetime.combine(day, local_time, tzinfo=IST).astimezone(UTC)


def slice_to_time(df: pd.DataFrame, current_ts: pd.Timestamp, bars: int) -> pd.DataFrame:
    end_pos = df.index.searchsorted(current_ts, side="right")
    start_pos = max(0, end_pos - bars)
    return df.iloc[start_pos:end_pos].copy()


def signal_events(signal: TradeSignal) -> List[Dict[str, Any]]:
    events = [
        {
            "time_ist": to_ist_str(signal.trend.timestamp if hasattr(signal.trend, "timestamp") else signal.timestamp),
            "phase": "Trend",
            "detail": signal.trend.reason,
            "price": None,
        },
        {
            "time_ist": to_ist_str(signal.sweep.level_time),
            "phase": "Reference liquidity",
            "detail": signal.sweep.source,
            "price": round(signal.sweep.level_price, 6),
        },
        {
            "time_ist": to_ist_str(signal.sweep.timestamp),
            "phase": "Liquidity sweep",
            "detail": f"{signal.direction} sweep and reclaim",
            "price": round(signal.sweep.extreme_price, 6),
        },
        {
            "time_ist": to_ist_str(signal.break_signal.timestamp),
            "phase": "5M break",
            "detail": f"Closed beyond {signal.break_signal.break_level:.6f}; body ATR {signal.break_signal.body_atr:.2f}",
            "price": round(signal.break_signal.close_price, 6),
        },
        {
            "time_ist": to_ist_str(signal.fvg.timestamp),
            "phase": "5M FVG",
            "detail": f"FVG {signal.fvg.low:.6f} - {signal.fvg.high:.6f}, CE {signal.fvg.ce:.6f}",
            "price": round(signal.fvg.ce, 6),
        },
        {
            "time_ist": to_ist_str(signal.timestamp),
            "phase": "Entry",
            "detail": f"{signal.direction} rejection into FVG, quality {signal.quality}",
            "price": round(signal.entry_model_price, 6),
        },
    ]
    return sorted(events, key=lambda item: item["time_ist"] or "")


def simulate_trade(
    signal: TradeSignal,
    m5: pd.DataFrame,
    start_pos: int,
    cfg: Config,
) -> Dict[str, Any]:
    direction = signal.direction
    entry = float(signal.entry_model_price)
    sl = float(signal.sl)
    tp1 = float(signal.tp1)
    tp2 = float(signal.tp2)
    initial_risk = abs(entry - sl)
    tp2_r = abs(tp2 - entry) / initial_risk if initial_risk > 0 else cfg.tp2_min_r
    tp1_fraction = float(cfg.tp1_lot_fraction)
    runner_fraction = max(0.0, 1.0 - tp1_fraction)

    tp1_hit = False
    tp1_time = None
    exit_time = None
    exit_price = None
    exit_reason = "OPEN_AT_END"
    pnl_r = 0.0

    future = m5.iloc[start_pos + 1 :]
    last_close = entry
    last_ts = signal.timestamp

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        last_close = close
        last_ts = ts

        if direction == "bullish":
            hit_sl = low <= sl
            hit_tp1 = high >= tp1
            hit_be = low <= entry
            hit_tp2 = high >= tp2
        else:
            hit_sl = high >= sl
            hit_tp1 = low <= tp1
            hit_be = high >= entry
            hit_tp2 = low <= tp2

        if not tp1_hit:
            if hit_sl:
                exit_time = ts
                exit_price = sl
                exit_reason = "SL"
                pnl_r = -1.0
                break
            if hit_tp1:
                tp1_hit = True
                tp1_time = ts
                pnl_r = tp1_fraction
                if hit_tp2:
                    exit_time = ts
                    exit_price = tp2
                    exit_reason = "TP1_TP2"
                    pnl_r = tp1_fraction + runner_fraction * tp2_r
                    break
            continue

        if hit_be:
            exit_time = ts
            exit_price = entry
            exit_reason = "TP1_BE"
            pnl_r = tp1_fraction
            break
        if hit_tp2:
            exit_time = ts
            exit_price = tp2
            exit_reason = "TP1_TP2"
            pnl_r = tp1_fraction + runner_fraction * tp2_r
            break

    if exit_time is None:
        exit_time = last_ts
        exit_price = last_close
        if direction == "bullish":
            open_r = (last_close - entry) / initial_risk if initial_risk else 0.0
        else:
            open_r = (entry - last_close) / initial_risk if initial_risk else 0.0
        pnl_r = tp1_fraction + runner_fraction * open_r if tp1_hit else open_r
        exit_reason = "TP1_OPEN_AT_END" if tp1_hit else "OPEN_AT_END"

    events = signal_events(signal)
    if tp1_time is not None:
        events.append({
            "time_ist": to_ist_str(tp1_time),
            "phase": "TP1",
            "detail": f"Booked {tp1_fraction:.0%}; runner SL moves to breakeven",
            "price": round(tp1, 6),
        })
    events.append({
        "time_ist": to_ist_str(exit_time),
        "phase": "Exit",
        "detail": exit_reason,
        "price": round(float(exit_price), 6),
    })

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk": initial_risk,
        "tp1_time_ist": to_ist_str(tp1_time) if tp1_time is not None else None,
        "exit_time_ist": to_ist_str(exit_time),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "pnl_r": round(float(pnl_r), 3),
        "events": sorted(events, key=lambda item: item["time_ist"] or ""),
    }


def run_backtest(config: BacktestConfig) -> Dict[str, Any]:
    cache = MT5HistoryCache()
    strategy_cfg = default_strategy_config(config.risk_inr, config.min_quality)
    bridge = HistoricalBridge(cache)
    cache_infos: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Let trades close after the selected window when historical data exists,
    # but do not ask MT5 for future candles on an in-progress day.
    fetch_end = min(config.end_utc + timedelta(days=2), datetime.now(tz=UTC))

    for raw_symbol in config.symbols:
        symbol = clean_symbol(raw_symbol)
        try:
            frames: Dict[str, pd.DataFrame] = {}
            for tf in TIMEFRAMES:
                fetch_start = config.start_utc - timedelta(days=FETCH_WARMUP_DAYS[tf])
                df, info = cache.ensure_history(symbol, tf, fetch_start, fetch_end, config.force_refresh)
                frames[tf] = df
                cache_infos.append({**asdict(info), "path": str(info.path)})

            m5 = frames["M5"]
            scan_m5 = m5[(m5.index >= config.start_utc) & (m5.index <= config.end_utc)]
            sent_ids = set()
            skip_until: Optional[pd.Timestamp] = None
            symbol_trades = 0

            for current_ts, m5_row in scan_m5.iterrows():
                if skip_until is not None and current_ts <= skip_until:
                    continue
                data = {
                    tf: slice_to_time(frames[tf], current_ts, TAIL_BARS[tf])
                    for tf in TIMEFRAMES
                }
                if any(df.empty for df in data.values()):
                    continue
                bridge.set_context(symbol, m5_row)
                signal = build_signal(symbol, data, bridge, strategy_cfg)
                if signal is None:
                    continue
                signal_id = signal.signal_id()
                if signal_id in sent_ids:
                    continue
                sent_ids.add(signal_id)

                start_pos = m5.index.searchsorted(current_ts, side="left")
                outcome = simulate_trade(signal, m5, int(start_pos), strategy_cfg)
                symbol_trades += 1
                skip_until = pd.Timestamp(outcome["exit_time_ist"].replace(" IST", ""), tz=IST).tz_convert(UTC) if outcome["exit_time_ist"] else current_ts
                pnl_inr = round(float(outcome["pnl_r"]) * float(config.risk_inr), 2)
                trades.append({
                    "id": f"{symbol}-{symbol_trades}-{signal.timestamp.isoformat()}",
                    "symbol": symbol,
                    "direction": "BUY" if signal.direction == "bullish" else "SELL",
                    "entry_time_ist": to_ist_str(signal.timestamp),
                    "entry": round(outcome["entry"], 6),
                    "sl": round(outcome["sl"], 6),
                    "tp1": round(outcome["tp1"], 6),
                    "tp2": round(outcome["tp2"], 6),
                    "quality": signal.quality,
                    "session": signal.session,
                    "spread_pips": round(signal.spread_pips, 2),
                    "exit_time_ist": outcome["exit_time_ist"],
                    "exit_reason": outcome["exit_reason"],
                    "exit_price": round(outcome["exit_price"], 6),
                    "pnl_r": outcome["pnl_r"],
                    "pnl_inr": pnl_inr,
                    "events": outcome["events"],
                    "signal": signal.to_json(),
                })
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")

    wins = [t for t in trades if t["pnl_r"] > 0]
    losses = [t for t in trades if t["pnl_r"] < 0]
    total_pnl = round(sum(float(t["pnl_inr"]) for t in trades), 2)
    total_r = round(sum(float(t["pnl_r"]) for t in trades), 3)
    summary = {
        "symbols": [clean_symbol(s) for s in config.symbols],
        "start_ist": config.start_utc.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "end_ist": config.end_utc.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "risk_inr": config.risk_inr,
        "min_quality": config.min_quality,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round((len(wins) / len(trades) * 100.0), 2) if trades else 0.0,
        "total_r": total_r,
        "total_pnl_inr": total_pnl,
        "avg_r": round(total_r / len(trades), 3) if trades else 0.0,
    }
    return {"summary": summary, "trades": trades, "cache": cache_infos, "errors": errors}


def symbols_from_text(value: str) -> List[str]:
    if not value.strip():
        return DEFAULT_FOREX_PAIRS[:6]
    return [clean_symbol(part) for part in value.replace(";", ",").split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Strat2 historical backtest")
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY", help="Comma-separated symbols")
    parser.add_argument("--start", required=True, help="Start date in IST, YYYY-MM-DD or ISO datetime")
    parser.add_argument("--end", required=True, help="End date in IST, YYYY-MM-DD or ISO datetime")
    parser.add_argument("--risk-inr", type=float, default=500.0)
    parser.add_argument("--min-quality", type=int, default=78)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of a table")
    args = parser.parse_args()

    result = run_backtest(BacktestConfig(
        symbols=symbols_from_text(args.symbols),
        start_utc=parse_date_ist(args.start),
        end_utc=parse_date_ist(args.end, end_of_day=True),
        risk_inr=args.risk_inr,
        min_quality=args.min_quality,
        force_refresh=args.force_refresh,
    ))
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return
    print(json.dumps(result["summary"], indent=2))
    for trade in result["trades"]:
        print(
            f"{trade['entry_time_ist']} {trade['symbol']} {trade['direction']} "
            f"entry={trade['entry']} exit={trade['exit_reason']} pnl={trade['pnl_r']}R/{trade['pnl_inr']} INR"
        )
    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
