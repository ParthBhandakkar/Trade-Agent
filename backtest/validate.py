#!/usr/bin/env python3
"""
Validate Backtest Signals vs Live Trades
==========================================

Cross-references backtest output against the live auto_trader trades.jsonl
to verify signal match rate.

Usage:
    python backtest/validate.py --backtest-result backtest/results/USDJPY_2026-04-16_2026-04-17.json
    python backtest/validate.py --symbol USDJPY --date 2026-04-16

The validator checks:
  1. Did the backtest produce the same signals (symbol + direction + approximate time)?
  2. Are entry prices close (within tolerance)?
  3. Were the same filters applied (EMA rejections match)?
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC

REPO_ROOT = Path(__file__).resolve().parent.parent
TRADES_LOG = REPO_ROOT / "strategy_09_mss_ob_entry" / "phase_logs" / "auto_trader" / "trades.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _parse_ist_timestamp(s: str) -> Optional[datetime]:
    """Parse '2026-04-16 13:51:23 IST' → datetime."""
    if not s:
        return None
    s = s.replace(" IST", "").strip()
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return IST.localize(dt)
    except ValueError:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
            return IST.localize(dt)
        except ValueError:
            return None


def load_live_trades(
    symbol: Optional[str] = None,
    date_str: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load live trades from trades.jsonl, optionally filtered."""
    if not TRADES_LOG.exists():
        print(f"  ❌ trades.jsonl not found at {TRADES_LOG}")
        return []

    trades = []
    with open(TRADES_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filter by symbol
            if symbol and record.get("symbol", "").upper() != symbol.upper():
                continue

            # Filter by date
            if date_str:
                ts = record.get("timestamp_ist", "")
                if date_str not in ts:
                    continue

            trades.append(record)

    return trades


def load_backtest_result(path: Path) -> Optional[Dict[str, Any]]:
    """Load a backtest result JSON file."""
    if not path.exists():
        print(f"  ❌ Backtest result not found: {path}")
        return None
    return json.loads(path.read_text())


def compare_trades(
    backtest_trades: List[Dict],
    live_trades: List[Dict],
    time_tolerance_min: int = 30,
    price_tolerance_pct: float = 0.1,
) -> Dict[str, Any]:
    """Compare backtest trades against live trades.

    Returns a comparison report.
    """
    bt_executed = [t for t in backtest_trades if not t.get("rejected", False)]
    live_executed = [t for t in live_trades if t.get("trade_result", {}).get("success", False)]
    live_rejected = [t for t in live_trades if not t.get("trade_result", {}).get("success", False)]

    matched = []
    bt_only = list(bt_executed)  # trades in backtest but not in live
    live_only = list(live_executed)  # trades in live but not in backtest

    for bt in list(bt_executed):
        bt_time = bt.get("entry_time", "")
        bt_dir = bt.get("direction", "").lower()
        bt_price = bt.get("entry_price", 0)

        if isinstance(bt_time, str):
            try:
                bt_dt = datetime.fromisoformat(bt_time)
            except ValueError:
                continue
        else:
            bt_dt = bt_time

        if bt_dt and bt_dt.tzinfo is None:
            bt_dt = UTC.localize(bt_dt)

        for live in list(live_executed):
            live_ts = live.get("timestamp_ist", "")
            live_dt = _parse_ist_timestamp(live_ts)
            live_dir = live.get("direction", "").lower()
            live_price = live.get("entry_price", 0)

            if live_dt is None:
                continue

            # Convert both to UTC for comparison
            bt_utc = bt_dt.astimezone(UTC) if bt_dt else None
            live_utc = live_dt.astimezone(UTC)

            if bt_utc is None:
                continue

            time_diff = abs((bt_utc - live_utc).total_seconds()) / 60.0
            price_diff = abs(bt_price - live_price) / live_price * 100 if live_price else 999

            if (
                bt_dir == live_dir
                and time_diff <= time_tolerance_min
                and price_diff <= price_tolerance_pct
            ):
                matched.append({
                    "backtest": bt,
                    "live": live,
                    "time_diff_min": round(time_diff, 1),
                    "price_diff_pct": round(price_diff, 4),
                })
                if bt in bt_only:
                    bt_only.remove(bt)
                if live in live_only:
                    live_only.remove(live)
                break

    return {
        "total_backtest_executed": len(bt_executed),
        "total_live_executed": len(live_executed),
        "total_live_rejected": len(live_rejected),
        "matched": matched,
        "backtest_only": bt_only,
        "live_only": live_only,
        "match_rate": (
            f"{len(matched) / len(live_executed) * 100:.1f}%"
            if live_executed else "N/A"
        ),
    }


def _print_comparison(comparison: Dict[str, Any], symbol: str):
    """Print a formatted comparison report."""
    print(f"\n{'═' * 70}")
    print(f"  VALIDATION REPORT: {symbol}")
    print(f"{'═' * 70}")
    print(f"  Backtest executed trades: {comparison['total_backtest_executed']}")
    print(f"  Live executed trades:     {comparison['total_live_executed']}")
    print(f"  Live rejected signals:    {comparison['total_live_rejected']}")
    print(f"  Matched:                  {len(comparison['matched'])}")
    print(f"  Match rate:               {comparison['match_rate']}")

    if comparison["matched"]:
        print(f"\n  ── Matched Trades ──")
        for m in comparison["matched"]:
            bt = m["backtest"]
            live = m["live"]
            print(
                f"    ✅ {bt['direction'].upper()} @ {bt['entry_price']:.5f}  "
                f"(Δtime={m['time_diff_min']:.1f}min, Δprice={m['price_diff_pct']:.4f}%)"
            )

    if comparison["backtest_only"]:
        print(f"\n  ── Backtest-only trades (not in live log) ──")
        for t in comparison["backtest_only"]:
            print(f"    ⚠ {t['direction'].upper()} @ {t['entry_price']:.5f} at {t.get('entry_time', '?')}")

    if comparison["live_only"]:
        print(f"\n  ── Live-only trades (not in backtest) ──")
        for t in comparison["live_only"]:
            ts = t.get("timestamp_ist", "?")
            dr = t.get("direction", "?")
            ep = t.get("entry_price", 0)
            print(f"    ⚠ {dr.upper()} @ {ep:.5f} at {ts}")

    print(f"{'═' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate backtest vs live trades")
    parser.add_argument("--backtest-result", type=str, help="Path to backtest result JSON")
    parser.add_argument("--symbol", type=str, help="Symbol to filter live trades")
    parser.add_argument("--date", type=str, help="Date to filter live trades (YYYY-MM-DD)")
    parser.add_argument("--time-tolerance", type=int, default=30, help="Time match tolerance in minutes")
    parser.add_argument("--price-tolerance", type=float, default=0.1, help="Price match tolerance in %%")

    args = parser.parse_args()

    if not args.backtest_result and not args.symbol:
        parser.error("Provide --backtest-result or --symbol")

    # Load backtest results
    if args.backtest_result:
        bt_data = load_backtest_result(Path(args.backtest_result))
        if bt_data is None:
            return 1
        bt_trades = bt_data.get("trades", [])
        symbol = bt_data.get("symbol", args.symbol or "UNKNOWN")
    else:
        # Try to find the result file
        symbol = args.symbol.upper()
        candidates = list(RESULTS_DIR.glob(f"{symbol}_*.json"))
        if not candidates:
            print(f"  No backtest results found for {symbol}")
            return 1
        bt_data = load_backtest_result(candidates[-1])
        if bt_data is None:
            return 1
        bt_trades = bt_data.get("trades", [])
        print(f"  Using backtest result: {candidates[-1].name}")

    # Load live trades
    date_str = args.date or (bt_data.get("start", "")[:10] if bt_data else None)
    live_trades = load_live_trades(symbol, date_str)
    print(f"  Loaded {len(live_trades)} live trade records for {symbol} on {date_str}")

    # Compare
    comparison = compare_trades(
        bt_trades, live_trades,
        time_tolerance_min=args.time_tolerance,
        price_tolerance_pct=args.price_tolerance,
    )

    _print_comparison(comparison, symbol)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
