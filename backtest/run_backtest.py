#!/usr/bin/env python3
"""
Strategy 09 Backtester — CLI Runner
=====================================

Usage:
    # Full backtest on a pair
    python backtest/run_backtest.py --symbol USDJPY --start 2026-04-01 --end 2026-05-01

    # Validate against a specific day (compare with live trades.jsonl)
    python backtest/run_backtest.py --symbol USDJPY --start 2026-04-16 --end 2026-04-17 --verbose

    # Disable specific filters
    python backtest/run_backtest.py --symbol USDJPY --start 2026-04-01 --end 2026-05-01 --no-ema

    # List available symbols
    python backtest/run_backtest.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytz

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data_loader import load_all_timeframes, list_available_symbols, detect_market
from backtest.backtester import run_backtest, BacktestResult, BacktestTrade

UTC = pytz.UTC
IST = pytz.timezone("Asia/Kolkata")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Formatting helpers ────────────────────────────────────────────────

def _fmt_time(dt, tz=IST):
    """Format datetime to IST string."""
    if dt is None:
        return "—"
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M IST")


def _print_trade(i: int, t: BacktestTrade, pfmt: str = ".5f"):
    """Print a single trade line."""
    arrow = "↗" if t.direction in ("bullish",) else "↘"
    status_emoji = {
        "TP": "✅", "SL": "❌", "TIMEOUT": "⏱",
    }.get(t.outcome, "—")

    if t.rejected:
        print(
            f"  {i:3d}. ⊘ {_fmt_time(t.entry_time)}  "
            f"{t.direction.upper():8s} @ {t.entry_price:{pfmt}}  "
            f"Q={t.quality:2d}  REJECTED: {t.reject_reason}"
        )
    else:
        print(
            f"  {i:3d}. {status_emoji} {_fmt_time(t.entry_time)}  "
            f"{arrow} {t.direction.upper():8s} @ {t.entry_price:{pfmt}}  "
            f"SL={t.sl_price:{pfmt}}  TP={t.tp_price:{pfmt}}  "
            f"Q={t.quality:2d}  → {t.outcome} ({t.pnl_pct:+.3f}% / ₹{t.pnl_inr:+,.0f})"
        )
        if t.exit_time:
            print(f"       Exit: {_fmt_time(t.exit_time)} @ {t.exit_price:{pfmt}}")


def _print_summary(result: BacktestResult):
    """Print the backtest summary statistics."""
    s = result.summary()
    ex = result.executed_trades
    wins = result.winning_trades
    losses = result.losing_trades

    print(f"\n{'═' * 70}")
    print(f"  BACKTEST SUMMARY: {s['symbol']}")
    print(f"  Period: {s['period']}")
    print(f"{'═' * 70}")
    print(f"  Signals found:    {s['signals_found']}")
    print(f"  Signals rejected: {s['signals_rejected']}")
    print(f"  Trades executed:  {s['trades_executed']}")
    print(f"  ─────────────────────────")
    print(f"  Wins:    {s['wins']}  ({s['win_rate']})")
    print(f"  Losses:  {len(losses)}")
    print(f"  Total PnL:  {s['total_pnl_pct']}")
    print(f"  Avg PnL:    {s['avg_pnl_pct']}")
    total_inr = sum(t.pnl_inr for t in ex)
    print(f"  Total PnL (₹): {total_inr:+,.0f}  (₹1000 × 2000x)")
    if ex:
        print(f"  Avg PnL (₹):   {total_inr / len(ex):+,.0f}/trade")

    if ex:
        pnl_list = [t.pnl_pct for t in ex]
        cum_pnl = []
        total = 0
        for p in pnl_list:
            total += p
            cum_pnl.append(total)
        max_dd = 0
        peak = 0
        for c in cum_pnl:
            if c > peak:
                peak = c
            dd = peak - c
            if dd > max_dd:
                max_dd = dd
        print(f"  Max Drawdown: {max_dd:.3f}%")
        if wins:
            avg_win = sum(t.pnl_pct for t in wins) / len(wins)
            print(f"  Avg Win:  {avg_win:+.3f}%")
        if losses:
            avg_loss = sum(t.pnl_pct for t in losses) / len(losses)
            print(f"  Avg Loss: {avg_loss:+.3f}%")

    # Rejection breakdown
    rejected = [t for t in result.trades if t.rejected]
    if rejected:
        reasons = {}
        for t in rejected:
            key = t.reject_reason.split(":")[0] if ":" in t.reject_reason else t.reject_reason
            reasons[key] = reasons.get(key, 0) + 1
        print(f"\n  ── Rejection Breakdown ──")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    print(f"{'═' * 70}\n")


def _save_results(result: BacktestResult, output_path: Path):
    """Save results to JSON for later comparison."""
    data = {
        "symbol": result.symbol,
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        "summary": result.summary(),
        "trades": [],
    }
    for t in result.trades:
        trade_dict = {
            "symbol": t.symbol,
            "market": t.market,
            "direction": t.direction,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "entry_price": t.entry_price,
            "sl_price": t.sl_price,
            "tp_price": t.tp_price,
            "sl_reason": t.sl_reason,
            "quality": t.quality,
            "ob_top": t.ob_top,
            "ob_bottom": t.ob_bottom,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "exit_price": t.exit_price,
            "outcome": t.outcome,
            "pnl_pct": t.pnl_pct,
            "pnl_inr": t.pnl_inr,
            "rejected": t.rejected,
            "reject_reason": t.reject_reason,
            "ema_detail": t.ema_detail,
            "killzone": t.killzone,
        }
        data["trades"].append(trade_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\n  Results saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Strategy 09 (MSS + OB) Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", type=str, help="Symbol to backtest (e.g. USDJPY)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    parser.add_argument("--list", action="store_true", help="List available symbols")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all signals including rejections")
    parser.add_argument("--no-ema", action="store_true", help="Disable 4H EMA filter")
    parser.add_argument("--no-killzone", action="store_true", help="Disable killzone filter")
    parser.add_argument("--no-london-block", action="store_true", help="Disable London open block")
    parser.add_argument("--no-blacklist", action="store_true", help="Disable crypto blacklist")
    parser.add_argument("--show-rejected", action="store_true", help="Show rejected trades in output")
    parser.add_argument("--min-quality", type=int, default=50, help="Minimum quality score (default: 50)")
    parser.add_argument(
        "--fresh-window",
        type=int,
        default=15,
        help="Max age (minutes) of OB tap vs bar close — matches live (default: 15)",
    )
    parser.add_argument(
        "--bias-ttl-hours",
        type=float,
        default=16.0,
        help="Hours until a 4H bias track expires (default: 16, same as auto_trader)",
    )
    parser.add_argument(
        "--fill-model",
        choices=("tap_bar_close", "strategy"),
        default="tap_bar_close",
        help=(
            "tap_bar_close: fill at tap candle close/time (OHLC-consistent, default). "
            "strategy: OB-edge heuristic entry + poll clock."
        ),
    )
    parser.add_argument(
        "--no-intrabar-htf",
        action="store_true",
        help=(
            "Disable forming-4H OHLC rebuild for Phase 1 bias only (strict closed 4H like "
            "auto_trader.py). Phase 2 MSS always uses closed 1H only — unchanged."
        ),
    )
    parser.add_argument(
        "--server-utc-offset",
        type=int,
        default=None,
        help=(
            "MT5 server UTC offset in hours (e.g. 3 for Exness summer / EEST, 2 for "
            "EET winter). When set, 4H bars are rebuilt from 5m data to match the "
            "live terminal's candle grid. Omit to use the pre-exported 4H CSV as-is."
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON path")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir) if args.data_dir else None

    if args.list:
        symbols = list_available_symbols(data_dir)
        if symbols:
            print(f"\nAvailable symbols ({len(symbols)}):")
            for s in symbols:
                print(f"  • {s}")
        else:
            print("\nNo data found. Place CSV files in backtest/data/<SYMBOL>/")
        return 0

    if not args.symbol:
        parser.error("--symbol is required (or use --list to see available symbols)")

    if not args.start or not args.end:
        parser.error("Both --start and --end are required")

    symbol = args.symbol.upper()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    market = detect_market(symbol)

    print(f"\n{'═' * 70}")
    print(f"  Strategy 09 Backtester — {symbol} ({market})")
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Filters: EMA={'ON' if not args.no_ema else 'OFF'}, "
          f"KZ={'ON' if not args.no_killzone else 'OFF'}, "
          f"London={'ON' if not args.no_london_block else 'OFF'}, "
          f"Blacklist={'ON' if not args.no_blacklist else 'OFF'}")
    if args.server_utc_offset is not None:
        print(f"  Server UTC offset: +{args.server_utc_offset} (4H bars resampled from 5m)")
    print(f"{'═' * 70}\n")

    # ── Load data ─────────────────────────────────────────────
    print("  Loading data...")
    data = load_all_timeframes(symbol, start_dt, end_dt, data_dir,
                               server_utc_offset=args.server_utc_offset)

    missing = [tf for tf, df in data.items() if df is None]
    if missing:
        print(f"\n  ❌ Missing data for timeframes: {', '.join(missing)}")
        print(f"     Place CSV files in backtest/data/{symbol}/")
        return 1

    for tf, df in data.items():
        print(f"    {tf:>3s}: {len(df):,} bars ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")

    # ── Run backtest ──────────────────────────────────────────
    print(f"\n  Running backtest...")
    result = run_backtest(
        symbol=symbol,
        df_4h=data["4h"],
        df_1h=data["1h"],
        df_15m=data["15m"],
        df_5m=data["5m"],
        start=start_dt,
        end=end_dt,
        market=market,
        min_quality=args.min_quality,
        fresh_window_minutes=args.fresh_window,
        bias_ttl_hours=args.bias_ttl_hours,
        fill_model=args.fill_model,
        intrabar_htf_bias_mss=not args.no_intrabar_htf,
        enable_ema_filter=not args.no_ema,
        enable_killzone=not args.no_killzone,
        enable_london_block=not args.no_london_block,
        enable_blacklist=not args.no_blacklist,
        verbose=args.verbose,
    )

    # ── Print trades ──────────────────────────────────────────
    pfmt = ".5f" if market == "FOREX" else ".6f"
    printed = 0

    if args.show_rejected or args.verbose:
        # Show all trades including rejections
        trades_to_show = result.trades
    else:
        # Only executed trades
        trades_to_show = result.executed_trades

    if trades_to_show:
        print(f"\n  ── Trades ──")
        for i, t in enumerate(trades_to_show, 1):
            _print_trade(i, t, pfmt)
            printed += 1
    else:
        print(f"\n  No trades found in this period.")

    # ── Summary ───────────────────────────────────────────────
    _print_summary(result)

    # ── Save results ──────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = RESULTS_DIR / f"{symbol}_{args.start}_{args.end}.json"

    _save_results(result, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
