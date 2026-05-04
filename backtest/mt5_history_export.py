#!/usr/bin/env python3
"""
Export MT5 historical bars to per-timeframe CSV files with incremental updates.

CSV columns include UTC bar time plus `Day_IST` / `Time_IST` (calendar date and clock time in IST, UTC+5:30).

Default output (DEFAULT_MT5_HISTORY_OUT_DIR, under Exness structured data on O:):
- .../structured/history/<SYMBOL>/<TF>/<SYMBOL>_<TF>_<OLD>_<NEW>.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Ensure project root is importable when running as:
#   python mt5_history_export.py
ROOT = Path(__file__).resolve().parent

# # Canonical on-disk location for exported MT5 history CSVs (symbol/timeframe subdirs).
# DEFAULT_MT5_HISTORY_OUT_DIR = Path(r"O:\D temp\UltimateTradeBot\Data\Exness\structured\history")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_trader.offline_market import DEFAULT_EXNESS_STRUCTURED_HISTORY_ROOT

# Same root as offline training (`ML_OFFLINE_STRUCTURED_ROOT` default / resolution).
DEFAULT_MT5_HISTORY_OUT_DIR = DEFAULT_EXNESS_STRUCTURED_HISTORY_ROOT

import mt5_history_probe as probe
from mt5_client import MT5Client, MT5Credentials
from terminal_style import format_status_line

# India Standard Time (no DST); fixed offset avoids extra tz database deps on Windows.
_IST = timezone(timedelta(hours=5, minutes=30))

CSV_HEADER = [
    "time_utc",
    "Day_IST",
    "Time_IST",
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]


def _ist_day_and_time_strings_from_unix_ts(ts: int) -> Tuple[str, str]:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(_IST)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def _to_int(raw: object, *, default: Optional[int] = None) -> Optional[int]:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return int(raw)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return default


def _to_float(raw: object, *, default: float = 0.0) -> float:
    if raw is None:
        return float(default)
    if isinstance(raw, bool):
        return float(raw)
    try:
        value = float(str(raw).strip())
    except Exception:
        return float(default)
    return value if value == value else float(default)


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _safe_date(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "unknown"


def _timeframe_to_seconds(tf: str) -> Optional[int]:
    key = (tf or "").strip().lower()
    if not key:
        return None
    if key.endswith("mo"):
        try:
            value = int(key[:-2])
        except ValueError:
            return None
        return max(1, value) * 30 * 24 * 60 * 60
    if len(key) < 2:
        return None
    unit = key[-1]
    try:
        value = int(key[:-1])
    except ValueError:
        return None
    if value <= 0:
        return None
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 60 * 60
    if unit == "d":
        return value * 24 * 60 * 60
    if unit == "w":
        return value * 7 * 24 * 60 * 60
    return None


def _parse_timestamp(raw: object) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return int(text)
        except Exception:
            return None
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _bar_time_utc(bar) -> Optional[datetime]:
    try:
        t_sec = int(bar["time"])
    except Exception:
        try:
            t_sec = int(getattr(bar, "time"))
        except Exception:
            return None
    return datetime.fromtimestamp(t_sec, tz=timezone.utc)


def _extract_bar_field(bar, key: str, default: object = None) -> object:
    if bar is None:
        return default
    if isinstance(bar, dict):
        if key in bar:
            return bar[key]
    else:
        if hasattr(bar, "__getitem__"):
            try:
                return bar[key]
            except Exception:
                try:
                    # Structured arrays often expose field names in dtype.names.
                    return getattr(bar, key)
                except Exception:
                    pass
        try:
            return getattr(bar, key)
        except Exception:
            pass
    return default


def _fetch_bar(mt5, symbol: str, tf_value: int, pos: int) -> Tuple[Optional[object], Optional[str]]:
    try:
        rates = mt5.copy_rates_from_pos(symbol, tf_value, int(pos), 1)
    except Exception as exc:
        return None, f"exception={exc}"
    if rates is None:
        last_error = getattr(mt5, "last_error", lambda: None)()
        return None, f"last_error={last_error}"
    if len(rates) == 0:
        return None, None
    return rates[0], None


def _is_retryable_invalid_params_error(last_error: object) -> bool:
    if last_error is None:
        return False
    try:
        if isinstance(last_error, tuple) and len(last_error) > 0 and int(last_error[0]) == -2:
            return True
    except Exception:
        pass
    msg = str(last_error).lower()
    return ("invalid param" in msg) or ("invalid params" in msg)


def _copy_rates_with_retries(
    mt5,
    symbol: str,
    tf_value: int,
    pos: int,
    count: int,
    *,
    min_count: int = 1,
) -> Tuple[Optional[object], Optional[str]]:
    count = max(int(min_count), int(count))
    initial_count = count
    while count >= min_count:
        try:
            rates = mt5.copy_rates_from_pos(symbol, tf_value, int(pos), int(count))
        except Exception as exc:
            if _is_retryable_invalid_params_error(exc):
                new_count = max(min_count, count // 2)
                if new_count == count:
                    return None, f"exception={exc}"
                count = new_count
                continue
            return None, f"exception={exc}"
        if rates is not None:
            return rates, None
        last_error = getattr(mt5, "last_error", lambda: None)()
        if _is_retryable_invalid_params_error(last_error):
            count = count // 2
            if count < min_count:
                count = min_count
            if count <= min_count:
                return None, f"last_error={last_error}"
            continue
        return None, f"last_error={last_error}"
    return None, f"copy_rates_from_pos failed after retries (initial_count={initial_count}, min_count={min_count})"


def _bar_to_row(bar) -> Optional[Dict[str, object]]:
    t_sec = _to_int(_extract_bar_field(bar, "time"), default=None)
    if t_sec is None:
        return None
    day_ist, time_ist = _ist_day_and_time_strings_from_unix_ts(int(t_sec))
    return {
        "time": int(t_sec),
        "time_utc": datetime.fromtimestamp(int(t_sec), tz=timezone.utc).isoformat(),
        "Day_IST": day_ist,
        "Time_IST": time_ist,
        "open": _to_float(_extract_bar_field(bar, "open"), default=0.0),
        "high": _to_float(_extract_bar_field(bar, "high"), default=0.0),
        "low": _to_float(_extract_bar_field(bar, "low"), default=0.0),
        "close": _to_float(_extract_bar_field(bar, "close"), default=0.0),
        "tick_volume": int(_to_int(_extract_bar_field(bar, "tick_volume"), default=0) or 0),
        "spread": int(_to_int(_extract_bar_field(bar, "spread"), default=0) or 0),
        "real_volume": int(_to_int(_extract_bar_field(bar, "real_volume"), default=0) or 0),
    }


def _row_from_csv(fields: Dict[str, object], *, source: Path) -> Optional[Dict[str, object]]:
    _ = source
    ts = _parse_timestamp(fields.get("time_utc") or fields.get("time"))
    if ts is None:
        return None
    day_ist, time_ist = _ist_day_and_time_strings_from_unix_ts(int(ts))
    return {
        "time": int(ts),
        "time_utc": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
        "Day_IST": day_ist,
        "Time_IST": time_ist,
        "open": _to_float(fields.get("open") or fields.get("Open")),
        "high": _to_float(fields.get("high") or fields.get("High")),
        "low": _to_float(fields.get("low") or fields.get("Low")),
        "close": _to_float(fields.get("close") or fields.get("Close")),
        "tick_volume": int(_to_int(fields.get("tick_volume") or fields.get("volume"), default=0) or 0),
        "spread": int(_to_int(fields.get("spread"), default=0) or 0),
        "real_volume": int(_to_int(fields.get("real_volume"), default=0) or 0),
    }


def _load_existing_rows(tf_dir: Path, *, symbol: str, tf_key: str) -> List[Dict[str, object]]:
    rows_by_time: Dict[int, Dict[str, object]] = {}
    symbol_key = symbol.upper()
    for path in sorted(tf_dir.glob(f"{symbol_key}_{tf_key}_*.csv")):
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                for raw in reader:
                    if not isinstance(raw, dict):
                        continue
                    normalized = {str(k): ("" if v is None else str(v)) for k, v in raw.items()}
                    row = _row_from_csv(normalized, source=path)
                    if row is None:
                        continue
                    rows_by_time[int(row["time"])] = row
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return [rows_by_time[k] for k in sorted(rows_by_time)]


def _estimate_missing_bars(existing: List[Dict[str, object]], step_seconds: Optional[int]) -> Tuple[int, int]:
    if len(existing) < 2 or not step_seconds or step_seconds <= 0:
        return 0, 0
    times = sorted({int(r["time"]) for r in existing if isinstance(r.get("time"), int)})
    if len(times) < 2:
        return 0, 0
    missing_total = 0
    largest_gap = 0
    prev = times[0]
    for current in times[1:]:
        gap = int(current) - int(prev)
        if gap > step_seconds:
            missing = max(0, (gap // int(step_seconds)) - 1)
            missing_total += missing
            largest_gap = max(largest_gap, missing)
        prev = current
    return missing_total, largest_gap


def _filter_native_resolution(rows: List[Dict[str, object]], tf_seconds: int) -> List[Dict[str, object]]:
    if not rows or tf_seconds <= 0 or tf_seconds >= 86400:
        return rows
        
    streak = 0
    last_native_idx = len(rows) - 1
    saw_native = False
    
    for i in range(len(rows) - 1, 0, -1):
        gap = int(rows[i]["time"]) - int(rows[i-1]["time"])
        if gap == tf_seconds:
            streak = 0
            last_native_idx = i - 1
            saw_native = True
        else:
            streak += 1
            if streak > 200:
                break
                
    if not saw_native:
        return []
        
    return rows[last_native_idx:]


def _filter_by_date_range(rows: List[Dict[str, object]], start_date: Optional[str], end_date: Optional[str]) -> List[Dict[str, object]]:
    """Filter rows to only include those within the specified date range."""
    if not start_date and not end_date:
        return rows
    
    start_ts = None
    end_ts = None
    
    if start_date:
        try:
            dt = datetime.fromisoformat(start_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            start_ts = int(dt.timestamp())
        except Exception:
            print(format_status_line(f"[!] invalid start-date format: {start_date}", status="warning"))
            return rows
    
    if end_date:
        try:
            dt = datetime.fromisoformat(end_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            end_ts = int(dt.timestamp())
        except Exception:
            print(format_status_line(f"[!] invalid end-date format: {end_date}", status="warning"))
            return rows
    
    filtered = []
    for row in rows:
        try:
            ts = int(row.get("time", 0))
            if start_ts and ts < start_ts:
                continue
            if end_ts and ts > end_ts:
                continue
            filtered.append(row)
        except Exception:
            continue
    
    return filtered


def _fetch_rows_from_pos_range(mt5, symbol: str, tf_value: int, max_pos: int, chunk_size: int) -> List[Dict[str, object]]:
    rows_by_time: Dict[int, Dict[str, object]] = {}
    pos_end = int(max_pos)
    chunk_size = max(1, int(chunk_size))
    while pos_end >= 0:
        pos_start = max(0, pos_end - chunk_size + 1)
        count = int(pos_end - pos_start + 1)
        rates, fetch_error = _copy_rates_with_retries(mt5, symbol, tf_value, int(pos_start), count)
        if fetch_error:
            last_error = getattr(mt5, "last_error", lambda: None)()
            raise RuntimeError(
                f"copy_rates_from_pos failed symbol={symbol} tf={tf_value} pos_start={pos_start} count={count} last_error={last_error}"
            )
        if len(rates) == 0:
            break
        fetched = int(len(rates))
        for r in rates:
            row = _bar_to_row(r)
            if row is not None:
                rows_by_time[int(row["time"])] = row
        pos_end = pos_start - fetched
    return [rows_by_time[k] for k in sorted(rows_by_time)]


def _fetch_newer_rows(
    mt5,
    *,
    symbol: str,
    tf_value: int,
    max_pos: int,
    cutoff_time: int,
    chunk_size: int,
) -> List[Dict[str, object]]:
    rows_by_time: Dict[int, Dict[str, object]] = {}
    pos = 0
    chunk_size = max(1, int(chunk_size))
    while pos <= max_pos:
        rates, fetch_error = _copy_rates_with_retries(mt5, symbol, tf_value, int(pos), chunk_size)
        if fetch_error:
            last_error = getattr(mt5, "last_error", lambda: None)()
            raise RuntimeError(
                f"copy_rates_from_pos failed symbol={symbol} tf={tf_value} pos={pos} last_error={last_error}"
            )
        if len(rates) == 0:
            break

        seen_times: List[int] = []
        for r in rates:
            row = _bar_to_row(r)
            if row is None:
                continue
            ts = int(row["time"])
            seen_times.append(ts)
            if ts > cutoff_time:
                rows_by_time[ts] = row
        if not seen_times:
            break
        if max(seen_times) <= cutoff_time:
            break
        pos += int(len(rates))
    return [rows_by_time[k] for k in sorted(rows_by_time)]


def _write_history_rows(
    *,
    rows: Iterable[Dict[str, object]],
    symbol: str,
    tf_key: str,
    out_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[int, Path]:
    merged = {int(row["time"]): row for row in rows if isinstance(row, dict) and isinstance(row.get("time"), int)}
    if not merged:
        return 0, out_dir / symbol / tf_key / f"{symbol}_{tf_key}_empty.csv"

    sorted_times = sorted(merged)
    sorted_rows = [merged[ts] for ts in sorted_times]
    
    tf_seconds = _timeframe_to_seconds(tf_key) or 0
    filtered_rows = _filter_native_resolution(sorted_rows, tf_seconds)
    
    # Apply date range filtering
    filtered_rows = _filter_by_date_range(filtered_rows, start_date, end_date)
    
    if not filtered_rows:
        return 0, out_dir / symbol / tf_key / f"{symbol}_{tf_key}_empty.csv"

    first_ts = int(filtered_rows[0]["time"])
    last_ts = int(filtered_rows[-1]["time"])

    symbol_dir = out_dir / symbol / tf_key
    symbol_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{symbol}_{tf_key}_{_safe_date(datetime.fromtimestamp(first_ts, tz=timezone.utc))}_{_safe_date(datetime.fromtimestamp(last_ts, tz=timezone.utc))}.csv"
    final_path = symbol_dir / out_name
    tmp_path = final_path.with_suffix(".partial")

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in filtered_rows:
            writer.writerow(
                [
                    row.get("time_utc"),
                    row.get("Day_IST"),
                    row.get("Time_IST"),
                    int(row.get("time")),
                    float(row.get("open", 0.0)),
                    float(row.get("high", 0.0)),
                    float(row.get("low", 0.0)),
                    float(row.get("close", 0.0)),
                    int(row.get("tick_volume", 0)),
                    int(row.get("spread", 0)),
                    int(row.get("real_volume", 0)),
                ]
            )
    if final_path.exists():
        final_path.unlink()
    tmp_path.replace(final_path)

    # Keep filesystem tidy and deterministic: keep only one canonical file per timeframe.
    for stale in sorted(symbol_dir.glob(f"{symbol}_{tf_key}_*.csv")):
        if stale != final_path:
            try:
                stale.unlink()
            except OSError:
                continue

    return len(filtered_rows), final_path


def _write_full_history_csv(
    mt5,
    *,
    symbol: str,
    tf_key: str,
    tf_value: int,
    max_pos: int,
    out_dir: Path,
    chunk_size: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[int, Path]:
    rows = _fetch_rows_from_pos_range(mt5, symbol=symbol, tf_value=tf_value, max_pos=max_pos, chunk_size=chunk_size)
    return _write_history_rows(rows=rows, symbol=symbol, tf_key=tf_key, out_dir=out_dir, start_date=start_date, end_date=end_date)


def _write_history_csv(
    *,
    mt5,
    symbol: str,
    tf_key: str,
    tf_value: int,
    max_pos: int,
    out_dir: Path,
    chunk_size: int,
    skip_existing: bool,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    tf_seconds = _timeframe_to_seconds(tf_key) or 0
    local_rows = _load_existing_rows(out_dir / symbol / tf_key, symbol=symbol, tf_key=tf_key)

    original_len = len(local_rows)
    local_rows = _filter_native_resolution(local_rows, tf_seconds)
    if len(local_rows) < original_len:
        print(format_status_line(f"[*] dropped {original_len - len(local_rows):,} low-res padding bars from local {symbol} {tf_key}", status="info"))

    if skip_existing and local_rows:
        first_ts = int(local_rows[0]["time"])
        last_ts = int(local_rows[-1]["time"])
        print(
            format_status_line(
                f"[*] skip existing {symbol} {tf_key} rows={len(local_rows):,} range={_safe_date(datetime.fromtimestamp(first_ts, tz=timezone.utc))} -> {_safe_date(datetime.fromtimestamp(last_ts, tz=timezone.utc))}",
                status="warning",
            )
        )
        return

    local_missing, local_largest_gap = _estimate_missing_bars(local_rows, tf_seconds)
    if local_rows:
        local_oldest = datetime.fromtimestamp(int(local_rows[0]["time"]), tz=timezone.utc)
        local_newest = datetime.fromtimestamp(int(local_rows[-1]["time"]), tz=timezone.utc)
        print(
            format_status_line(
                f"[*] existing {symbol} {tf_key} rows={len(local_rows):,} range={local_oldest.isoformat()} -> {local_newest.isoformat()} "
                f"est_internal_missing={local_missing:,} largest_gap={local_largest_gap:,}",
                status="info",
            )
        )

    newest_bar, err_newest = _fetch_bar(mt5, symbol, tf_value, 0)
    if newest_bar is None:
        print(format_status_line(f"[X] unable to fetch newest bar for {symbol} {tf_key} ({err_newest})", status="error"))
        return

    oldest_bar, err_oldest = _fetch_bar(mt5, symbol, tf_value, max_pos)
    if oldest_bar is None:
        print(format_status_line(f"[X] unable to fetch oldest bar for {symbol} {tf_key} ({err_oldest})", status="error"))
        return

    newest_time = _bar_time_utc(newest_bar)
    oldest_time = _bar_time_utc(oldest_bar)
    if newest_time is None or oldest_time is None:
        print(format_status_line(f"[X] invalid broker times for {symbol} {tf_key}", status="error"))
        return

    broker_newest_ts = int(newest_time.timestamp())
    broker_oldest_ts = int(oldest_time.timestamp())

    if not local_rows:
        print(format_status_line(f"[*] no local rows for {symbol} {tf_key}, downloading full history", status="info"))
        total_written, out_path = _write_full_history_csv(
            mt5,
            symbol=symbol,
            tf_key=tf_key,
            tf_value=tf_value,
            max_pos=max_pos,
            out_dir=out_dir,
            chunk_size=chunk_size,
            start_date=start_date,
            end_date=end_date,
        )
        if total_written:
            print(format_status_line(f"[+] done {symbol} {tf_key} rows={total_written:,} -> {out_path}", status="success"))
        return

    local_newest_ts = int(local_rows[-1]["time"])
    if broker_newest_ts <= local_newest_ts:
        print(
            format_status_line(
                f"[*] up-to-date {symbol} {tf_key}: local latest {_safe_date(datetime.fromtimestamp(local_newest_ts, tz=timezone.utc))} "
                f">= broker latest {_safe_date(datetime.fromtimestamp(broker_newest_ts, tz=timezone.utc))}",
                status="success",
            )
        )
        return

    if tf_seconds > 0:
        gap_count = max(0, (broker_newest_ts - local_newest_ts) // tf_seconds)
    else:
        gap_count = 0
    print(
        format_status_line(
            f"[*] missing tail {symbol} {tf_key}: broker_latest={newest_time.isoformat()} local_latest={_safe_date(datetime.fromtimestamp(local_newest_ts, tz=timezone.utc))} "
            f"est_missing={gap_count:,}",
            status="info",
        )
    )

    newer_rows = _fetch_newer_rows(
        mt5,
        symbol=symbol,
        tf_value=tf_value,
        max_pos=max_pos,
        cutoff_time=local_newest_ts,
        chunk_size=chunk_size,
    )
    if not newer_rows:
        print(format_status_line(f"[*] no new bars fetched for {symbol} {tf_key}", status="success"))
        return

    total_written, out_path = _write_history_rows(
        rows=[*local_rows, *newer_rows],
        symbol=symbol,
        tf_key=tf_key,
        out_dir=out_dir,
        start_date=start_date,
        end_date=end_date,
    )

    print(
        format_status_line(
            f"[+] updated {symbol} {tf_key} rows={total_written:,} added={len(newer_rows):,} -> {out_path}",
            status="success",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export full MT5 historical bars to CSV per timeframe.")
    parser.add_argument(
        "--symbols",
        default="BTCUSD,ETHUSD,XAUUSD,EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,NZDUSD,USDCHF",
        help="Comma-separated symbols to export (defaults include majors + crypto/metal).",
    )
    parser.add_argument("--timeframes", default="all", help="Comma-separated timeframes or 'all'.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MT5_HISTORY_OUT_DIR),
        help="Output directory for CSV files (default: Exness structured history root on O: drive).",
    )
    parser.add_argument("--chunk-size", type=int, default=100000, help="Bars per MT5 request.")
    parser.add_argument("--max-doublings", type=int, default=30, help="Doubling steps for upper-bound search.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip CSVs that already exist.")
    parser.add_argument("--start-date", default=None, help="Start date in YYYY-MM-DD format (e.g., 2021-04-01).")
    parser.add_argument("--end-date", default=None, help="End date in YYYY-MM-DD format (e.g., 2026-04-01).")
    args = parser.parse_args()

    symbols = _parse_csv(args.symbols)
    timeframes = probe._resolve_timeframes(args.timeframes)
    if not symbols:
        print(format_status_line("[X] No symbols provided.", status="error"))
        return 2
    if not timeframes:
        print(format_status_line("[X] No valid timeframes provided.", status="error"))
        return 2

    creds = MT5Credentials.from_env()
    client = MT5Client(creds)
    if not client.initialize():
        print(format_status_line("[X] MT5 login failed", status="error"))
        return 1

    try:
        mt5 = client._mt5
        out_dir = Path(args.output_dir)

        for sym in symbols:
            resolved = client.ensure_symbol(sym)
            if not resolved:
                print(format_status_line(f"[X] symbol not available in MT5: {sym}", status="error"))
                continue

            for tf in timeframes:
                tf_attr = probe.TIMEFRAME_MAP.get(tf)
                if not tf_attr:
                    print(format_status_line(f"[!] unsupported timeframe: {tf}", status="warning"))
                    continue
                tf_value = getattr(mt5, tf_attr, None)
                if tf_value is None:
                    print(format_status_line(f"[!] missing MT5 constant: {tf_attr}", status="warning"))
                    continue

                max_pos, err = probe._find_max_pos(mt5, resolved, tf_value, max_doublings=int(args.max_doublings))
                if max_pos is None:
                    print(format_status_line(f"[X] no data {resolved} {tf} ({err})", status="error"))
                    continue

                _write_history_csv(
                    mt5=mt5,
                    symbol=resolved,
                    tf_key=tf,
                    tf_value=tf_value,
                    max_pos=max_pos,
                    out_dir=out_dir,
                    chunk_size=int(args.chunk_size),
                    skip_existing=bool(args.skip_existing),
                    start_date=args.start_date,
                    end_date=args.end_date,
                )

        return 0
    finally:
        client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
