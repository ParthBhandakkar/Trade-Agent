from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pytz
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
STRATEGY_DIR = REPO_ROOT / "strategy_09_mss_ob_entry"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STRATEGY_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.utils.env_loader import load_root_env              # noqa: E402

load_root_env(REPO_ROOT)
load_dotenv(THIS_DIR / ".env", override=True)
os.environ["TV_SOURCE_TZ"] = os.getenv("TV_SOURCE_TZ", "Asia/Kolkata").strip() or "Asia/Kolkata"

from crypto_backtester import CRYPTO_PAIRS, CryptoDataFetcher  # noqa: E402
from entry_filters import FilterResult, calculate_ema, check_killzone  # noqa: E402
from forex_backtester import FOREX_PAIRS, ForexDataFetcher  # noqa: E402
from phase_logger import PhaseLogger  # noqa: E402
from strategy import (  # noqa: E402
    BiasType,
    DailyBias,
    MSSConfirmation,
    MSSOB_Signal,
    MSSOrderBlockStrategy,
    OBEntrySetup,
    format_signal_for_jsonl,
)
from scripts.utils.indicators import Direction, OrderBlock  # noqa: E402
from strategy_09_sheet_server.google_sheets_logger import (  # noqa: E402
    DEFAULT_TRADE_HEADERS,
    GoogleSheetsTradeLogger,
)

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.UTC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FRESH_WINDOW_MINUTES = 15
CRYPTO_SL_BUFFER_PCT = 0.35
FOREX_SL_BUFFER_PIPS = 15
CRYPTO_SYMBOL_BLACKLIST = {
    "AVAXUSDT", "LTCUSDT", "BTCUSDT", "ETCUSDT",
    "XLMUSDT", "NEARUSDT", "XRPUSDT", "ARBUSDT", "LINKUSDT",
}
_MIN_4H_BARS_FOR_EMA = 50

# London Open manipulation block (07:00-08:00 UTC)
# Real-data optimization across 90 combos: 07:00-08:00 is the sweet spot.
# Banks sweep liquidity in the first hour, faking out OB entries.
LONDON_BLOCK_START_UTC = 7 * 60   # 07:00 UTC in minutes
LONDON_BLOCK_END_UTC   = 8 * 60   # 08:00 UTC in minutes


@dataclass
class BiasTrack:
    key: str
    bias: DailyBias
    market: str
    created_utc: datetime
    expires_utc: datetime
    mss: Optional[MSSConfirmation] = None
    ob_entry: Optional[OBEntrySetup] = None
    alerted: bool = False
    completed: bool = False
    last_rejected_tap: Optional[datetime] = None


@dataclass
class ServiceConfig:
    host: str
    port: int
    poll_seconds: int
    min_quality: int
    fresh_window: int
    bias_ttl_hours: float
    bias_lookback_hours: int
    crypto_pairs: List[str]
    forex_pairs: List[str]
    crypto_only: bool
    forex_only: bool
    skip_session_filter: bool
    no_killzone: bool
    allow_asian: bool
    no_ema_filter: bool
    no_blacklist: bool
    log_dir: Path
    state_file: Path
    google_sheet_id: str
    google_worksheet_title: str
    google_service_account_file: Optional[str]
    google_service_account_json: Optional[str]
    no_london_block: bool


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name, "").strip()
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config() -> ServiceConfig:
    host = os.getenv("SHEET_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SHEET_SERVER_PORT", "8010"))
    poll_seconds = int(os.getenv("SHEET_POLL_SECONDS", "60"))
    min_quality = int(os.getenv("SHEET_MIN_QUALITY", "50"))
    fresh_window = int(os.getenv("SHEET_FRESH_WINDOW_MINUTES", str(FRESH_WINDOW_MINUTES)))
    bias_ttl_hours = float(os.getenv("SHEET_BIAS_TTL_HOURS", "16"))
    bias_lookback_hours = int(os.getenv("SHEET_BIAS_LOOKBACK_HOURS", "72"))
    crypto_only = _env_bool("SHEET_CRYPTO_ONLY", False)
    forex_only = _env_bool("SHEET_FOREX_ONLY", False)

    if crypto_only and forex_only:
        raise RuntimeError("SHEET_CRYPTO_ONLY and SHEET_FOREX_ONLY cannot both be true.")

    default_log_dir = THIS_DIR / "logs"
    default_state_file = THIS_DIR / "state" / "sent_signals.json"

    google_sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    google_worksheet_title = os.getenv("GOOGLE_WORKSHEET_TITLE", "trade_signals").strip()
    google_service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip() or None
    google_service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip() or None

    if not google_sheet_id:
        raise RuntimeError("Missing GOOGLE_SHEET_ID for Strategy 09 sheet server.")

    if not google_service_account_file and not google_service_account_json:
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_FILE or "
            "GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    return ServiceConfig(
        host=host,
        port=port,
        poll_seconds=poll_seconds,
        min_quality=min_quality,
        fresh_window=fresh_window,
        bias_ttl_hours=bias_ttl_hours,
        bias_lookback_hours=bias_lookback_hours,
        crypto_pairs=_env_list("SHEET_CRYPTO_PAIRS", CRYPTO_PAIRS),
        forex_pairs=_env_list("SHEET_FOREX_PAIRS", FOREX_PAIRS),
        crypto_only=crypto_only,
        forex_only=forex_only,
        skip_session_filter=_env_bool("SHEET_SKIP_SESSION_FILTER", False),
        no_killzone=_env_bool("SHEET_NO_KILLZONE", False),
        allow_asian=_env_bool("SHEET_ALLOW_ASIAN", False),
        no_ema_filter=_env_bool("SHEET_NO_EMA_FILTER", False),
        no_blacklist=_env_bool("SHEET_NO_BLACKLIST", False),
        log_dir=Path(os.getenv("SHEET_LOG_DIR", str(default_log_dir))).resolve(),
        state_file=Path(os.getenv("SHEET_STATE_FILE", str(default_state_file))).resolve(),
        google_sheet_id=google_sheet_id,
        google_worksheet_title=google_worksheet_title,
        google_service_account_file=google_service_account_file,
        google_service_account_json=google_service_account_json,
        no_london_block=_env_bool("SHEET_NO_LONDON_BLOCK", False),
    )


def _is_forex_session(now_ist: datetime) -> bool:
    weekday = now_ist.weekday()
    total_minutes = now_ist.hour * 60 + now_ist.minute
    if weekday == 6:
        return False
    if weekday == 5:
        return total_minutes < 90
    if weekday == 0:
        return total_minutes >= 90
    return True


def _is_london_open_block(now_utc: datetime) -> bool:
    """Block entries during London Open first hour (07:00-08:00 UTC).

    Real-data optimization across 90 filter combos: 07:00-08:00 is the
    sweet spot.  Banks sweep liquidity in the first hour, faking out
    OB entries.  After 08:00, the real directional move starts.
    """
    t = now_utc.hour * 60 + now_utc.minute
    return LONDON_BLOCK_START_UTC <= t <= LONDON_BLOCK_END_UTC


def _bias_key(bias: DailyBias) -> str:
    timestamp = bias.sweep_timestamp
    return f"{bias.direction.value}_{timestamp.isoformat() if timestamp is not None else 'none'}"


def _floor_5m(dt: datetime) -> datetime:
    dt0 = dt.replace(second=0, microsecond=0)
    return dt0.replace(minute=dt0.minute - (dt0.minute % 5))


def _due_5m_cycle(
    now_ist: datetime,
    poll_seconds: int,
    last_key: Optional[datetime],
) -> Optional[datetime]:
    trigger = _floor_5m(now_ist - timedelta(minutes=1)) + timedelta(minutes=1)
    if last_key is not None and trigger <= last_key:
        return None
    lag = (now_ist - trigger).total_seconds()
    if 0 <= lag <= max(90, poll_seconds + 5):
        return trigger
    return None


def _is_due_4h(cycle_time: datetime) -> bool:
    return cycle_time.minute == 31 and (cycle_time.hour % 4) == 1


def _is_due_1h(cycle_time: datetime) -> bool:
    return cycle_time.minute == 31


def _is_due_15m(cycle_time: datetime) -> bool:
    return cycle_time.minute in (1, 16, 31, 46)


def _drop_incomplete(
    df: pd.DataFrame,
    interval_min: int,
    now_utc: datetime,
    safety_s: int = 60,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    if out.index.tz is None:
        out = out.copy()
        out.index = out.index.tz_localize(UTC)
    cutoff = now_utc - timedelta(seconds=safety_s)
    while len(out) > 0:
        last_close = out.index[-1] + timedelta(minutes=interval_min)
        if last_close <= cutoff:
            break
        out = out.iloc[:-1]
    return out


def _signal_id(sig_json: Dict[str, Any]) -> str:
    return "|".join([
        str(sig_json.get("symbol", "")),
        str(sig_json.get("direction", "")),
        str(sig_json.get("signal_datetime_ist", "")),
    ])


def _is_fresh(tap_time: Any, now_utc: datetime, window_min: int) -> bool:
    if tap_time is None:
        return False
    timestamp = tap_time.to_pydatetime() if hasattr(tap_time, "to_pydatetime") else tap_time
    if timestamp.tzinfo is None:
        timestamp = UTC.localize(timestamp)
    age_minutes = (now_utc - timestamp).total_seconds() / 60.0
    return age_minutes <= window_min


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _is_jpy_pair(symbol: str) -> bool:
    return "JPY" in symbol.upper()


def _check_strict_ema(df_4h: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    if df_4h is None or df_4h.empty:
        return False, "No 4H data available - skipping trade"

    if len(df_4h) < _MIN_4H_BARS_FOR_EMA:
        return False, (
            f"Insufficient 4H history ({len(df_4h)} bars < "
            f"{_MIN_4H_BARS_FOR_EMA} required) - skipping trade"
        )

    close = df_4h["close"]
    ema21 = calculate_ema(close, 21)
    ema50 = calculate_ema(close, 50)

    last_close = float(close.iloc[-1])
    last_ema21 = float(ema21.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    bullish = direction.lower() in {"bullish", "long"}

    if bullish:
        passed = last_close > last_ema21 > last_ema50
        if passed:
            detail = (
                f"ALIGNED: Close({last_close:.5f}) > "
                f"EMA21({last_ema21:.5f}) > EMA50({last_ema50:.5f})"
            )
        elif last_ema21 <= last_ema50:
            detail = (
                f"COUNTER-TREND: Bullish but EMA21({last_ema21:.5f}) <= "
                f"EMA50({last_ema50:.5f})"
            )
        else:
            detail = (
                f"WEAK ALIGNMENT: EMA21({last_ema21:.5f}) > EMA50({last_ema50:.5f}) "
                f"but Close({last_close:.5f}) < EMA21"
            )
        return passed, detail

    passed = last_close < last_ema21 < last_ema50
    if passed:
        detail = (
            f"ALIGNED: Close({last_close:.5f}) < "
            f"EMA21({last_ema21:.5f}) < EMA50({last_ema50:.5f})"
        )
    elif last_ema21 >= last_ema50:
        detail = (
            f"COUNTER-TREND: Bearish but EMA21({last_ema21:.5f}) >= "
            f"EMA50({last_ema50:.5f})"
        )
    else:
        detail = (
            f"WEAK ALIGNMENT: EMA21({last_ema21:.5f}) < EMA50({last_ema50:.5f}) "
            f"but Close({last_close:.5f}) > EMA21"
        )
    return passed, detail


def compute_smart_sl(
    ob: OrderBlock,
    bias_direction: BiasType,
    df_5m: pd.DataFrame,
    tap_time: pd.Timestamp,
    entry_price: float,
    market: str,
    symbol: str,
) -> Tuple[float, str]:
    del entry_price

    if bias_direction == BiasType.BEARISH:
        base_sl = ob.top
        reason = f"OB high ({base_sl:.6f})"
    else:
        base_sl = ob.bottom
        reason = f"OB low ({base_sl:.6f})"

    ob_time = ob.datetime.to_pydatetime() if hasattr(ob.datetime, "to_pydatetime") else ob.datetime
    if ob_time.tzinfo is None:
        ob_time = UTC.localize(ob_time)

    tap_dt = tap_time.to_pydatetime() if hasattr(tap_time, "to_pydatetime") else tap_time
    if tap_dt.tzinfo is None:
        tap_dt = UTC.localize(tap_dt)

    if df_5m is not None and not df_5m.empty:
        mask = (df_5m.index >= ob_time) & (df_5m.index <= tap_dt)
        segment = df_5m.loc[mask]
        if len(segment) > 0:
            if bias_direction == BiasType.BEARISH:
                swing_high = float(segment["high"].max())
                if swing_high > base_sl:
                    reason = (
                        f"Swing high ({swing_high:.6f}) between OB and tap "
                        f"(was OB high {base_sl:.6f})"
                    )
                    base_sl = swing_high
            else:
                swing_low = float(segment["low"].min())
                if swing_low < base_sl:
                    reason = (
                        f"Swing low ({swing_low:.6f}) between OB and tap "
                        f"(was OB low {base_sl:.6f})"
                    )
                    base_sl = swing_low

    if market == "CRYPTO":
        buffer = base_sl * (CRYPTO_SL_BUFFER_PCT / 100.0)
        reason += f" + {CRYPTO_SL_BUFFER_PCT}% buffer"
    else:
        buffer = FOREX_SL_BUFFER_PIPS * (0.01 if _is_jpy_pair(symbol) else 0.0001)
        reason += f" + {FOREX_SL_BUFFER_PIPS}pip buffer"

    sl = base_sl + buffer if bias_direction == BiasType.BEARISH else base_sl - buffer
    return sl, reason


def compute_tp(entry_price: float, sl_price: float, rr: float = 1.5) -> float:
    risk = abs(entry_price - sl_price)
    if entry_price > sl_price:
        return entry_price + rr * risk
    return entry_price - rr * risk


class Strategy09SheetService:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.hostname = socket.gethostname()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.status_lock = threading.Lock()

        self.log_dir = self.config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.config.state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.trades_log = self.log_dir / "trades.jsonl"
        self.cycle_log = self.log_dir / "cycle_summary.jsonl"

        self.strategy = MSSOrderBlockStrategy()
        self.phase_log = PhaseLogger(self.log_dir / "phase_logs")
        self.sheet_logger = GoogleSheetsTradeLogger(
            spreadsheet_id=self.config.google_sheet_id,
            worksheet_title=self.config.google_worksheet_title,
            service_account_file=self.config.google_service_account_file,
            service_account_json=self.config.google_service_account_json,
            headers=DEFAULT_TRADE_HEADERS,
        )
        self.tv_source_tz = os.environ.get("TV_SOURCE_TZ", "Asia/Kolkata").strip() or "Asia/Kolkata"

        scan_crypto = not self.config.forex_only
        scan_forex = not self.config.crypto_only
        self.crypto_fetcher = CryptoDataFetcher() if scan_crypto else None
        self.forex_fetcher = ForexDataFetcher() if scan_forex else None

        self.pairs: List[Tuple[str, str, Any]] = []
        if scan_crypto and self.crypto_fetcher is not None:
            for symbol in self.config.crypto_pairs:
                self.pairs.append((symbol, "CRYPTO", self.crypto_fetcher))
        if scan_forex and self.forex_fetcher is not None:
            for symbol in self.config.forex_pairs:
                self.pairs.append((symbol, "FOREX", self.forex_fetcher))

        self.active: Dict[str, Dict[str, BiasTrack]] = {}
        self.last_cycle_key: Optional[datetime] = None
        self.first_run = True
        self.sent: Set[str] = self._load_sent_state()

        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        self.status: Dict[str, Any] = {
            "worker_running": False,
            "started_at_ist": now_ist,
            "last_cycle_ist": None,
            "last_cycle_label": None,
            "cycles_completed": 0,
            "signals_logged": 0,
            "last_trade_ist": None,
            "last_trade_id": None,
            "last_error": None,
            "last_error_at_ist": None,
        }

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.sheet_logger.connect()
        self.thread = threading.Thread(
            target=self.run_forever,
            name="strategy09-sheet-service",
            daemon=True,
        )
        self._update_status(worker_running=True)
        self.thread.start()
        logger.info("=" * 62)
        logger.info("  STRATEGY 09 - MSS + OB  |  GOOGLE SHEETS SERVER")
        logger.info("=" * 62)
        logger.info(
            "  Worksheet:      %s",
            self.config.google_worksheet_title,
        )
        logger.info(
            "  Crypto Pairs:   %s",
            0 if self.config.forex_only else len(self.config.crypto_pairs),
        )
        logger.info(
            "  Forex Pairs:    %s",
            0 if self.config.crypto_only else len(self.config.forex_pairs),
        )
        logger.info("  Poll Seconds:   %s", self.config.poll_seconds)
        logger.info("  Fresh Window:   %s min", self.config.fresh_window)
        logger.info("  TV Source TZ:   %s", self.tv_source_tz)
        logger.info("  Min Quality:    %s", self.config.min_quality)
        logger.info(
            "  Killzone:       %s",
            "OFF" if self.config.no_killzone else "London/NY only",
        )
        logger.info(
            "  EMA Filter:     %s",
            "OFF" if self.config.no_ema_filter else "Strict 4H",
        )
        logger.info(
            "  Blacklist:      %s",
            "OFF" if self.config.no_blacklist else f"{len(CRYPTO_SYMBOL_BLACKLIST)} symbols",
        )
        logger.info(
            "  London Block:   %s",
            "OFF" if self.config.no_london_block else "07:00-08:00 UTC blocked",
        )
        logger.info("  Log Dir:        %s", self.log_dir)
        logger.info("=" * 62)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        self._update_status(worker_running=False)

    def snapshot(self) -> Dict[str, Any]:
        with self.status_lock:
            return dict(self.status)

    def config_summary(self) -> Dict[str, Any]:
        return {
            "host": self.config.host,
            "port": self.config.port,
            "worksheet": self.config.google_worksheet_title,
            "sheet_id_suffix": self.config.google_sheet_id[-6:],
            "crypto_pairs": len(self.config.crypto_pairs) if not self.config.forex_only else 0,
            "forex_pairs": len(self.config.forex_pairs) if not self.config.crypto_only else 0,
            "poll_seconds": self.config.poll_seconds,
            "min_quality": self.config.min_quality,
            "fresh_window_minutes": self.config.fresh_window,
            "tv_source_tz": self.tv_source_tz,
            "killzone_filter": not self.config.no_killzone,
            "ema_filter": not self.config.no_ema_filter,
            "blacklist_filter": not self.config.no_blacklist,
            "london_block": not self.config.no_london_block,
            "log_dir": str(self.log_dir),
            "state_file": str(self.state_file),
        }

    def run_forever(self) -> None:
        while not self.stop_event.is_set():
            try:
                cycle_ran = self._maybe_run_cycle()
                if not cycle_ran:
                    self.stop_event.wait(min(5, self.config.poll_seconds))
            except Exception as exc:  # pragma: no cover - resilience path
                logger.error("Sheet service loop error: %s", exc)
                logger.debug(traceback.format_exc())
                self._update_status(
                    last_error=str(exc),
                    last_error_at_ist=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
                )
                self.stop_event.wait(30)

        self._update_status(worker_running=False)

    def _maybe_run_cycle(self) -> bool:
        now_ist = datetime.now(IST)
        cycle_key = _due_5m_cycle(now_ist, self.config.poll_seconds, self.last_cycle_key)
        if cycle_key is None and not self.first_run:
            return False

        if cycle_key is None:
            cycle_key = _floor_5m(now_ist)

        self.last_cycle_key = cycle_key
        loop_start = now_ist
        cycle_str = cycle_key.strftime("%H:%M")

        check_4h = _is_due_4h(cycle_key) or self.first_run
        check_1h = _is_due_1h(cycle_key) or self.first_run
        check_15m = _is_due_15m(cycle_key) or self.first_run

        phases: List[str] = []
        if check_4h:
            phases.append("4H")
        if check_1h:
            phases.append("1H")
        if check_15m:
            phases.append("15M")
        phases.append("5M")
        label = "INIT" if self.first_run else ", ".join(phases)
        logger.info("[%s] === Cycle: %s ===", cycle_str, label)

        refresh = {"5m", "15m"}
        if check_4h:
            refresh.add("4h")
        if check_1h:
            refresh.add("1h")

        forex_open = self.config.skip_session_filter or _is_forex_session(now_ist)
        cycle_logged = 0

        for symbol, market, fetcher in self.pairs:
            if market == "FOREX" and not forex_open:
                continue

            now_utc = datetime.now(UTC)
            symbol_tracks = self.active.setdefault(symbol, {})

            expired = [
                key for key, track in symbol_tracks.items()
                if track.completed or now_utc >= track.expires_utc
            ]
            for key in expired:
                del symbol_tracks[key]

            for interval in refresh:
                fetcher.fetch_ohlcv(
                    symbol,
                    interval,
                    {"4h": 200, "1h": 500, "15m": 1000, "5m": 3000}.get(interval, 800),
                    force_refresh=True,
                )

            df_4h = _drop_incomplete(fetcher.fetch_ohlcv(symbol, "4h", 200), 240, now_utc)
            df_1h = _drop_incomplete(fetcher.fetch_ohlcv(symbol, "1h", 500), 60, now_utc)
            df_15m = _drop_incomplete(fetcher.fetch_ohlcv(symbol, "15m", 1000), 15, now_utc)
            df_5m = _drop_incomplete(fetcher.fetch_ohlcv(symbol, "5m", 3000), 5, now_utc)

            if any(dataframe is None or dataframe.empty for dataframe in [df_4h, df_1h, df_15m, df_5m]):
                continue

            if check_4h:
                biases = self.strategy.determine_bias(df_4h, self.config.bias_lookback_hours)
                self.phase_log.log_bias(
                    symbol=symbol,
                    biases_found=len(biases),
                    bias_details=[
                        {
                            "direction": bias.direction.value,
                            "reason": bias.reason,
                            "sweep_time": str(bias.sweep_timestamp),
                        }
                        for bias in biases
                    ],
                    lookback_hours=self.config.bias_lookback_hours,
                )
                for bias in biases:
                    key = _bias_key(bias)
                    if key in symbol_tracks:
                        continue
                    symbol_tracks[key] = BiasTrack(
                        key=key,
                        bias=bias,
                        market=market,
                        created_utc=now_utc,
                        expires_utc=now_utc + timedelta(hours=self.config.bias_ttl_hours),
                    )

            if not symbol_tracks:
                continue

            if check_1h:
                for track in symbol_tracks.values():
                    if track.mss is not None:
                        continue
                    mss_conf = self.strategy.confirm_mss(df_1h, track.bias)
                    track.mss = mss_conf
                    self.phase_log.log_mss(
                        symbol=symbol,
                        bias_direction=track.bias.direction.value,
                        sweep_time=track.bias.sweep_timestamp,
                        confirmed=mss_conf is not None,
                        mss_break_price=mss_conf.mss.break_price if mss_conf else None,
                        mss_time=mss_conf.timestamp if mss_conf else None,
                        details=mss_conf.details if mss_conf else "",
                    )

            for track in symbol_tracks.values():
                if track.alerted or track.completed or track.mss is None:
                    continue

                df_5m_filtered = df_5m
                if track.last_rejected_tap is not None:
                    cutoff = track.last_rejected_tap
                    if cutoff.tzinfo is None:
                        cutoff = UTC.localize(cutoff)
                    df_5m_filtered = df_5m[df_5m.index > cutoff]
                    if df_5m_filtered.empty:
                        continue

                ob_entry = self.strategy.find_ob_entry(df_15m, df_5m_filtered, track.bias, track.mss)
                if ob_entry is None:
                    continue

                quality = self.strategy.calculate_quality_score(track.bias, track.mss, ob_entry)
                if quality < self.config.min_quality:
                    track.last_rejected_tap = self._normalize_utc(ob_entry.timestamp)
                    continue

                if not _is_fresh(ob_entry.timestamp, now_utc, self.config.fresh_window):
                    track.last_rejected_tap = self._normalize_utc(ob_entry.timestamp)
                    continue

                track.ob_entry = ob_entry
                self.phase_log.log_ob(
                    symbol=symbol,
                    bias_direction=track.bias.direction.value,
                    mss_time=track.mss.timestamp,
                    ob_found=True,
                    ob_time=ob_entry.order_block.datetime,
                    ob_top=ob_entry.order_block.top,
                    ob_bottom=ob_entry.order_block.bottom,
                    fib_level=ob_entry.fib_level,
                    in_ote=ob_entry.in_ote_zone,
                )
                self.phase_log.log_tap(
                    symbol=symbol,
                    bias_direction=track.bias.direction.value,
                    ob_time=ob_entry.order_block.datetime,
                    tapped=True,
                    tap_time=ob_entry.timestamp,
                    entry_price=ob_entry.entry_price,
                )

                signal = self._build_signal(symbol, track, ob_entry, quality)
                sig_json = format_signal_for_jsonl(signal)
                sig_json["market"] = market

                signal_id = _signal_id(sig_json)
                if signal_id in self.sent:
                    track.alerted = True
                    track.completed = True
                    logger.info(
                        "  [%s] %s skipped: already logged",
                        symbol,
                        track.bias.direction.value,
                    )
                    continue

                ema_detail = "EMA filter disabled"
                if not self.config.no_ema_filter:
                    ema_ok, ema_detail = _check_strict_ema(df_4h, track.bias.direction.value)
                    if not ema_ok:
                        track.last_rejected_tap = self._normalize_utc(ob_entry.timestamp)
                        self._log_rejection(
                            now_ist=now_ist,
                            market=market,
                            symbol=symbol,
                            direction=track.bias.direction.value,
                            entry_price=ob_entry.entry_price,
                            quality=quality,
                            signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                            reason=f"EMA: {ema_detail}",
                        )
                        continue

                direction_str = track.bias.direction.value

                if market == "CRYPTO" and not self.config.no_blacklist and symbol in CRYPTO_SYMBOL_BLACKLIST:
                    track.alerted = True
                    track.completed = True
                    self._log_rejection(
                        now_ist=now_ist,
                        market=market,
                        symbol=symbol,
                        direction=direction_str,
                        entry_price=ob_entry.entry_price,
                        quality=quality,
                        signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                        reason="Blacklisted symbol",
                    )
                    continue

                killzone_result = FilterResult(
                    name="Killzone Session",
                    passed=True,
                    reason="Killzone filter disabled",
                )
                if not self.config.no_killzone:
                    killzone_result = check_killzone(
                        now_ist,
                        market=market,
                        allow_asian=self.config.allow_asian,
                    )
                    if not killzone_result.passed:
                        track.alerted = True
                        track.completed = True
                        self._log_rejection(
                            now_ist=now_ist,
                            market=market,
                            symbol=symbol,
                            direction=direction_str,
                            entry_price=ob_entry.entry_price,
                            quality=quality,
                            signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                            reason=f"Killzone: {killzone_result.reason}",
                        )
                        continue

                # ── London Open Block (07:00-08:00 UTC) ──────────
                london_blocked = False
                london_reason = "London block disabled"
                if not self.config.no_london_block:
                    now_utc_check = datetime.now(UTC)
                    if _is_london_open_block(now_utc_check):
                        london_blocked = True
                        london_reason = (
                            f"London Open block (07:00-08:00 UTC) — "
                            f"manipulation phase, banks sweeping liquidity"
                        )
                        track.last_rejected_tap = self._normalize_utc(ob_entry.timestamp)
                        track.alerted = False
                        track.completed = False
                        self._log_rejection(
                            now_ist=now_ist,
                            market=market,
                            symbol=symbol,
                            direction=direction_str,
                            entry_price=ob_entry.entry_price,
                            quality=quality,
                            signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                            reason=f"London Block: {london_reason}",
                        )
                        continue
                    else:
                        london_reason = "Outside London Open block"

                sl_price, sl_reason = compute_smart_sl(
                    ob=ob_entry.order_block,
                    bias_direction=track.bias.direction,
                    df_5m=df_5m,
                    tap_time=ob_entry.timestamp,
                    entry_price=ob_entry.entry_price,
                    market=market,
                    symbol=symbol,
                )
                tp_price = compute_tp(ob_entry.entry_price, sl_price, rr=1.5)

                if track.bias.direction == BiasType.BEARISH and sl_price <= ob_entry.entry_price:
                    track.alerted = True
                    track.completed = True
                    self._log_rejection(
                        now_ist=now_ist,
                        market=market,
                        symbol=symbol,
                        direction=direction_str,
                        entry_price=ob_entry.entry_price,
                        quality=quality,
                        signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                        reason=f"Invalid SL {sl_price} <= entry {ob_entry.entry_price}",
                    )
                    continue

                if track.bias.direction == BiasType.BULLISH and sl_price >= ob_entry.entry_price:
                    track.alerted = True
                    track.completed = True
                    self._log_rejection(
                        now_ist=now_ist,
                        market=market,
                        symbol=symbol,
                        direction=direction_str,
                        entry_price=ob_entry.entry_price,
                        quality=quality,
                        signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                        reason=f"Invalid SL {sl_price} >= entry {ob_entry.entry_price}",
                    )
                    continue

                self.phase_log.log_signal(
                    symbol=symbol,
                    direction=direction_str,
                    signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                    entry_price=ob_entry.entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    quality_score=quality,
                    risk_reward=1.5,
                    extra={
                        "market": market,
                        "sl_reason": sl_reason,
                        "delivery_target": f"GoogleSheets:{self.config.google_worksheet_title}",
                    },
                )

                price_fmt = ".5f" if market == "FOREX" else ".6f"
                logger.info("%s", "=" * 62)
                logger.info(
                    "  SIGNAL: [%s] %s %s",
                    market,
                    symbol,
                    direction_str.upper(),
                )
                logger.info(
                    f"     Entry:  {ob_entry.entry_price:{price_fmt}}"
                )
                logger.info(
                    f"     SL:     {sl_price:{price_fmt}}  ({sl_reason})"
                )
                logger.info(
                    f"     TP:     {tp_price:{price_fmt}}  (1.5R)"
                )
                logger.info("     Q:      %s/100", quality)
                logger.info("%s", "=" * 62)

                sheet_record = self._build_sheet_record(
                    signal_id=signal_id,
                    now_ist=now_ist,
                    market=market,
                    signal_json=sig_json,
                    track=track,
                    quality=quality,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_reason=sl_reason,
                    ema_detail=ema_detail,
                    killzone_result=killzone_result,
                    london_blocked=london_blocked,
                    london_reason=london_reason,
                )

                try:
                    sheet_result = self.sheet_logger.append_record(sheet_record)
                except Exception as exc:
                    self._log_rejection(
                        now_ist=now_ist,
                        market=market,
                        symbol=symbol,
                        direction=direction_str,
                        entry_price=ob_entry.entry_price,
                        quality=quality,
                        signal_time_ist=sig_json.get("signal_datetime_ist", ""),
                        reason=f"Google Sheets append failed: {exc}",
                    )
                    self._update_status(
                        last_error=str(exc),
                        last_error_at_ist=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
                    )
                    continue

                trade_record = {
                    "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "record_id": signal_id,
                    "market": market,
                    "symbol": symbol,
                    "direction": direction_str,
                    "entry_price": ob_entry.entry_price,
                    "smart_sl": sl_price,
                    "sl_reason": sl_reason,
                    "tp": tp_price,
                    "quality": quality,
                    "signal_time_ist": sig_json.get("signal_datetime_ist", ""),
                    "ob_top": ob_entry.order_block.top,
                    "ob_bottom": ob_entry.order_block.bottom,
                    "sheet_result": sheet_result,
                    "trade_result": {
                        "success": True,
                        "mode": "GOOGLE_SHEETS_ONLY",
                        "worksheet": self.config.google_worksheet_title,
                    },
                    "signal_json": sig_json,
                }
                _append_jsonl(self.trades_log, trade_record)

                self.sent.add(signal_id)
                self._persist_sent_state()
                track.alerted = True
                track.completed = True
                cycle_logged += 1

                snapshot = self.snapshot()
                self._update_status(
                    signals_logged=snapshot["signals_logged"] + 1,
                    last_trade_ist=now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                    last_trade_id=signal_id,
                    last_error=None,
                    last_error_at_ist=None,
                )
                logger.info(
                    "Logged signal to sheet | %s | %s | %s | entry=%s",
                    market,
                    symbol,
                    direction_str,
                    ob_entry.entry_price,
                )

        pairs_scanned = sum(
            1 for _, market, _ in self.pairs
            if market == "CRYPTO" or (market == "FOREX" and forex_open)
        )
        cycle_time_ist = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
        _append_jsonl(
            self.cycle_log,
            {
                "timestamp_ist": cycle_time_ist,
                "cycle": cycle_str,
                "phases": label,
                "pairs_scanned": pairs_scanned,
                "signals_logged_to_sheet": cycle_logged,
            },
        )
        self.phase_log.log_cycle(
            cycle_time_ist=cycle_time_ist,
            symbols_scanned=pairs_scanned,
            signals_generated=cycle_logged,
        )
        if cycle_logged == 0:
            logger.info("  -- No trades this cycle --")
        else:
            logger.info("  %s trade(s) logged to Google Sheets", cycle_logged)
        snapshot = self.snapshot()
        self._update_status(
            last_cycle_ist=cycle_time_ist,
            last_cycle_label=label,
            cycles_completed=snapshot["cycles_completed"] + 1,
        )

        elapsed = (datetime.now(IST) - loop_start).total_seconds()
        self.first_run = False
        wait_seconds = max(1, self.config.poll_seconds - int(elapsed))
        self.stop_event.wait(wait_seconds)
        return True

    def _build_signal(
        self,
        symbol: str,
        track: BiasTrack,
        ob_entry: OBEntrySetup,
        quality: int,
    ) -> MSSOB_Signal:
        signal_direction = (
            Direction.BULLISH if track.bias.direction == BiasType.BULLISH
            else Direction.BEARISH
        )
        return MSSOB_Signal(
            index=0,
            datetime=ob_entry.timestamp,
            direction=signal_direction,
            entry_price=ob_entry.entry_price,
            stop_loss=ob_entry.stop_loss,
            take_profit=ob_entry.tp_price,
            risk_reward=ob_entry.risk_reward,
            signal_type="MSS_OB_ENTRY",
            symbol=symbol,
            daily_bias=track.bias,
            mss_confirmation=track.mss,
            ob_entry=ob_entry,
            quality_score=quality,
        )

    def _build_sheet_record(
        self,
        signal_id: str,
        now_ist: datetime,
        market: str,
        signal_json: Dict[str, Any],
        track: BiasTrack,
        quality: int,
        sl_price: float,
        tp_price: float,
        sl_reason: str,
        ema_detail: str,
        killzone_result: FilterResult,
        london_blocked: bool = False,
        london_reason: str = "",
    ) -> Dict[str, Any]:
        ob_entry = signal_json.get("ob_entry", {})
        daily_bias = signal_json.get("daily_bias", {})
        mss_confirmation = signal_json.get("mss_confirmation", {})
        entry_price = signal_json.get("entry_price")
        return {
            "record_id": signal_id,
            "logged_at_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
            "detected_at_ist": signal_json.get("detected_at_ist", ""),
            "signal_time_ist": signal_json.get("signal_datetime_ist", ""),
            "market": market,
            "symbol": signal_json.get("symbol", ""),
            "direction": signal_json.get("direction", ""),
            "trade_status": "SIGNAL_LOGGED",
            "source_strategy": "strategy_09_mss_ob_entry",
            "delivery_target": self.config.google_worksheet_title,
            "quality_score": quality,
            "risk_reward": 1.5,
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_per_unit": abs(float(entry_price) - float(sl_price)),
            "sl_reason": sl_reason,
            "london_block_status": "BLOCKED" if london_blocked else "PASS",
            "london_block_reason": london_reason,
            "killzone_status": "PASS" if killzone_result.passed else "FAIL",
            "killzone_reason": killzone_result.reason,
            "ema_filter_status": "SKIPPED" if self.config.no_ema_filter else "PASS",
            "ema_filter_detail": ema_detail,
            "bias_direction": daily_bias.get("direction", ""),
            "bias_confidence": daily_bias.get("confidence", ""),
            "bias_reason": daily_bias.get("reason", ""),
            "bias_sweep_time_ist": daily_bias.get("sweep_time_ist", ""),
            "bias_source_time_ist": daily_bias.get("source_time_ist", ""),
            "mss_time_ist": mss_confirmation.get("time_ist", ""),
            "mss_break_price": mss_confirmation.get("break_price", ""),
            "mss_confirmation_close": mss_confirmation.get("confirmation_close", ""),
            "mss_details": mss_confirmation.get("details", ""),
            "ob_time_ist": ob_entry.get("ob_time_ist", ""),
            "ob_top": ob_entry.get("ob_top", ""),
            "ob_bottom": ob_entry.get("ob_bottom", ""),
            "ob_body_top": ob_entry.get("ob_body_top", ""),
            "ob_body_bottom": ob_entry.get("ob_body_bottom", ""),
            "ob_fib_level": ob_entry.get("fib_level", ""),
            "ob_in_ote_zone": ob_entry.get("in_ote_zone", ""),
            "vm_hostname": self.hostname,
            "raw_signal_json": {
                **signal_json,
                "sheet_meta": {
                    "market": market,
                    "smart_sl": sl_price,
                    "tp_price": tp_price,
                    "sl_reason": sl_reason,
                    "bias_direction": track.bias.direction.value,
                },
            },
        }

    def _load_sent_state(self) -> Set[str]:
        if not self.state_file.exists():
            return set()
        try:
            return set(json.loads(self.state_file.read_text("utf-8")))
        except Exception:
            return set()

    def _persist_sent_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(sorted(self.sent), indent=2),
            encoding="utf-8",
        )

    def _log_rejection(
        self,
        now_ist: datetime,
        market: str,
        symbol: str,
        direction: str,
        entry_price: float,
        quality: int,
        signal_time_ist: str,
        reason: str,
    ) -> None:
        logger.info(
            "  [%s] %s rejected: %s",
            symbol,
            direction,
            reason,
        )
        _append_jsonl(
            self.trades_log,
            {
                "timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                "market": market,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "quality": quality,
                "signal_time_ist": signal_time_ist,
                "trade_result": {
                    "success": False,
                    "error": reason,
                },
            },
        )

    def _update_status(self, **updates: Any) -> None:
        with self.status_lock:
            self.status.update(updates)

    @staticmethod
    def _normalize_utc(timestamp: Any) -> datetime:
        value = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
        if value.tzinfo is None:
            value = UTC.localize(value)
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    service = Strategy09SheetService(config)
    app.state.sheet_service = service
    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(
    title="Strategy 09 Google Sheets Signal Server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root(request: Request) -> Dict[str, Any]:
    service: Strategy09SheetService = request.app.state.sheet_service
    return {
        "service": "strategy_09_google_sheets_server",
        "status": "running" if service.snapshot()["worker_running"] else "stopped",
        "health_endpoint": "/health",
        "status_endpoint": "/status",
        "worksheet": service.config.google_worksheet_title,
    }


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    service: Strategy09SheetService = request.app.state.sheet_service
    snapshot = service.snapshot()
    return {
        "ok": snapshot["worker_running"] and snapshot["last_error"] is None,
        "worker_running": snapshot["worker_running"],
        "last_cycle_ist": snapshot["last_cycle_ist"],
        "last_error": snapshot["last_error"],
        "worksheet": service.config.google_worksheet_title,
    }


@app.get("/status")
def status(request: Request) -> Dict[str, Any]:
    service: Strategy09SheetService = request.app.state.sheet_service
    return {
        "runtime": service.snapshot(),
        "config": service.config_summary(),
    }


@app.post("/test")
@app.get("/test")
async def test_sheet_write(request: Request) -> Dict[str, Any]:
    """Write a random test entry to verify Google Sheets connectivity.

    This also ensures headers are present — if missing, they are auto-created
    by the GoogleSheetsTradeLogger._ensure_headers() method.
    """
    import random
    service: Strategy09SheetService = request.app.state.sheet_service

    now_ist = datetime.now(IST)
    test_symbols = ["EURUSD", "GBPJPY", "XAUUSD", "BTCUSD", "ETHUSD"]
    test_directions = ["bullish", "bearish"]
    symbol = random.choice(test_symbols)
    direction = random.choice(test_directions)
    entry_price = round(random.uniform(1.0, 2000.0), 5)
    sl_price = round(entry_price * (0.995 if direction == "bullish" else 1.005), 5)
    tp_price = round(entry_price * (1.01 if direction == "bullish" else 0.99), 5)

    test_record = {
        "record_id": f"TEST_{now_ist.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}",
        "logged_at_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "detected_at_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "signal_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market": "TEST",
        "symbol": symbol,
        "direction": direction,
        "trade_status": "TEST_ENTRY",
        "source_strategy": "strategy_09_mss_ob_entry",
        "delivery_target": service.config.google_worksheet_title,
        "quality_score": random.randint(60, 95),
        "risk_reward": 1.5,
        "entry_price": entry_price,
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "risk_per_unit": abs(entry_price - sl_price),
        "sl_reason": "TEST — smart SL on OB boundary",
        "london_block_status": "PASS",
        "london_block_reason": "Test entry — outside London block",
        "killzone_status": "PASS",
        "killzone_reason": "Test entry — killzone check skipped",
        "ema_filter_status": "PASS",
        "ema_filter_detail": "Test entry — EMA filter skipped",
        "bias_direction": direction,
        "bias_confidence": "high",
        "bias_reason": "TEST — 4H liquidity sweep",
        "bias_sweep_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "bias_source_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "mss_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "mss_break_price": entry_price,
        "mss_confirmation_close": entry_price,
        "mss_details": "TEST MSS confirmation",
        "ob_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "ob_top": round(entry_price * 1.002, 5),
        "ob_bottom": round(entry_price * 0.998, 5),
        "ob_body_top": round(entry_price * 1.001, 5),
        "ob_body_bottom": round(entry_price * 0.999, 5),
        "ob_fib_level": round(random.uniform(0.618, 0.786), 3),
        "ob_in_ote_zone": True,
        "vm_hostname": service.hostname,
        "raw_signal_json": {"test": True, "timestamp": now_ist.isoformat()},
    }

    try:
        result = service.sheet_logger.append_record(test_record)
        return {
            "success": True,
            "message": f"Test entry written to sheet '{service.config.google_worksheet_title}'",
            "record_id": test_record["record_id"],
            "symbol": symbol,
            "direction": direction,
            "sheet_result": result,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "message": "Failed to write test entry to Google Sheets",
        }


if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "strategy_09_sheet_server.server:app",
        host=config.host,
        port=config.port,
        reload=False,
        workers=1,
    )
