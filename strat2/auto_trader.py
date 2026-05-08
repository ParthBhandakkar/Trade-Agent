"""
Strat2: HTF trend continuation after liquidity sweep.

Standalone MT5 forex bot. It does not import any files from the rest of this
repo. The model is deliberately strict:

1. Daily + 4H trend alignment.
2. 15M liquidity sweep against the trend.
3. 5M market-structure break back with the trend.
4. 5M fair-value-gap retracement entry.
5. Stop beyond the swept liquidity, TP split at 1R and external liquidity.

Run:
    python auto_trader.py --once
    python auto_trader.py
    python auto_trader.py --live
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MetaTrader5 is required: pip install MetaTrader5") from exc


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
STATE_DIR = BASE_DIR / "state"
ENV_PATH = BASE_DIR / ".env"

IST = ZoneInfo("Asia/Kolkata") if ZoneInfo else timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Environment and config
# ---------------------------------------------------------------------------


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(key: str, default: int) -> int:
    try:
        return int(float(os.getenv(key, str(default)).strip()))
    except Exception:
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def env_list(key: str, default: Iterable[str]) -> List[str]:
    value = os.getenv(key)
    if not value:
        return list(default)
    return [part.strip().upper() for part in value.split(",") if part.strip()]


DEFAULT_FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "AUDJPY",
    "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    "AUDCAD", "AUDCHF", "AUDNZD",
    "NZDCAD", "NZDCHF", "CADJPY", "CHFJPY",
]


@dataclass
class Config:
    mt5_login: int
    mt5_password: str
    mt5_server: str
    pairs: List[str]
    poll_seconds: int
    dry_run: bool
    magic: int
    risk_inr: float
    risk_pct_equity: float
    max_risk_inr: float
    max_open_positions: int
    max_daily_loss_inr: float
    min_quality: int
    max_spread_pips: float
    min_stop_pips: float
    sweep_lookback_15m: int
    max_sweep_age_min: int
    max_fvg_age_min: int
    fvg_touch_tolerance_pips: float
    tp1_r: float
    tp2_min_r: float
    tp1_lot_fraction: float
    allow_asian: bool
    allow_london: bool
    allow_ny: bool
    once: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        load_env()
        login_raw = os.getenv("MT5_LOGIN", "0").strip()
        try:
            login = int(login_raw)
        except ValueError:
            login = 0
        return cls(
            mt5_login=login,
            mt5_password=os.getenv("MT5_PASSWORD", "").strip(),
            mt5_server=os.getenv("MT5_SERVER", "").strip(),
            pairs=env_list("STRAT2_FOREX_PAIRS", DEFAULT_FOREX_PAIRS),
            poll_seconds=env_int("STRAT2_POLL_SECONDS", 60),
            dry_run=env_bool("STRAT2_DRY_RUN", True),
            magic=env_int("STRAT2_MAGIC", 909102),
            risk_inr=env_float("STRAT2_RISK_INR", 500.0),
            risk_pct_equity=env_float("STRAT2_RISK_PCT_EQUITY", 0.0),
            max_risk_inr=env_float("STRAT2_MAX_RISK_INR", 1000.0),
            max_open_positions=env_int("STRAT2_MAX_OPEN_POSITIONS", 2),
            max_daily_loss_inr=env_float("STRAT2_MAX_DAILY_LOSS_INR", 1500.0),
            min_quality=env_int("STRAT2_MIN_QUALITY", 78),
            max_spread_pips=env_float("STRAT2_MAX_SPREAD_PIPS", 2.2),
            min_stop_pips=env_float("STRAT2_MIN_STOP_PIPS", 6.0),
            sweep_lookback_15m=env_int("STRAT2_SWEEP_LOOKBACK_15M", 96),
            max_sweep_age_min=env_int("STRAT2_MAX_SWEEP_AGE_MIN", 240),
            max_fvg_age_min=env_int("STRAT2_MAX_FVG_AGE_MIN", 180),
            fvg_touch_tolerance_pips=env_float("STRAT2_FVG_TOUCH_TOLERANCE_PIPS", 0.5),
            tp1_r=env_float("STRAT2_TP1_R", 1.0),
            tp2_min_r=env_float("STRAT2_TP2_MIN_R", 1.8),
            tp1_lot_fraction=env_float("STRAT2_TP1_LOT_FRACTION", 0.70),
            allow_asian=env_bool("STRAT2_ALLOW_ASIAN", False),
            allow_london=env_bool("STRAT2_ALLOW_LONDON", True),
            allow_ny=env_bool("STRAT2_ALLOW_NY", True),
        )


# ---------------------------------------------------------------------------
# Logging and small utilities
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("strat2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "strat2.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


LOGGER = setup_logging()


def now_ist() -> datetime:
    return datetime.now(tz=IST)


def to_ist_str(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def floor_to_step(value: float, step: float, digits: int = 2) -> float:
    if step <= 0:
        return round(value, digits)
    return round(math.floor(value / step) * step, digits)


def is_jpy_pair(symbol: str) -> bool:
    return "JPY" in symbol.upper()


def pip_size(symbol: str, info: Any = None) -> float:
    if is_jpy_pair(symbol):
        return 0.01
    if info is not None and getattr(info, "digits", 5) in (2, 3):
        return 0.01
    return 0.0001


def drop_incomplete(df: pd.DataFrame, interval_min: int, current_utc: datetime) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    last_ts = df.index[-1].to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)
    if current_utc < last_ts + timedelta(minutes=interval_min):
        return df.iloc[:-1]
    return df


def in_allowed_session(ts_utc: datetime, cfg: Config) -> Tuple[bool, str]:
    minute = ts_utc.hour * 60 + ts_utc.minute
    london = 7 * 60 <= minute <= 10 * 60 + 30
    ny = 13 * 60 <= minute <= 16 * 60 + 30
    asian = 0 <= minute <= 5 * 60
    if london and cfg.allow_london:
        return True, "London"
    if ny and cfg.allow_ny:
        return True, "NewYork"
    if asian and cfg.allow_asian:
        return True, "Asian"
    return False, "Outside allowed session"


# ---------------------------------------------------------------------------
# MT5 wrapper
# ---------------------------------------------------------------------------


class MT5Bridge:
    TF = {
        "D1": mt5.TIMEFRAME_D1,
        "H4": mt5.TIMEFRAME_H4,
        "H1": mt5.TIMEFRAME_H1,
        "M15": mt5.TIMEFRAME_M15,
        "M5": mt5.TIMEFRAME_M5,
    }
    TF_MIN = {"D1": 1440, "H4": 240, "H1": 60, "M15": 15, "M5": 5}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.symbol_cache: Dict[str, str] = {}
        self.connected = False

    def connect(self) -> bool:
        if not mt5.initialize():
            LOGGER.error("MT5 initialize failed: %s", mt5.last_error())
            return False
        if self.cfg.mt5_login and self.cfg.mt5_password:
            if not mt5.login(
                self.cfg.mt5_login,
                password=self.cfg.mt5_password,
                server=self.cfg.mt5_server or None,
            ):
                LOGGER.error("MT5 login failed: %s", mt5.last_error())
                return False
        account = mt5.account_info()
        if account is None:
            LOGGER.error("MT5 account unavailable: %s", mt5.last_error())
            return False
        self.connected = True
        LOGGER.info(
            "MT5 connected: login=%s server=%s currency=%s leverage=1:%s",
            account.login,
            account.server,
            account.currency,
            account.leverage,
        )
        return True

    def ensure_connected(self) -> bool:
        if self.connected and mt5.account_info() is not None:
            return True
        return self.connect()

    def shutdown(self) -> None:
        try:
            mt5.shutdown()
        except Exception:
            pass

    def resolve_symbol(self, requested: str) -> Optional[str]:
        key = requested.upper()
        if key in self.symbol_cache:
            return self.symbol_cache[key]
        candidates = [key, f"{key}m", f"{key}.m"]
        for candidate in candidates:
            if mt5.symbol_select(candidate, True):
                self.symbol_cache[key] = candidate
                return candidate
        matches = mt5.symbols_get(f"*{key}*") or []
        for item in matches:
            name = getattr(item, "name", "")
            if name and mt5.symbol_select(name, True):
                self.symbol_cache[key] = name
                return name
        LOGGER.warning("[%s] symbol not found in MT5", requested)
        return None

    def fetch(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        if not self.ensure_connected():
            return None
        mt5_symbol = self.resolve_symbol(symbol)
        if not mt5_symbol:
            return None
        rates = mt5.copy_rates_from_pos(mt5_symbol, self.TF[timeframe], 0, bars)
        if rates is None or len(rates) == 0:
            LOGGER.warning("[%s] no %s rates: %s", symbol, timeframe, mt5.last_error())
            return None
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)
        volume_col = "tick_volume" if "tick_volume" in df.columns else "real_volume"
        df["volume"] = df[volume_col] if volume_col in df.columns else 0
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        return drop_incomplete(df, self.TF_MIN[timeframe], datetime.now(tz=UTC))

    def fetch_all(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        bars = {"D1": 260, "H4": 520, "H1": 800, "M15": 1500, "M5": 2500}
        data = {tf: self.fetch(symbol, tf, n) for tf, n in bars.items()}
        if any(df is None or df.empty for df in data.values()):
            return None
        return data

    def tick(self, symbol: str) -> Optional[Any]:
        mt5_symbol = self.resolve_symbol(symbol)
        if not mt5_symbol:
            return None
        return mt5.symbol_info_tick(mt5_symbol)

    def symbol_info(self, symbol: str) -> Optional[Any]:
        mt5_symbol = self.resolve_symbol(symbol)
        if not mt5_symbol:
            return None
        return mt5.symbol_info(mt5_symbol)

    def spread_pips(self, symbol: str) -> Optional[float]:
        info = self.symbol_info(symbol)
        tick = self.tick(symbol)
        if info is None or tick is None:
            return None
        pip = pip_size(symbol, info)
        spread = abs(float(tick.ask) - float(tick.bid)) / pip
        if spread <= 0:
            point = float(getattr(info, "point", 0.0) or 0.0)
            broker_spread = float(getattr(info, "spread", 0.0) or 0.0)
            if point > 0 and broker_spread > 0:
                spread = broker_spread * point / pip
        return spread

    def account_info(self) -> Optional[Any]:
        return mt5.account_info()

    def open_positions(self) -> List[Any]:
        positions = mt5.positions_get() or []
        return [p for p in positions if getattr(p, "magic", 0) == self.cfg.magic]

    def today_closed_pnl(self) -> float:
        start = datetime.now(tz=IST).replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(start.astimezone(UTC), datetime.now(tz=UTC)) or []
        pnl = 0.0
        for deal in deals:
            if getattr(deal, "magic", 0) == self.cfg.magic:
                pnl += float(getattr(deal, "profit", 0.0))
                pnl += float(getattr(deal, "commission", 0.0))
                pnl += float(getattr(deal, "swap", 0.0))
        return pnl

    def fx_rate(self, from_ccy: str, to_ccy: str) -> Optional[float]:
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()
        if from_ccy == to_ccy:
            return 1.0
        if from_ccy == "USD" and to_ccy == "INR":
            return 84.0
        if from_ccy == "INR" and to_ccy == "USD":
            return 1.0 / 84.0
        for direct, inverse in ((from_ccy + to_ccy, False), (to_ccy + from_ccy, True)):
            resolved = self.resolve_symbol(direct)
            if not resolved:
                continue
            tick = mt5.symbol_info_tick(resolved)
            if tick is None:
                continue
            if inverse and getattr(tick, "ask", 0) > 0:
                return 1.0 / float(tick.ask)
            if not inverse and getattr(tick, "bid", 0) > 0:
                return float(tick.bid)
        if from_ccy != "USD" and to_ccy != "USD":
            a = self.fx_rate(from_ccy, "USD")
            b = self.fx_rate("USD", to_ccy)
            if a and b:
                return a * b
        return None

    def estimate_loss(self, symbol: str, direction: str, lot: float, entry: float, sl: float) -> Optional[float]:
        mt5_symbol = self.resolve_symbol(symbol)
        if not mt5_symbol:
            return None
        order_type = mt5.ORDER_TYPE_BUY if direction == "bullish" else mt5.ORDER_TYPE_SELL
        profit = mt5.order_calc_profit(order_type, mt5_symbol, lot, entry, sl)
        if profit is not None and abs(float(profit)) > 0:
            return abs(float(profit))

        info = mt5.symbol_info(mt5_symbol)
        account = mt5.account_info()
        if info is None or account is None:
            return None
        pip = pip_size(symbol, info)
        pips = abs(entry - sl) / pip
        quote = symbol.upper()[3:6] if len(symbol) >= 6 else "USD"
        pip_value_quote = float(info.trade_contract_size) * pip * lot
        rate = self.fx_rate(quote, account.currency)
        if rate is None:
            rate = self.fx_rate("USD", account.currency) or 84.0
        return pips * pip_value_quote * rate

    def calc_lot_for_risk(self, symbol: str, direction: str, entry: float, sl: float) -> float:
        info = self.symbol_info(symbol)
        account = self.account_info()
        if info is None or account is None:
            return 0.0

        risk_amount = self.cfg.risk_inr
        if self.cfg.risk_pct_equity > 0:
            risk_amount = float(account.equity) * self.cfg.risk_pct_equity / 100.0
        risk_amount = min(risk_amount, self.cfg.max_risk_inr)

        loss_1_lot = self.estimate_loss(symbol, direction, 1.0, entry, sl)
        if not loss_1_lot or loss_1_lot <= 0:
            return float(info.volume_min)
        raw = risk_amount / loss_1_lot
        step = float(info.volume_step)
        decimals = max(0, len(str(step).rstrip("0").split(".")[-1]))
        lot = floor_to_step(raw, step, decimals)
        lot = max(float(info.volume_min), min(lot, float(info.volume_max)))

        mt5_symbol = self.resolve_symbol(symbol)
        order_type = mt5.ORDER_TYPE_BUY if direction == "bullish" else mt5.ORDER_TYPE_SELL
        margin = mt5.order_calc_margin(order_type, mt5_symbol, lot, entry) if mt5_symbol else None
        free = float(getattr(account, "margin_free", 0.0))
        if margin and margin > free * 0.60:
            scale = (free * 0.60) / margin
            lot = floor_to_step(lot * scale, step, decimals)
            lot = max(float(info.volume_min), lot)
        return round(lot, decimals)

    def send_market_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str,
    ) -> Dict[str, Any]:
        mt5_symbol = self.resolve_symbol(symbol)
        info = self.symbol_info(symbol)
        tick = self.tick(symbol)
        if not mt5_symbol or info is None or tick is None:
            return {"success": False, "error": "missing symbol info or tick"}
        is_buy = direction == "bullish"
        price = float(tick.ask if is_buy else tick.bid)
        digits = int(info.digits)
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": round(price, digits),
            "sl": round(sl, digits),
            "tp": round(tp, digits),
            "deviation": 20,
            "magic": self.cfg.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        check = mt5.order_check(req)
        result = {
            "request": {k: v for k, v in req.items() if k != "action"},
            "order_check": {
                "retcode": getattr(check, "retcode", None),
                "comment": getattr(check, "comment", None),
                "margin": getattr(check, "margin", None),
                "margin_free": getattr(check, "margin_free", None),
            } if check is not None else None,
        }
        sent = mt5.order_send(req)
        if sent is None:
            result.update({"success": False, "error": f"order_send None: {mt5.last_error()}"})
            return result
        result.update({
            "success": sent.retcode == mt5.TRADE_RETCODE_DONE,
            "retcode": sent.retcode,
            "comment": sent.comment,
            "order": str(sent.order),
            "deal": str(sent.deal),
            "exec_price": sent.price,
            "volume": volume,
            "sl": round(sl, digits),
            "tp": round(tp, digits),
        })
        if sent.retcode != mt5.TRADE_RETCODE_DONE:
            result["error"] = f"Rejected: {sent.comment} ({sent.retcode})"
        return result

    def modify_sl(self, ticket: int, symbol: str, sl: float) -> bool:
        mt5_symbol = self.resolve_symbol(symbol)
        info = self.symbol_info(symbol)
        if not mt5_symbol or info is None:
            return False
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": mt5_symbol,
            "sl": round(float(sl), int(info.digits)),
            "tp": 0.0,
            "magic": self.cfg.magic,
        }
        res = mt5.order_send(req)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)


# ---------------------------------------------------------------------------
# Strategy objects and indicator helpers
# ---------------------------------------------------------------------------


@dataclass
class Swing:
    kind: str
    index: int
    timestamp: Any
    price: float


@dataclass
class Trend:
    direction: str
    quality: int
    reason: str


@dataclass
class Sweep:
    direction: str
    timestamp: Any
    level_time: Any
    level_price: float
    extreme_price: float
    close_price: float
    source: str


@dataclass
class BreakSignal:
    direction: str
    timestamp: Any
    break_level: float
    close_price: float
    body_atr: float


@dataclass
class FVG:
    direction: str
    timestamp: Any
    low: float
    high: float
    ce: float


@dataclass
class TradeSignal:
    symbol: str
    direction: str
    timestamp: Any
    entry_model_price: float
    sl: float
    tp1: float
    tp2: float
    risk: float
    quality: int
    trend: Trend
    sweep: Sweep
    break_signal: BreakSignal
    fvg: FVG
    session: str
    spread_pips: float
    notes: List[str]

    def signal_id(self) -> str:
        return "|".join([
            self.symbol,
            self.direction,
            to_ist_str(self.fvg.timestamp) or "",
            to_ist_str(self.timestamp) or "",
        ])

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp_ist"] = to_ist_str(self.timestamp)
        data["trend"]["timestamp_note"] = "D1/H4 closed candles"
        data["sweep"]["timestamp_ist"] = to_ist_str(self.sweep.timestamp)
        data["sweep"]["level_time_ist"] = to_ist_str(self.sweep.level_time)
        data["break_signal"]["timestamp_ist"] = to_ist_str(self.break_signal.timestamp)
        data["fvg"]["timestamp_ist"] = to_ist_str(self.fvg.timestamp)
        return data


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[Swing]:
    swings: List[Swing] = []
    if df is None or len(df) < left + right + 2:
        return swings
    for idx in range(left, len(df) - right):
        window = df.iloc[idx - left:idx + right + 1]
        row = df.iloc[idx]
        if row["high"] == window["high"].max() and (window["high"] == row["high"]).sum() == 1:
            swings.append(Swing("high", idx, df.index[idx], float(row["high"])))
        if row["low"] == window["low"].min() and (window["low"] == row["low"]).sum() == 1:
            swings.append(Swing("low", idx, df.index[idx], float(row["low"])))
    return swings


def candle_body(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def candle_range(row: pd.Series) -> float:
    return max(float(row["high"]) - float(row["low"]), 1e-12)


def detect_trend(d1: pd.DataFrame, h4: pd.DataFrame) -> Optional[Trend]:
    if len(d1) < 80 or len(h4) < 220:
        return None
    d1_close = d1["close"]
    h4_close = h4["close"]
    d1_ema50 = ema(d1_close, 50)
    h4_ema50 = ema(h4_close, 50)
    h4_ema200 = ema(h4_close, 200)

    d1_last = float(d1_close.iloc[-1])
    h4_last = float(h4_close.iloc[-1])
    h4_50 = float(h4_ema50.iloc[-1])
    h4_200 = float(h4_ema200.iloc[-1])
    d1_50 = float(d1_ema50.iloc[-1])
    h4_slope_up = h4_ema50.iloc[-1] > h4_ema50.iloc[-6]
    h4_slope_down = h4_ema50.iloc[-1] < h4_ema50.iloc[-6]

    h4_swings = find_swings(h4.tail(140), 2, 2)
    highs = [s.price for s in h4_swings if s.kind == "high"][-3:]
    lows = [s.price for s in h4_swings if s.kind == "low"][-3:]
    structure_up = len(highs) >= 2 and len(lows) >= 2 and highs[-1] > highs[-2] and lows[-1] > lows[-2]
    structure_down = len(highs) >= 2 and len(lows) >= 2 and highs[-1] < highs[-2] and lows[-1] < lows[-2]

    if d1_last > d1_50 and h4_last > h4_50 > h4_200 and h4_slope_up:
        quality = 70 + (10 if structure_up else 0)
        return Trend("bullish", quality, "D1 above EMA50 and H4 close > EMA50 > EMA200")
    if d1_last < d1_50 and h4_last < h4_50 < h4_200 and h4_slope_down:
        quality = 70 + (10 if structure_down else 0)
        return Trend("bearish", quality, "D1 below EMA50 and H4 close < EMA50 < EMA200")
    return None


def detect_liquidity_sweep(m15: pd.DataFrame, direction: str, cfg: Config) -> Optional[Sweep]:
    swings = find_swings(m15, 2, 2)
    if not swings:
        return None
    now_ts = m15.index[-1]
    max_age = timedelta(minutes=cfg.max_sweep_age_min)
    candidates: List[Sweep] = []

    for idx in range(max(5, len(m15) - cfg.sweep_lookback_15m), len(m15)):
        row = m15.iloc[idx]
        ts = m15.index[idx]
        if now_ts - ts > max_age:
            continue
        prior = [s for s in swings if s.index < idx]
        if direction == "bullish":
            lows = [s for s in prior if s.kind == "low"]
            if not lows:
                continue
            level = lows[-1]
            if float(row["low"]) < level.price and float(row["close"]) > level.price:
                candidates.append(Sweep(direction, ts, level.timestamp, level.price, float(row["low"]), float(row["close"]), "M15 swing low"))
        else:
            highs = [s for s in prior if s.kind == "high"]
            if not highs:
                continue
            level = highs[-1]
            if float(row["high"]) > level.price and float(row["close"]) < level.price:
                candidates.append(Sweep(direction, ts, level.timestamp, level.price, float(row["high"]), float(row["close"]), "M15 swing high"))

    if not candidates:
        return None
    return candidates[-1]


def detect_break_after_sweep(m5: pd.DataFrame, sweep: Sweep, direction: str) -> Optional[BreakSignal]:
    swings = find_swings(m5, 2, 2)
    start = m5.index.searchsorted(sweep.timestamp)
    if start >= len(m5) - 2:
        return None
    atr5 = atr(m5, 14)
    prior = [s for s in swings if s.index < start]
    if direction == "bullish":
        highs = [s for s in prior if s.kind == "high"]
        if not highs:
            return None
        level = highs[-1].price
        for idx in range(start + 1, len(m5)):
            row = m5.iloc[idx]
            body_ratio = candle_body(row) / max(float(atr5.iloc[idx] or 0), 1e-12)
            if float(row["close"]) > level and float(row["close"]) > float(row["open"]) and body_ratio >= 0.35:
                return BreakSignal(direction, m5.index[idx], level, float(row["close"]), float(body_ratio))
    else:
        lows = [s for s in prior if s.kind == "low"]
        if not lows:
            return None
        level = lows[-1].price
        for idx in range(start + 1, len(m5)):
            row = m5.iloc[idx]
            body_ratio = candle_body(row) / max(float(atr5.iloc[idx] or 0), 1e-12)
            if float(row["close"]) < level and float(row["close"]) < float(row["open"]) and body_ratio >= 0.35:
                return BreakSignal(direction, m5.index[idx], level, float(row["close"]), float(body_ratio))
    return None


def detect_fvgs_after_break(m5: pd.DataFrame, break_signal: BreakSignal, direction: str, cfg: Config) -> List[FVG]:
    start = max(2, m5.index.searchsorted(break_signal.timestamp) - 2)
    last_ts = m5.index[-1]
    fvgs: List[FVG] = []
    for idx in range(start, len(m5)):
        if last_ts - m5.index[idx] > timedelta(minutes=cfg.max_fvg_age_min):
            continue
        a = m5.iloc[idx - 2]
        c = m5.iloc[idx]
        if direction == "bullish":
            if float(c["low"]) > float(a["high"]):
                low = float(a["high"])
                high = float(c["low"])
                fvgs.append(FVG(direction, m5.index[idx], low, high, (low + high) / 2.0))
        else:
            if float(c["high"]) < float(a["low"]):
                low = float(c["high"])
                high = float(a["low"])
                fvgs.append(FVG(direction, m5.index[idx], low, high, (low + high) / 2.0))
    return fvgs


def fvg_tapped_on_last_candle(m5: pd.DataFrame, fvg: FVG, direction: str, tolerance: float) -> bool:
    last = m5.iloc[-1]
    if m5.index[-1] <= fvg.timestamp:
        return False
    if direction == "bullish":
        return float(last["low"]) <= fvg.ce + tolerance and float(last["close"]) > fvg.ce
    return float(last["high"]) >= fvg.ce - tolerance and float(last["close"]) < fvg.ce


def external_liquidity_target(m15: pd.DataFrame, direction: str, entry: float, fallback: float) -> float:
    swings = find_swings(m15.tail(220), 2, 2)
    if direction == "bullish":
        highs = sorted([s.price for s in swings if s.kind == "high" and s.price > entry])
        if highs:
            return max(fallback, highs[0])
        return fallback
    lows = sorted([s.price for s in swings if s.kind == "low" and s.price < entry], reverse=True)
    if lows:
        return min(fallback, lows[0])
    return fallback


def build_signal(symbol: str, data: Dict[str, pd.DataFrame], bridge: MT5Bridge, cfg: Config) -> Optional[TradeSignal]:
    trend = detect_trend(data["D1"], data["H4"])
    if trend is None:
        return None
    session_ok, session_name = in_allowed_session(data["M5"].index[-1].to_pydatetime(), cfg)
    if not session_ok:
        return None

    spread = bridge.spread_pips(symbol)
    if spread is None or spread > cfg.max_spread_pips:
        return None

    direction = trend.direction
    sweep = detect_liquidity_sweep(data["M15"], direction, cfg)
    if sweep is None:
        return None
    break_signal = detect_break_after_sweep(data["M5"], sweep, direction)
    if break_signal is None:
        return None
    fvgs = detect_fvgs_after_break(data["M5"], break_signal, direction, cfg)
    if not fvgs:
        return None

    info = bridge.symbol_info(symbol)
    pip = pip_size(symbol, info)
    touch_tol = cfg.fvg_touch_tolerance_pips * pip
    selected = None
    for fvg in reversed(fvgs):
        if fvg_tapped_on_last_candle(data["M5"], fvg, direction, touch_tol):
            selected = fvg
            break
    if selected is None:
        return None

    last = data["M5"].iloc[-1]
    entry_model = float(last["close"])
    buffer = max(cfg.min_stop_pips * pip, (spread or 0) * pip * 1.5)
    if direction == "bullish":
        sl = min(sweep.extreme_price - buffer, selected.low - buffer)
        risk = entry_model - sl
        if risk <= cfg.min_stop_pips * pip:
            return None
        tp1 = entry_model + cfg.tp1_r * risk
        tp2_fallback = entry_model + cfg.tp2_min_r * risk
    else:
        sl = max(sweep.extreme_price + buffer, selected.high + buffer)
        risk = sl - entry_model
        if risk <= cfg.min_stop_pips * pip:
            return None
        tp1 = entry_model - cfg.tp1_r * risk
        tp2_fallback = entry_model - cfg.tp2_min_r * risk
    tp2 = external_liquidity_target(data["M15"], direction, entry_model, tp2_fallback)

    quality = 0
    notes: List[str] = []
    quality += min(30, int(trend.quality * 0.35))
    quality += 20
    if break_signal.body_atr >= 0.65:
        quality += 15
        notes.append("strong displacement")
    else:
        quality += 8
        notes.append("moderate displacement")
    fvg_size = abs(selected.high - selected.low) / pip
    if 1.0 <= fvg_size <= 12.0:
        quality += 15
    else:
        quality += 8
    quality += 10 if session_name in {"London", "NewYork"} else 5
    if spread <= cfg.max_spread_pips * 0.6:
        quality += 10
    else:
        quality += 5

    if quality < cfg.min_quality:
        return None

    return TradeSignal(
        symbol=symbol,
        direction=direction,
        timestamp=data["M5"].index[-1],
        entry_model_price=entry_model,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        risk=risk,
        quality=min(100, quality),
        trend=trend,
        sweep=sweep,
        break_signal=break_signal,
        fvg=selected,
        session=session_name,
        spread_pips=float(spread),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Execution and loop
# ---------------------------------------------------------------------------


def execute_signal(bridge: MT5Bridge, signal: TradeSignal, cfg: Config) -> Dict[str, Any]:
    tick = bridge.tick(signal.symbol)
    if tick is None:
        return {"success": False, "error": "no tick"}
    live_entry = float(tick.ask if signal.direction == "bullish" else tick.bid)

    # Recalculate TP distances from the executable entry, but keep the model SL.
    risk = abs(live_entry - signal.sl)
    if risk <= 0:
        return {"success": False, "error": "invalid risk after live price"}
    tp1 = live_entry + cfg.tp1_r * risk if signal.direction == "bullish" else live_entry - cfg.tp1_r * risk
    if signal.direction == "bullish":
        tp2 = max(signal.tp2, live_entry + cfg.tp2_min_r * risk)
    else:
        tp2 = min(signal.tp2, live_entry - cfg.tp2_min_r * risk)

    total_lot = bridge.calc_lot_for_risk(signal.symbol, signal.direction, live_entry, signal.sl)
    info = bridge.symbol_info(signal.symbol)
    if info is None or total_lot <= 0:
        return {"success": False, "error": "could not calculate lot"}

    min_vol = float(info.volume_min)
    step = float(info.volume_step)
    decimals = max(0, len(str(step).rstrip("0").split(".")[-1]))
    lot1 = floor_to_step(total_lot * cfg.tp1_lot_fraction, step, decimals)
    lot2 = floor_to_step(total_lot - lot1, step, decimals)
    orders = []

    if lot1 >= min_vol and lot2 >= min_vol:
        orders.append(("S2_TP1", lot1, tp1))
        orders.append(("S2_TP2", lot2, tp2))
    else:
        orders.append(("S2_FULL", total_lot, tp1))

    if cfg.dry_run:
        return {
            "success": True,
            "dry_run": True,
            "live_entry": live_entry,
            "total_lot": total_lot,
            "orders": [{"comment": c, "volume": v, "sl": signal.sl, "tp": t} for c, v, t in orders],
        }

    results = []
    ok = True
    for comment, lot, tp in orders:
        res = bridge.send_market_order(signal.symbol, signal.direction, lot, signal.sl, tp, comment)
        results.append(res)
        ok = ok and bool(res.get("success"))
    return {"success": ok, "live_entry": live_entry, "total_lot": total_lot, "orders": results}


def manage_open_positions(bridge: MT5Bridge, cfg: Config) -> None:
    for pos in bridge.open_positions():
        comment = str(getattr(pos, "comment", ""))
        if "TP2" not in comment:
            continue
        symbol = str(getattr(pos, "symbol", ""))
        direction = "bullish" if getattr(pos, "type", None) == mt5.POSITION_TYPE_BUY else "bearish"
        open_price = float(pos.price_open)
        current = float(pos.price_current)
        sl = float(pos.sl)
        initial_risk = abs(open_price - sl)
        if initial_risk <= 0:
            continue
        moved_1r = (current - open_price) >= initial_risk if direction == "bullish" else (open_price - current) >= initial_risk
        be_better = sl < open_price if direction == "bullish" else sl > open_price
        if moved_1r and be_better:
            if bridge.modify_sl(int(pos.ticket), symbol, open_price):
                append_jsonl(LOG_DIR / "management.jsonl", {
                    "timestamp_ist": to_ist_str(now_ist()),
                    "symbol": symbol,
                    "action": "MOVE_SL_BE",
                    "ticket": str(pos.ticket),
                    "old_sl": sl,
                    "new_sl": open_price,
                })


def scan_once(bridge: MT5Bridge, cfg: Config, sent: set) -> int:
    signals_found = 0
    daily_pnl = bridge.today_closed_pnl()
    if daily_pnl <= -abs(cfg.max_daily_loss_inr):
        LOGGER.warning("Daily loss limit reached: %.2f", daily_pnl)
        return 0

    for symbol in cfg.pairs:
        try:
            data = bridge.fetch_all(symbol)
            if data is None:
                continue
            signal = build_signal(symbol, data, bridge, cfg)
            if signal is None:
                continue
            sid = signal.signal_id()
            if sid in sent:
                continue
            if len(bridge.open_positions()) >= cfg.max_open_positions:
                append_jsonl(LOG_DIR / "rejections.jsonl", {
                    "timestamp_ist": to_ist_str(now_ist()),
                    "symbol": symbol,
                    "reason": "max open positions",
                    "signal": signal.to_json(),
                })
                sent.add(sid)
                continue

            append_jsonl(LOG_DIR / "signals.jsonl", signal.to_json())
            result = execute_signal(bridge, signal, cfg)
            append_jsonl(LOG_DIR / "trades.jsonl", {
                "timestamp_ist": to_ist_str(now_ist()),
                "signal_id": sid,
                "signal": signal.to_json(),
                "trade_result": result,
            })
            sent.add(sid)
            signals_found += 1
            LOGGER.info("[%s] %s signal quality=%s result=%s", symbol, signal.direction, signal.quality, result.get("success"))
        except Exception as exc:
            LOGGER.exception("[%s] scan error: %s", symbol, exc)
    return signals_found


def main() -> None:
    parser = argparse.ArgumentParser(description="Strat2 standalone MT5 trend-continuation trader")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument("--live", action="store_true", help="Override .env and place live orders")
    parser.add_argument("--dry-run", action="store_true", help="Override .env and never place orders")
    args = parser.parse_args()

    cfg = Config.from_env()
    cfg.once = bool(args.once)
    if args.live:
        cfg.dry_run = False
    if args.dry_run:
        cfg.dry_run = True

    LOGGER.info("Starting Strat2 | dry_run=%s | pairs=%s | risk_inr=%.2f", cfg.dry_run, len(cfg.pairs), cfg.risk_inr)
    bridge = MT5Bridge(cfg)
    if not bridge.connect():
        raise SystemExit(1)

    sent_path = STATE_DIR / "sent_signals.json"
    sent = set(read_json(sent_path, []))
    try:
        while True:
            loop_start = time.time()
            manage_open_positions(bridge, cfg)
            count = scan_once(bridge, cfg, sent)
            write_json(sent_path, sorted(sent))
            append_jsonl(LOG_DIR / "cycles.jsonl", {
                "timestamp_ist": to_ist_str(now_ist()),
                "signals_found": count,
                "dry_run": cfg.dry_run,
            })
            if cfg.once:
                break
            elapsed = time.time() - loop_start
            time.sleep(max(5, cfg.poll_seconds - elapsed))
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user")
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
