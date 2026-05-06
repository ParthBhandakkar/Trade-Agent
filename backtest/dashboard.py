#!/usr/bin/env python3
"""
Strategy 09 Backtest Dashboard — API Server (v2)
==================================================

Features:
  - Live backend log streaming via SSE (Server-Sent Events)
  - Auto-fetch missing symbol data from Google Drive
  - Editable symbol input with custom entries
  - Proper error handling and status reporting

Usage:
    python3 backtest/dashboard.py
    # Open http://localhost:8050
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "strategy_09_mss_ob_entry"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from backtest.data_loader import (
    load_all_timeframes,
    list_available_symbols,
    detect_market,
    DATA_DIR,
)
from backtest.backtester import run_backtest, BacktestResult, BacktestTrade

# ── Logging with SSE broadcast ────────────────────────────────────────

log_subscribers: list[queue.Queue] = []

class SSELogHandler(logging.Handler):
    """Custom log handler that broadcasts messages to all SSE subscribers."""
    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            event_data = json.dumps({"level": level, "message": msg, "time": record.created})
            dead = []
            for q in log_subscribers:
                try:
                    q.put_nowait(event_data)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                if q in log_subscribers:
                    log_subscribers.remove(q)
        except Exception:
            pass

# Setup logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
))
root_logger.addHandler(console_handler)

sse_handler = SSELogHandler()
sse_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
))
root_logger.addHandler(sse_handler)

logger = logging.getLogger("dashboard")

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Strategy 09 Backtester Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

app.add_middleware(NoCacheStaticMiddleware)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Track running backtests (sync POST or async worker)
backtest_running = False

# Background backtest state — avoids browser dropping long POST requests (>~10min idle).
_bt_async_lock = threading.Lock()
_bt_async: Dict[str, Any] = {"running": False, "result": None}

# ── Google Drive data fetcher ─────────────────────────────────────────

def fetch_symbol_from_drive(symbol: str) -> bool:
    """Download symbol data from Google Drive using service account."""
    creds_path = REPO_ROOT / "credentials.json"
    if not creds_path.exists():
        logger.warning(f"credentials.json not found — cannot auto-fetch {symbol}")
        return False

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import io
    except ImportError:
        logger.warning("google-api-python-client not installed — cannot auto-fetch")
        return False

    DRIVE_ROOT_ID = "1gJmnli48Y6KEolcNt_hphbKYzPU51Zer"
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    NEEDED_TFS = ["4h", "1h", "15m", "5m"]

    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(creds_path), scopes=SCOPES,
        )
        service = build("drive", "v3", credentials=credentials)

        def list_children(folder_id):
            items = []
            page_token = None
            while True:
                resp = service.files().list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, size)",
                    pageToken=page_token, pageSize=100,
                ).execute()
                items.extend(resp.get("files", []))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            return items

        def find_folder(parent_id, name):
            for c in list_children(parent_id):
                if c["name"].lower() == name.lower() and c["mimeType"] == "application/vnd.google-apps.folder":
                    return c["id"]
            return None

        def find_csv(folder_id):
            for c in list_children(folder_id):
                if c["name"].endswith(".csv"):
                    return c
            return None

        def download_file(file_id, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(str(dest), "wb")
            downloader = MediaIoBaseDownload(fh, request, chunksize=10*1024*1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.close()

        # Find symbol folder
        sym_folder_id = find_folder(DRIVE_ROOT_ID, symbol.upper())
        if not sym_folder_id:
            available = [c["name"] for c in list_children(DRIVE_ROOT_ID)
                         if c["mimeType"] == "application/vnd.google-apps.folder"]
            logger.error(f"Symbol {symbol} not found on Drive. Available: {', '.join(sorted(available))}")
            return False

        logger.info(f"📥 Downloading {symbol} data from Google Drive...")
        sym_dir = DATA_DIR / symbol.upper()
        sym_dir.mkdir(parents=True, exist_ok=True)

        for tf in NEEDED_TFS:
            tf_folder_id = find_folder(sym_folder_id, tf)
            if not tf_folder_id:
                logger.warning(f"  ⚠ {symbol}/{tf} folder not found on Drive")
                continue

            csv_file = find_csv(tf_folder_id)
            if not csv_file:
                logger.warning(f"  ⚠ No CSV in {symbol}/{tf}")
                continue

            dest = sym_dir / f"{tf}.csv"
            if dest.exists():
                logger.info(f"  ⊜ {tf}.csv already exists — skipping")
                continue

            logger.info(f"  ↓ Downloading {csv_file['name']} → {tf}.csv")
            download_file(csv_file["id"], dest)
            size_mb = dest.stat().st_size / 1024 / 1024
            logger.info(f"  ✓ {tf}.csv saved ({size_mb:.1f} MB)")

        logger.info(f"✅ {symbol} data download complete")
        return True

    except Exception as e:
        logger.error(f"Drive fetch failed for {symbol}: {e}")
        traceback.print_exc()
        return False


# ── Request/Response Models ───────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    enable_ema: bool = True
    enable_killzone: bool = True
    enable_london_block: bool = True
    enable_blacklist: bool = True
    min_quality: int = 50
    auto_fetch: bool = True  # Auto-fetch missing data from Drive
    fill_model: str = "tap_bar_close"  # tap_bar_close | strategy
    intrabar_htf_bias_mss: bool = True  # forming 4H for sweep timing; EMA stays closed 4H
    server_utc_offset: Optional[int] = None  # MT5 server UTC offset (3=Exness summer)
    verbose_logs: bool = False  # Per-bar INFO logs (floods SSE; slow on ALL symbols)


class TradeResponse(BaseModel):
    symbol: str
    market: str
    direction: str
    entry_time: Optional[str]
    entry_price: float
    sl_price: float
    tp_price: float
    sl_reason: str
    quality: int
    exit_time: Optional[str]
    exit_price: Optional[float]
    outcome: Optional[str]
    pnl_pct: float
    pnl_inr: float = 0.0
    rejected: bool
    reject_reason: str
    bias_reason: str
    ema_detail: str
    killzone: str
    # Phase timeline
    sweep_time: Optional[str] = None
    swept_level_time: Optional[str] = None
    mss_time: Optional[str] = None
    ob_time: Optional[str] = None
    tap_time: Optional[str] = None
    ob_top: Optional[float] = None
    ob_bottom: Optional[float] = None


class SymbolResult(BaseModel):
    symbol: str
    market: str
    signals_found: int
    signals_rejected: int
    trades_executed: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    total_pnl_inr: float = 0.0
    avg_pnl_pct: float
    max_drawdown: float
    trades: List[TradeResponse]
    rejection_breakdown: Dict[str, int]
    equity_curve: List[float]
    equity_curve_inr: List[float] = []


class BacktestResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    elapsed_sec: float = 0
    symbols_processed: int = 0
    results: List[SymbolResult] = []
    aggregate: Optional[Dict[str, Any]] = None


# ── Helpers ───────────────────────────────────────────────────────────

def _format_dt(dt) -> Optional[str]:
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def _calc_max_drawdown(pnl_list: list[float]) -> float:
    if not pnl_list:
        return 0.0
    cum, total, peak, max_dd = [], 0.0, 0.0, 0.0
    for p in pnl_list:
        total += p
        cum.append(total)
    for c in cum:
        if c > peak:
            peak = c
        dd = peak - c
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def _calc_equity_curve(trades: list[BacktestTrade]) -> list[float]:
    curve = [0.0]
    total = 0.0
    for t in trades:
        if not t.rejected:
            total += t.pnl_pct
            curve.append(round(total, 4))
    return curve


def _calc_equity_curve_inr(trades: list[BacktestTrade]) -> list[float]:
    curve = [0.0]
    total = 0.0
    for t in trades:
        if not t.rejected:
            total += t.pnl_inr
            curve.append(round(total, 2))
    return curve


def _rejection_breakdown(trades: list[BacktestTrade]) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for t in trades:
        if t.rejected:
            key = t.reject_reason.split(":")[0] if ":" in t.reject_reason else t.reject_reason
            reasons[key] = reasons.get(key, 0) + 1
    return dict(sorted(reasons.items(), key=lambda x: -x[1]))


def _build_symbol_result(result: BacktestResult, market: str) -> SymbolResult:
    executed = result.executed_trades
    wins = result.winning_trades
    losses = result.losing_trades
    pnl_list = [t.pnl_pct for t in executed]

    trades_resp = []
    for t in result.trades:
        trades_resp.append(TradeResponse(
            symbol=t.symbol, market=t.market, direction=t.direction,
            entry_time=_format_dt(t.entry_time),
            entry_price=round(t.entry_price, 6),
            sl_price=round(t.sl_price, 6),
            tp_price=round(t.tp_price, 6),
            sl_reason=t.sl_reason, quality=t.quality,
            exit_time=_format_dt(t.exit_time),
            exit_price=round(t.exit_price, 6) if t.exit_price else None,
            outcome=t.outcome, pnl_pct=round(t.pnl_pct, 4),
            pnl_inr=round(t.pnl_inr, 2),
            rejected=t.rejected, reject_reason=t.reject_reason,
            bias_reason=t.bias_reason, ema_detail=t.ema_detail,
            killzone=t.killzone,
            sweep_time=_format_dt(t.sweep_time),
            swept_level_time=_format_dt(t.swept_level_time),
            mss_time=_format_dt(t.mss_time),
            ob_time=_format_dt(t.ob_time),
            tap_time=_format_dt(t.tap_time),
            ob_top=round(t.ob_top, 6) if t.ob_top else None,
            ob_bottom=round(t.ob_bottom, 6) if t.ob_bottom else None,
        ))

    total_pnl_inr = round(sum(t.pnl_inr for t in executed), 2)

    return SymbolResult(
        symbol=result.symbol, market=market,
        signals_found=result.signals_found,
        signals_rejected=result.signals_rejected,
        trades_executed=len(executed),
        wins=len(wins), losses=len(losses),
        win_rate=round(result.win_rate, 2),
        total_pnl_pct=round(result.total_pnl_pct, 4),
        total_pnl_inr=total_pnl_inr,
        avg_pnl_pct=round(result.total_pnl_pct / len(executed), 4) if executed else 0.0,
        max_drawdown=_calc_max_drawdown(pnl_list),
        trades=trades_resp,
        rejection_breakdown=_rejection_breakdown(result.trades),
        equity_curve=_calc_equity_curve(result.trades),
        equity_curve_inr=_calc_equity_curve_inr(result.trades),
    )


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found. Check static/index.html</h1>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/symbols")
async def get_symbols():
    symbols = list_available_symbols()
    return {"symbols": symbols, "data_dir": str(DATA_DIR)}


@app.get("/api/logs")
async def stream_logs():
    """SSE endpoint for live log streaming."""
    log_queue: queue.Queue = queue.Queue(maxsize=500)
    log_subscribers.append(log_queue)

    async def event_generator():
        try:
            yield "data: {\"level\": \"INFO\", \"message\": \"Connected to log stream\", \"time\": 0}\n\n"
            while True:
                try:
                    msg = log_queue.get_nowait()
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.2)
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if log_queue in log_subscribers:
                log_subscribers.remove(log_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/backtest/start")
async def start_backtest_async(req: BacktestRequest):
    """Kick off a backtest in a daemon thread and return immediately.

    Poll ``GET /api/backtest/status`` until ``running`` is false. Using this avoids
    ``fetch()`` failures when a single POST stays open for many minutes (sleep,
    browser limits, or network idle timeouts)."""
    global backtest_running

    try:
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

    with _bt_async_lock:
        if _bt_async["running"] or backtest_running:
            raise HTTPException(status_code=409, detail="A backtest is already running.")
        _bt_async["running"] = True
        _bt_async["result"] = None

    backtest_running = True

    def worker() -> None:
        global backtest_running
        try:
            res = _run_backtest_sync(req, start_dt, end_dt)
            with _bt_async_lock:
                _bt_async["result"] = res
        except Exception as e:
            logger.exception("Background backtest failed")
            with _bt_async_lock:
                _bt_async["result"] = BacktestResponse(success=False, error=str(e))
        finally:
            backtest_running = False
            with _bt_async_lock:
                _bt_async["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}


@app.get("/api/backtest/status")
async def backtest_async_status():
    """Poll after ``/api/backtest/start`` — ``result`` is set when ``running`` is false."""
    with _bt_async_lock:
        running = bool(_bt_async["running"])
        result = _bt_async["result"]
    payload: Dict[str, Any] = {"running": running}
    if result is not None:
        payload["result"] = result.model_dump()
    return payload


@app.post("/api/backtest", response_model=BacktestResponse)
async def run_backtest_api(req: BacktestRequest):
    global backtest_running
    if backtest_running:
        return BacktestResponse(success=False, error="A backtest is already running. Please wait.")

    backtest_running = True

    try:
        start_dt = datetime.strptime(req.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    except ValueError as e:
        backtest_running = False
        return BacktestResponse(success=False, error=f"Invalid date format: {e}")

    if start_dt >= end_dt:
        backtest_running = False
        return BacktestResponse(success=False, error="Start date must be before end date")

    # Run heavy computation in a thread so SSE logs can stream in real-time
    try:
        result = await asyncio.to_thread(
            _run_backtest_sync, req, start_dt, end_dt
        )
        return result
    finally:
        backtest_running = False


def _run_backtest_sync(
    req: BacktestRequest,
    start_dt: datetime,
    end_dt: datetime,
) -> BacktestResponse:
    """Blocking backtest logic — runs in a thread pool."""
    t0 = time.time()

    try:
        # Determine symbols
        if req.symbol.upper() == "ALL":
            symbols = list_available_symbols()
            if not symbols:
                return BacktestResponse(success=False, error="No data found. Download data first.")
        else:
            sym = req.symbol.upper()
            available = list_available_symbols()
            if sym not in available:
                if req.auto_fetch:
                    logger.info(f"🔍 {sym} not found locally — attempting Drive download...")
                    fetched = fetch_symbol_from_drive(sym)
                    if fetched:
                        available = list_available_symbols()
                    if sym not in available:
                        return BacktestResponse(
                            success=False,
                            error=f"Symbol '{sym}' not available. Could not fetch from Drive. "
                                  f"Available locally: {', '.join(available) if available else 'none'}",
                        )
                else:
                    return BacktestResponse(
                        success=False,
                        error=f"Symbol '{sym}' not found. Available: {', '.join(available) if available else 'none'}",
                    )
            symbols = [sym]

        results: list[SymbolResult] = []
        errors: list[str] = []

        for symbol in symbols:
            market = detect_market(symbol)
            logger.info(f"═══ Running backtest: {symbol} ({market}) {req.start_date} → {req.end_date} ═══")

            try:
                data = load_all_timeframes(symbol, start_dt, end_dt,
                                           server_utc_offset=req.server_utc_offset)
                missing = [tf for tf, df in data.items() if df is None]
                if missing:
                    errors.append(f"{symbol}: Missing timeframes {', '.join(missing)}")
                    continue

                fm = req.fill_model if req.fill_model in ("tap_bar_close", "strategy") else "tap_bar_close"
                bt_result = run_backtest(
                    symbol=symbol,
                    df_4h=data["4h"], df_1h=data["1h"],
                    df_15m=data["15m"], df_5m=data["5m"],
                    start=start_dt, end=end_dt,
                    market=market,
                    min_quality=req.min_quality,
                    fill_model=fm,
                    intrabar_htf_bias_mss=req.intrabar_htf_bias_mss,
                    enable_ema_filter=req.enable_ema,
                    enable_killzone=req.enable_killzone,
                    enable_london_block=req.enable_london_block,
                    enable_blacklist=req.enable_blacklist,
                    verbose=req.verbose_logs,
                )

                results.append(_build_symbol_result(bt_result, market))
                logger.info(
                    f"  ✅ {symbol}: {len(bt_result.executed_trades)} trades, "
                    f"{bt_result.win_rate:.1f}% WR, {bt_result.total_pnl_pct:+.3f}% PnL"
                )
            except Exception as e:
                logger.error(f"  ❌ {symbol}: {e}")
                traceback.print_exc()
                errors.append(f"{symbol}: {type(e).__name__}: {e}")

        elapsed = time.time() - t0

        # Aggregate
        aggregate = None
        if len(results) > 1:
            total_trades = sum(r.trades_executed for r in results)
            total_wins = sum(r.wins for r in results)
            combined_pnl = sum(r.total_pnl_pct for r in results)
            combined_pnl_inr = sum(r.total_pnl_inr for r in results)
            aggregate = {
                "symbols_tested": len(results),
                "total_signals": sum(r.signals_found for r in results),
                "total_rejected": sum(r.signals_rejected for r in results),
                "total_trades": total_trades,
                "total_wins": total_wins,
                "total_losses": sum(r.losses for r in results),
                "win_rate": round(total_wins / total_trades * 100, 2) if total_trades else 0,
                "combined_pnl_pct": round(combined_pnl, 4),
                "combined_pnl_inr": round(combined_pnl_inr, 2),
                "avg_pnl_per_trade": round(combined_pnl / total_trades, 4) if total_trades else 0,
                "avg_inr_per_trade": round(combined_pnl_inr / total_trades, 2) if total_trades else 0,
            }

        error_msg = "; ".join(errors) if errors else None
        logger.info(f"═══ Backtest complete in {elapsed:.1f}s — {len(results)} symbols processed ═══")

        return BacktestResponse(
            success=len(results) > 0, error=error_msg,
            elapsed_sec=round(elapsed, 2),
            symbols_processed=len(results),
            results=results, aggregate=aggregate,
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        traceback.print_exc()
        return BacktestResponse(success=False, error=str(e))


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  ══════════════════════════════════════════════════")
    print("    Strategy 09 Backtest Dashboard")
    print("    http://localhost:8050")
    print("  ══════════════════════════════════════════════════\n")
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="warning")

