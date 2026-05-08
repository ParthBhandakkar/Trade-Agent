from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACKTEST_DIR = Path(__file__).resolve().parent
STATIC_DIR = BACKTEST_DIR / "static"
REPO_ROOT = BACKTEST_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strat2.auto_trader import DEFAULT_FOREX_PAIRS  # noqa: E402
from strat2.backtest.engine import BacktestConfig, parse_date_ist, run_backtest, symbols_from_text  # noqa: E402


class BacktestRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: DEFAULT_FOREX_PAIRS[:6])
    start: str
    end: str
    risk_inr: float = 500.0
    min_quality: int = 78
    force_refresh: bool = False


app = FastAPI(title="Strat2 Backtester")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/defaults")
def defaults() -> dict:
    return {
        "symbols": DEFAULT_FOREX_PAIRS,
        "default_symbols": DEFAULT_FOREX_PAIRS[:6],
        "risk_inr": 500,
        "min_quality": 78,
    }


@app.post("/api/backtest")
def backtest(req: BacktestRequest) -> dict:
    symbols = symbols_from_text(",".join(req.symbols))
    if not symbols:
        raise HTTPException(status_code=400, detail="Select at least one symbol")
    try:
        start_utc = parse_date_ist(req.start)
        end_utc = parse_date_ist(req.end, end_of_day=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if end_utc <= start_utc:
        raise HTTPException(status_code=400, detail="End date must be after start date")
    try:
        return run_backtest(BacktestConfig(
            symbols=symbols,
            start_utc=start_utc,
            end_utc=end_utc,
            risk_inr=req.risk_inr,
            min_quality=req.min_quality,
            force_refresh=req.force_refresh,
        ))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

