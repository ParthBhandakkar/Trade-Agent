# Strat2 Backtester

This folder contains a self-contained historical backtester for `strat2`.

It reuses the live strategy logic from `strat2/auto_trader.py`, fetches the
required MT5 timeframes when cache files are missing, and then replays closed
5-minute candles from the selected date range.

## Web UI

```powershell
.\.venv\Scripts\python.exe -m uvicorn strat2.backtest.server:app --reload --port 8092
```

Open:

```text
http://127.0.0.1:8092
```

## CLI

```powershell
.\.venv\Scripts\python.exe strat2\backtest\engine.py --symbols EURUSD,GBPUSD --start 2026-05-01 --end 2026-05-08
```

Cached candles are stored under:

```text
strat2/backtest/data/<SYMBOL>/<SYMBOL>_<TIMEFRAME>.csv
```

If a cache file already covers the requested range, the backtester uses it
offline. If the range is missing, it connects to MT5 and merges newly fetched
history into the cache.
