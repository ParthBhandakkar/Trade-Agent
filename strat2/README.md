# Strat2

Standalone MT5 forex strategy. It does not import anything from the other
strategy folders.

## Model

Strat2 is a strict trend-continuation model:

1. Daily and 4H trend must align.
2. 15M price must sweep liquidity against that trend.
3. 5M price must break structure back in the trend direction.
4. A 5M fair-value gap must form.
5. The latest closed 5M candle must retrace into that FVG and reject.

It uses loss-based sizing, not margin-based sizing.

## Files

- `auto_trader.py`: full bot, strategy, data fetch, execution, and management.
- `.env`: MT5 connection plus Strat2 settings.
- `logs/`: JSONL logs created at runtime.
- `state/`: sent-signal dedup state.

## Run

Dry run one scan:

```powershell
python strat2\auto_trader.py --once
```

Dry run continuously:

```powershell
python strat2\auto_trader.py
```

Live mode:

```powershell
python strat2\auto_trader.py --live
```

The `.env` file defaults to dry run. Keep it that way until the logs show the
signals are behaving as expected.
