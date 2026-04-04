# Strategy 09 Google Sheets Server

This service runs the existing Strategy 09 signal pipeline on a Linux VM and writes each fresh trade signal into a Google Sheet instead of sending orders through MT5 or Binance.

## What it does

- Reuses the current Strategy 09 flow:
  - 4H bias
  - 1H MSS confirmation
  - 15M order block
  - 5M tap detection
- Keeps the important live filters:
  - quality threshold
  - killzone filter
  - strict 4H EMA filter
  - crypto blacklist
- Computes the same smart SL and 1.5R TP
- Appends every valid signal to:
  - Google Sheets
  - local JSONL logs in `strategy_09_sheet_server/logs/`
- Exposes lightweight health endpoints with FastAPI

## Sheet columns

The worksheet is auto-created with headers like:

- `record_id`
- `signal_time_ist`
- `market`
- `symbol`
- `direction`
- `entry_price`
- `stop_loss`
- `take_profit`
- `quality_score`
- `sl_reason`
- `bias_*`
- `mss_*`
- `ob_*`
- `raw_signal_json`

## Setup

1. Install Python packages:

```bash
pip install -r strategy_09_sheet_server/requirements.txt
pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
```

2. Copy the env template:

```bash
cp strategy_09_sheet_server/.env.example strategy_09_sheet_server/.env
```

3. Create a Google service account and download its JSON key.

4. Share your target Google Sheet with the service account email.

5. Fill these in `strategy_09_sheet_server/.env`:

```ini
GOOGLE_SHEET_ID=your_sheet_id
GOOGLE_WORKSHEET_TITLE=trade_signals
GOOGLE_SERVICE_ACCOUNT_FILE=/home/ubuntu/yt_learning/strategy_09_sheet_server/google-service-account.json
TV_SOURCE_TZ=Asia/Kolkata
TV_USERNAME=...
TV_PASSWORD=...
```

`TV_SOURCE_TZ=Asia/Kolkata` keeps the hosted server and the local `auto_trader` aligned on IST-based source timestamp handling.

## Run locally

```bash
uvicorn strategy_09_sheet_server.server:app --host 0.0.0.0 --port 8010 --workers 1
```

Use a single worker only. Multiple workers would start multiple polling loops.

## Endpoints

- `GET /health`
- `GET /status`

## Linux service

The systemd unit is at:

- `strategy_09_sheet_server/deploy/strategy09-sheet-server.service`

Typical install:

```bash
sudo cp strategy_09_sheet_server/deploy/strategy09-sheet-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable strategy09-sheet-server
sudo systemctl start strategy09-sheet-server
sudo systemctl status strategy09-sheet-server
```

## Notes

- Signals are deduplicated with `strategy_09_sheet_server/state/sent_signals.json`.
- Rejected signals are still written locally to `logs/trades.jsonl`, but only valid trade signals are appended to Google Sheets.
- This service assumes a single worksheet named by `GOOGLE_WORKSHEET_TITLE`. If you want separate tabs for forex and crypto, the row mapper can be split easily.
