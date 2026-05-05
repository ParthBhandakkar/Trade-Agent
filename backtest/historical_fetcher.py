#!/usr/bin/env python3
"""
Historical Data Fetcher — Google Drive (Service Account)
==========================================================

Downloads OHLCV CSV files from the shared Exness Google Drive folder
into backtest/data/<SYMBOL>/<timeframe>.csv using a service account.

Drive structure:
    Exness/  (root shared folder)
      USDJPY/  EURUSD/  GBPUSD/  ...
        └── 5m/  15m/  1h/  4h/  ...
              └── <SYMBOL>_<tf>_<start>_<end>.csv

Usage:
    python3 backtest/historical_fetcher.py --symbol USDJPY
    python3 backtest/historical_fetcher.py --symbol USDJPY --timeframes 4h 1h
    python3 backtest/historical_fetcher.py --list

Requires:
    pip3 install google-api-python-client google-auth google-auth-httplib2
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Configuration ─────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = REPO_ROOT / "credentials.json"
DATA_DIR = Path(__file__).resolve().parent / "data"

DRIVE_ROOT_FOLDER_ID = "1gJmnli48Y6KEolcNt_hphbKYzPU51Zer"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
NEEDED_TIMEFRAMES = ["4h", "1h", "15m", "5m"]


# ── Google Drive Client ──────────────────────────────────────────────

def get_drive_service(credentials_path: Path | None = None):
    """Authenticate with the service account and return a Drive client."""
    creds_path = credentials_path or CREDENTIALS_PATH
    if not creds_path.exists():
        print(f"  ❌ Credentials file not found: {creds_path}")
        print(f"     Place your service_credentials.json at {creds_path}")
        sys.exit(1)

    credentials = service_account.Credentials.from_service_account_file(
        str(creds_path), scopes=SCOPES,
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def list_folder_children(service, folder_id: str) -> list[dict]:
    """List all files/folders inside a Drive folder."""
    items = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
            pageSize=100,
        ).execute()

        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items


def download_file(service, file_id: str, dest_path: Path):
    """Download a file from Drive to a local path."""
    request = service.files().get_media(fileId=file_id)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    fh = io.FileIO(str(dest_path), "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"    ↓ {pct}%", end="\r")
    fh.close()
    print(f"    ✓ Saved: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")


def find_folder_by_name(service, parent_id: str, name: str) -> Optional[str]:
    """Find a subfolder by name inside a parent folder. Returns folder ID."""
    children = list_folder_children(service, parent_id)
    for child in children:
        if (child["name"].lower() == name.lower()
                and child["mimeType"] == "application/vnd.google-apps.folder"):
            return child["id"]
    return None


def find_csv_in_folder(service, folder_id: str) -> Optional[dict]:
    """Find the first CSV file in a folder."""
    children = list_folder_children(service, folder_id)
    for child in children:
        if child["name"].endswith(".csv"):
            return child
    return None


# ── Main Download Logic ──────────────────────────────────────────────

def download_symbol(
    service,
    symbol: str,
    timeframes: list[str] | None = None,
    data_dir: Path | None = None,
):
    """Download all needed timeframes for a symbol."""
    tfs = timeframes or NEEDED_TIMEFRAMES
    sym = symbol.upper()
    out_dir = (data_dir or DATA_DIR) / sym
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find the symbol folder inside root
    print(f"  Looking for {sym} in Drive...")
    sym_folder_id = find_folder_by_name(service, DRIVE_ROOT_FOLDER_ID, sym)
    if not sym_folder_id:
        print(f"  ❌ Symbol folder '{sym}' not found in Drive!")
        root_children = list_folder_children(service, DRIVE_ROOT_FOLDER_ID)
        available = [c["name"] for c in root_children
                     if c["mimeType"] == "application/vnd.google-apps.folder"]
        print(f"     Available symbols: {', '.join(sorted(available))}")
        return False

    print(f"  ✓ Found {sym} folder")

    # Step 2: For each timeframe, find the subfolder and download the CSV
    for tf in tfs:
        print(f"\n  [{sym}/{tf}]")
        tf_folder_id = find_folder_by_name(service, sym_folder_id, tf)
        if not tf_folder_id:
            print(f"    ⚠ Timeframe folder '{tf}' not found — skipping")
            continue

        csv_file = find_csv_in_folder(service, tf_folder_id)
        if not csv_file:
            print(f"    ⚠ No CSV file found in {tf} folder — skipping")
            continue

        # Download and save with canonical name
        original_name = csv_file["name"]
        dest = out_dir / f"{tf}.csv"

        # Skip if already downloaded and same size
        if dest.exists():
            existing_size = dest.stat().st_size
            remote_size = int(csv_file.get("size", 0))
            if remote_size > 0 and abs(existing_size - remote_size) < 1024:
                print(f"    ⊜ Already exists ({existing_size / 1024 / 1024:.1f} MB) — skipping")
                continue

        print(f"    Downloading {original_name}...")
        download_file(service, csv_file["id"], dest)

    return True


def list_available_symbols(service) -> list[str]:
    """List all symbol folders in the Drive root."""
    children = list_folder_children(service, DRIVE_ROOT_FOLDER_ID)
    return sorted(
        c["name"] for c in children
        if c["mimeType"] == "application/vnd.google-apps.folder"
    )


def list_local_data(data_dir: Path | None = None):
    """Show what data is already downloaded."""
    base = data_dir or DATA_DIR
    if not base.exists():
        print("  No data directory found.")
        return

    found = False
    for sym_dir in sorted(base.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name.startswith((".", "_")):
            continue
        csvs = list(sym_dir.glob("*.csv"))
        if csvs:
            found = True
            print(f"  {sym_dir.name}:")
            for f in sorted(csvs, key=lambda x: x.stem):
                size_mb = f.stat().st_size / 1024 / 1024
                # Count rows (fast: just count newlines)
                with open(f) as fh:
                    rows = sum(1 for _ in fh) - 1  # subtract header
                print(f"    {f.stem:>4s}.csv  {size_mb:6.1f} MB  ({rows:,} bars)")
        else:
            found = True
            print(f"  {sym_dir.name}: (empty)")

    if not found:
        print("  No data found. Use --symbol to download.")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download historical data from Google Drive for backtesting",
    )
    parser.add_argument("--symbol", type=str, help="Symbol to download (e.g. USDJPY)")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Timeframes (default: 4h 1h 15m 5m)")
    parser.add_argument("--list", action="store_true",
                        help="List available symbols on Drive and local data")
    parser.add_argument("--credentials", type=str, default=None,
                        help="Path to service account JSON")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom output data directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    creds = Path(args.credentials) if args.credentials else None

    print(f"\n  ══ Historical Data Fetcher (Service Account) ══\n")

    service = get_drive_service(creds)

    if args.list:
        print("  ── Drive Symbols ──")
        symbols = list_available_symbols(service)
        print(f"  Available on Drive: {', '.join(symbols)}")
        print(f"\n  ── Local Data ──")
        list_local_data(data_dir)
        print()
        return 0

    if not args.symbol:
        parser.error("--symbol is required (or use --list)")

    symbol = args.symbol.upper()
    target = (data_dir or DATA_DIR) / symbol
    print(f"  Symbol:  {symbol}")
    print(f"  Target:  {target}")
    print(f"  TFs:     {', '.join(args.timeframes or NEEDED_TIMEFRAMES)}\n")

    ok = download_symbol(service, symbol, args.timeframes, data_dir)
    if not ok:
        return 1

    print(f"\n  ── Local Data ──")
    list_local_data(data_dir)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
