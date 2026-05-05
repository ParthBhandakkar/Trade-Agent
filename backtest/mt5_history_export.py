#!/usr/bin/env python3
"""
Fetch historical OHLCV data for backtesting.

Downloads data from Google Drive (Exness export) into
  backtest/data/<SYMBOL>/<tf>.csv

Usage:
  # List available symbols on Drive
  python3 backtest/mt5_history_export.py --list

  # Download a single symbol (all timeframes)
  python3 backtest/mt5_history_export.py --symbols EURUSD

  # Download multiple symbols
  python3 backtest/mt5_history_export.py --symbols EURUSD,GBPUSD,XAUUSD

  # Download ALL available symbols from Drive
  python3 backtest/mt5_history_export.py --symbols all

  # Specify timeframes
  python3 backtest/mt5_history_export.py --symbols USDJPY --timeframes 4h,1h,15m,5m

  # Custom output directory
  python3 backtest/mt5_history_export.py --symbols USDJPY --output-dir ./my_data

Requires:
  pip3 install google-api-python-client google-auth
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_OUT_DIR = ROOT / "data"
CREDENTIALS_PATH = REPO_ROOT / "credentials.json"

DRIVE_ROOT_FOLDER_ID = "1gJmnli48Y6KEolcNt_hphbKYzPU51Zer"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

ALL_TIMEFRAMES = ["4h", "1h", "15m", "5m"]


# ── Drive helpers ─────────────────────────────────────────────────────

def _get_drive_service(credentials_path: Path):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    if not credentials_path.exists():
        print(f"  ✗ Credentials not found: {credentials_path}")
        print(f"    Place your service account JSON at: {credentials_path}")
        sys.exit(1)

    creds = service_account.Credentials.from_service_account_file(
        str(credentials_path), scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds)


def _list_children(service, folder_id: str) -> list[dict]:
    items, page_token = [], None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token, pageSize=200,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _find_folder(service, parent_id: str, name: str) -> Optional[str]:
    for c in _list_children(service, parent_id):
        if c["name"].lower() == name.lower() and "folder" in c["mimeType"]:
            return c["id"]
    return None


def _find_csv(service, folder_id: str) -> Optional[dict]:
    for c in _list_children(service, folder_id):
        if c["name"].endswith(".csv"):
            return c
    return None


def _download_file(service, file_id: str, dest: Path):
    from googleapiclient.http import MediaIoBaseDownload

    dest.parent.mkdir(parents=True, exist_ok=True)
    fh = io.FileIO(str(dest), "wb")
    downloader = MediaIoBaseDownload(fh, service.files().get_media(fileId=file_id),
                                     chunksize=10 * 1024 * 1024)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"      ↓ {int(status.progress() * 100)}%", end="\r")
    fh.close()


def _list_drive_symbols(service) -> list[str]:
    children = _list_children(service, DRIVE_ROOT_FOLDER_ID)
    return sorted(c["name"] for c in children if "folder" in c["mimeType"])


# ── Main logic ────────────────────────────────────────────────────────

def download_symbol(
    service,
    symbol: str,
    timeframes: list[str],
    out_dir: Path,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """Download all timeframes for a symbol. Returns (downloaded, skipped)."""
    sym = symbol.upper()
    downloaded, skipped = 0, 0

    sym_folder_id = _find_folder(service, DRIVE_ROOT_FOLDER_ID, sym)
    if not sym_folder_id:
        print(f"  ✗ {sym} — not found on Drive")
        return 0, 0

    sym_dir = out_dir / sym
    sym_dir.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        dest = sym_dir / f"{tf}.csv"

        # Skip if exists
        if skip_existing and dest.exists() and dest.stat().st_size > 100:
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"    ⊜ {tf}.csv already exists ({size_mb:.1f} MB) — skipped")
            skipped += 1
            continue

        tf_folder_id = _find_folder(service, sym_folder_id, tf)
        if not tf_folder_id:
            print(f"    ⚠ {sym}/{tf} — folder not found on Drive")
            continue

        csv_file = _find_csv(service, tf_folder_id)
        if not csv_file:
            print(f"    ⚠ {sym}/{tf} — no CSV file found")
            continue

        print(f"    ↓ {csv_file['name']} → {tf}.csv ...")
        _download_file(service, csv_file["id"], dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"    ✓ {tf}.csv saved ({size_mb:.1f} MB)")
        downloaded += 1

    return downloaded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from Google Drive for backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           # Show available symbols
  %(prog)s --symbols EURUSD                 # Download EURUSD
  %(prog)s --symbols EURUSD,GBPUSD,XAUUSD  # Download multiple
  %(prog)s --symbols all                    # Download everything
  %(prog)s --symbols USDJPY --timeframes 5m,15m
  %(prog)s --symbols USDJPY --skip-existing
        """,
    )
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols to download, or 'all' for everything.",
    )
    parser.add_argument(
        "--timeframes", default=",".join(ALL_TIMEFRAMES),
        help=f"Comma-separated timeframes (default: {','.join(ALL_TIMEFRAMES)}).",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--credentials", default=str(CREDENTIALS_PATH),
        help="Path to Google service account JSON.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available symbols on Drive and exit.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip CSVs that already exist locally.",
    )
    args = parser.parse_args()

    creds_path = Path(args.credentials)
    out_dir = Path(args.output_dir)

    print(f"\n  ═══ Historical Data Fetcher (Google Drive) ═══\n")

    service = _get_drive_service(creds_path)

    # ── List mode ──
    if args.list:
        drive_symbols = _list_drive_symbols(service)
        print(f"  Available on Drive ({len(drive_symbols)} symbols):")
        for s in drive_symbols:
            print(f"    • {s}")

        # Show local data
        if out_dir.exists():
            local = sorted(
                d.name for d in out_dir.iterdir()
                if d.is_dir() and not d.name.startswith((".", "_"))
                and d.name.isalpha() and d.name == d.name.upper()
            )
            if local:
                print(f"\n  Downloaded locally ({len(local)} symbols):")
                for s in local:
                    csvs = list((out_dir / s).glob("*.csv"))
                    tfs = ", ".join(sorted(f.stem for f in csvs))
                    print(f"    • {s}: {tfs}")
        print()
        return 0

    # ── Download mode ──
    if not args.symbols:
        parser.error("--symbols is required (or use --list)")

    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    if args.symbols.lower() == "all":
        symbols = _list_drive_symbols(service)
        print(f"  Downloading ALL {len(symbols)} symbols...")
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Output:     {out_dir}\n")

    total_dl, total_sk = 0, 0
    for sym in symbols:
        print(f"  [{sym}]")
        dl, sk = download_symbol(service, sym, timeframes, out_dir, args.skip_existing)
        total_dl += dl
        total_sk += sk
        print()

    print(f"  ═══ Done: {total_dl} files downloaded, {total_sk} skipped ═══\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
