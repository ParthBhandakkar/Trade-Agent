"""
Check SELL confirmation filter against real 1H data for every SELL trade.
Uses TvDatafeed directly (no multiprocessing) to fetch real OANDA 1H data.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.utils.env_loader import load_root_env
load_root_env(REPO_ROOT)

import time
import pandas as pd
import pytz
import requests

UTC = pytz.UTC
TV_SOURCE_TZ = pytz.timezone(os.getenv("TV_SOURCE_TZ", "Asia/Kolkata").strip() or "Asia/Kolkata")

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    print("tvDatafeed not installed"); sys.exit(1)


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def resolve_auth_token(session_id):
    import re
    try:
        session = requests.Session()
        session.cookies.set('sessionid', session_id, domain='.tradingview.com')
        resp = session.get(
            'https://www.tradingview.com/chart/',
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=30,
        )
        m = re.search(r'"auth_token":"([^"]+)"', resp.text)
        return m.group(1) if m else None
    except Exception:
        return None


def fetch_1h(tv, symbol, n_bars=500):
    """Fetch 1H OANDA data directly (no subprocess)."""
    for attempt in range(1, 4):
        try:
            df = tv.get_hist(symbol=symbol, exchange="OANDA",
                             interval=Interval.in_1_hour, n_bars=n_bars)
            if df is not None and not df.empty:
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize(TV_SOURCE_TZ)
                df.index = df.index.tz_convert(UTC)
                return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"    Attempt {attempt} failed: {e}")
            time.sleep(2)
    return None


# ── All SELL trades ────────────────────────────────────────────
sell_trades = [
    {"ticket": 1042640024, "time": "2026-04-02T09:08:22", "symbol": "NZDCHF", "profit": 501.63},
    {"ticket": 1044801344, "time": "2026-04-02T13:57:21", "symbol": "AUDJPY", "profit": 500.78},
    {"ticket": 1045286518, "time": "2026-04-02T14:42:23", "symbol": "AUDCAD", "profit": -882.97},
    {"ticket": 1045287991, "time": "2026-04-02T14:42:32", "symbol": "NZDCAD", "profit": 494.38},
    {"ticket": 1059472654, "time": "2026-04-07T08:08:48", "symbol": "EURUSD", "profit": -2453.48},
    {"ticket": 1059472893, "time": "2026-04-07T08:08:54", "symbol": "GBPUSD", "profit": -2540.68},
    {"ticket": 1059481827, "time": "2026-04-07T08:11:08", "symbol": "EURUSD", "profit": -1752.73},
    {"ticket": 1059483729, "time": "2026-04-07T08:11:17", "symbol": "GBPUSD", "profit": -1743.63},
    {"ticket": 1074951684, "time": "2026-04-09T08:27:55", "symbol": "GBPAUD", "profit": 1002.34},
    {"ticket": 1107638691, "time": "2026-04-16T07:40:22", "symbol": "USDCHF", "profit": -2982.99},
    {"ticket": 1107825244, "time": "2026-04-16T08:23:56", "symbol": "USDJPY", "profit": -2817.25},
    {"ticket": 1107825857, "time": "2026-04-16T08:24:08", "symbol": "USDJPY", "profit": -2370.37},
    {"ticket": 1108084641, "time": "2026-04-16T09:33:14", "symbol": "EURGBP", "profit": 1993.69},
    {"ticket": 1108097633, "time": "2026-04-16T09:36:10", "symbol": "EURGBP", "profit": 998.95},
    {"ticket": 1109847843, "time": "2026-04-16T14:30:36", "symbol": "EURGBP", "profit": -409.62},
    {"ticket": 1114144800, "time": "2026-04-17T10:47:41", "symbol": "GBPNZD", "profit": 5171.79},
]

if __name__ == "__main__":
    # Connect
    sid = os.environ.get("TV_SESSION_TOKEN", "").strip()
    if sid:
        token = resolve_auth_token(sid)
        tv = TvDatafeed()
        if token:
            tv.token = token
            print("Authenticated with TradingView")
    else:
        tv = TvDatafeed()
        print("Using TvDatafeed nologin mode")

    # Fetch data per symbol
    symbols_needed = sorted(set(t["symbol"] for t in sell_trades))
    symbol_data = {}

    print(f"\nFetching 1H data for {len(symbols_needed)} symbols...\n")
    for sym in symbols_needed:
        print(f"  {sym}...", end=" ", flush=True)
        df = fetch_1h(tv, sym, 500)
        if df is not None:
            symbol_data[sym] = df
            print(f"OK ({len(df)} bars, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        else:
            print("FAILED")
        time.sleep(1)

    # ── Results ────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  SELL CONFIRMATION FILTER - REAL DATA RESULTS")
    print(f"  Rule: 1H Close must be BELOW EMA21 to allow a SELL trade")
    print(f"{'=' * 100}")
    print(f"\n  {'Ticket':<12} {'Date':>16} {'Sym':<8} {'P/L':>10} {'1H Close':>11} {'EMA21':>11} {'Gap':>8} {'Filter':>7} {'Verdict'}")
    print(f"  {'-' * 95}")

    winners_blocked = 0
    winners_kept = 0
    losses_blocked = 0
    losses_kept = 0
    loss_saved = 0.0
    win_lost = 0.0

    for t in sell_trades:
        sym = t["symbol"]
        trade_time = pd.Timestamp(t["time"]).tz_localize(UTC)

        if sym not in symbol_data:
            print(f"  {t['ticket']:<12} {t['time'][5:16]:>16} {sym:<8} Rs{t['profit']:>8,.0f}   {'NO DATA':>11} {'':>11} {'':>8} {'???':>7}")
            continue

        df_1h = symbol_data[sym]
        mask = df_1h.index <= trade_time
        if mask.sum() < 25:
            print(f"  {t['ticket']:<12} {t['time'][5:16]:>16} {sym:<8} Rs{t['profit']:>8,.0f}   {'FEW BARS':>11} {'':>11} {'':>8} {'FAIL':>7} Insufficient data")
            losses_blocked += 1 if t["profit"] < 0 else 0
            winners_blocked += 1 if t["profit"] > 0 else 0
            continue

        df_before = df_1h[mask]
        close = df_before["close"]
        ema21 = calculate_ema(close, 21)

        last_close = float(close.iloc[-1])
        last_ema21 = float(ema21.iloc[-1])
        gap = last_close - last_ema21

        passed = last_close < last_ema21
        filt = "PASS" if passed else "FAIL"

        if passed and t["profit"] > 0:
            verdict = "KEPT (winner)"
            winners_kept += 1
        elif passed and t["profit"] < 0:
            verdict = "KEPT (still loses)"
            losses_kept += 1
        elif not passed and t["profit"] > 0:
            verdict = "BLOCKED (lost winner!)"
            winners_blocked += 1
            win_lost += t["profit"]
        else:
            verdict = "BLOCKED (saved loss!)"
            losses_blocked += 1
            loss_saved += abs(t["profit"])

        print(f"  {t['ticket']:<12} {t['time'][5:16]:>16} {sym:<8} Rs{t['profit']:>8,.0f}  {last_close:>11.5f} {last_ema21:>11.5f} {gap:>+8.5f} {filt:>7} {verdict}")

    print(f"\n  {'-' * 95}")
    print(f"  SUMMARY:")
    print(f"    Winners KEPT:    {winners_kept}")
    print(f"    Winners BLOCKED: {winners_blocked}  (Rs {win_lost:,.0f} given up)")
    print(f"    Losses BLOCKED:  {losses_blocked}  (Rs {loss_saved:,.0f} saved)")
    print(f"    Losses KEPT:     {losses_kept}")
    print(f"    NET IMPACT:      Rs {loss_saved - win_lost:+,.0f}")
    print(f"{'=' * 100}")
